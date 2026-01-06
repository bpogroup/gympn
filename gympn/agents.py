"""Policy gradient agents that support changing state spaces, specifically for graph environments.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy optimization
agent.
"""
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from gympn.data import TrajectoryBuffer, print_status_bar
from gympn.logging_utils import Logger, TrainingMetrics, TestMetrics, get_logger

#torch.autograd.set_detect_anomaly(True)




class Agent:
    """Base class for policy gradient agents.

    All functionality for policy gradient is implemented in this
    class. Derived classes must define the property `policy_loss`
    which is used to train the policy.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    policy_lr : float, optional
        The learning rate for the policy model.
    policy_updates : int, optional
        The number of policy updates per epoch of training.
    value_network : network, None, or string, optional
        The network for the value model.
    value_lr : float, optional
        The learning rate for the value model.
    value_updates : int, optional
        The number of value updates per epoch of training.
    gam : float, optional
        The discount rate.
    lam : float, optional
        The parameter for generalized advantage estimation.
    normalize_advantages : bool, optional
        Whether to normalize advantages.
    kld_limit : float, optional
        The limit on KL divergence for early stopping policy updates.
    ent_bonus : float, optional
        Bonus factor for sampled policy entropy.

    """

    def __init__(self,
                 policy_network, value_network, policy_lr=1e-4, policy_updates=1,
                 value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=0.01, ent_bonus=0.01, test_in_train=True, vf_coeff=0.5,
                 normalize_returns=True, lr_schedule=True):
        self.policy_model = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = torch.optim.Adam(params=list(policy_network.parameters()),
                                                 lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = torch.nn.MSELoss()
        self.value_optimizer = torch.optim.Adam(params=list(value_network.parameters()), lr=value_lr)
        self.value_updates = value_updates

        self.lam = lam
        self.gam = gam
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam)
        self.normalize_advantages = normalize_advantages
        self.normalize_returns = normalize_returns  # New parameter
        self.lr_schedule = lr_schedule  # New parameter
        self.kld_limit = kld_limit
        self.ent_bonus = ent_bonus

        self.previous_policy_loss = 0
        self.best_test_metric = float('-inf')  # Initialize the best test metric

        self.test_during_train = test_in_train
        self.eps = eps
        self.vf_coeff = vf_coeff


    def act(self, state, return_logprob=False, deterministic=False):
        """Return an action for the given state using the policy model.

        Parameters
        ----------
        state : np.array
            The state of the environment.
        return_logprob : bool, optional
            Whether to return the log probability of choosing the chosen action.
        deterministic : bool, optional
            Whether to use a deterministic policy.

        """
        self.policy_model.eval()  # set model to evaluation mode
        pi = self.policy_model(state)
        logpi = pi.log()

        if deterministic:
            action = torch.argmax(pi).item()  # Choose the action with the highest probability
        else:
            action = torch.multinomial(torch.exp(logpi.squeeze(1)), 1)[0]

        if return_logprob:
            return action.item(), logpi[action].item(), logpi
        else:
            return action

    def value(self, state):
        """Return the predicted value for the given state using the value model.

        Parameters
        ----------
        state : np.array
            The state of the environment.

        """
        self.value_model.eval()  # set model to evaluation mode
        with torch.no_grad():  # disable gradient calculation
            return self.value_model(state)

    def redistribute_rewards(self, token_history, reward_transitions):
        """
        reward_transitions: dict of {transition_id: reward_value}
        Returns: dict of {action_index: cumulative_reward}
        """
        from collections import defaultdict
        action_rewards = defaultdict(float)

        for transition_id, reward in reward_transitions.items():
            token_ids = token_history.get_tokens_by_transition(transition_id)
            for tid in token_ids:
                chain = token_history.get_causal_chain(tid)
                if chain:
                    per_action_reward = reward / len(chain)
                    for action, _ in chain:
                        if action is not None:
                            action_rewards[action] += per_action_reward

        return action_rewards



    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, batch_size=64, sort_states=False, test_env=None, test_freq=5, test_episodes=10):
        """Train the agent on env with optional testing during training.

        Parameters
        ----------
        env : environment
            The environment to train on.
        test_env : environment, optional
            The test environment for evaluation during training.
        test_freq : int, optional
            Frequency (in epochs) to run testing during training.

        Returns
        -------
        history : dict
            Dictionary with statistics from training and testing.
        """
        tb_writer = None if logdir is None else SummaryWriter(log_dir=logdir)

        # Initialize learning rate schedulers if enabled
        # Cosine annealing helps convergence by gradually reducing LR over training
        policy_scheduler = None
        value_scheduler = None
        if self.lr_schedule:
            policy_scheduler = CosineAnnealingLR(
                self.policy_optimizer,
                T_max=epochs,
                eta_min=1e-6
            )
            value_scheduler = CosineAnnealingLR(
                self.value_optimizer,
                T_max=epochs,
                eta_min=1e-6
            )

        history = {'mean_returns': np.zeros(epochs),
                   'min_returns': np.zeros(epochs),
                   'max_returns': np.zeros(epochs),
                   'std_returns': np.zeros(epochs),
                   'mean_ep_lens': np.zeros(epochs),
                   'min_ep_lens': np.zeros(epochs),
                   'max_ep_lens': np.zeros(epochs),
                   'std_ep_lens': np.zeros(epochs),
                   'policy_updates': np.zeros(epochs),
                   'delta_policy_loss': np.zeros(epochs),
                   'policy_ent': np.zeros(epochs),
                   'policy_kld': np.zeros(epochs)}

        if test_env is not None:
            history.update({'test_mean_returns': np.zeros(epochs // test_freq + 1),
                            'test_min_returns': np.zeros(epochs // test_freq + 1),
                            'test_max_returns': np.zeros(epochs // test_freq + 1),
                            'test_std_returns': np.zeros(epochs // test_freq + 1)})

        for i in range(epochs):
            self.buffer.clear()
            return_history = self.run_episodes(env, episodes=episodes, max_episode_length=max_episode_length,
                                               store=True)

            dataloader = self.buffer.get(normalize_advantages=self.normalize_advantages,
                                         normalize_returns=self.normalize_returns,
                                         batch_size=batch_size,
                                         sort=sort_states, drop_remainder=True)

            #logpis = self.buffer.logpis

            #value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
            #policy_history = self._fit_policy_model(dataloader, logpis, epochs=self.policy_updates)
            #
            policy_history = self._fit_policy_and_value_models(dataloader, epochs=self.policy_updates)

            # Update training history
            history['mean_returns'][i] = np.mean(return_history['returns'])
            history['min_returns'][i] = np.min(return_history['returns'])
            history['max_returns'][i] = np.max(return_history['returns'])
            history['std_returns'][i] = np.std(return_history['returns'])
            history['mean_ep_lens'][i] = np.mean(return_history['lengths'])
            history['min_ep_lens'][i] = np.min(return_history['lengths'])
            history['max_ep_lens'][i] = np.max(return_history['lengths'])
            history['std_ep_lens'][i] = np.std(return_history['lengths'])
            history['policy_updates'][i] = len(policy_history['loss'])
            history['delta_policy_loss'][i] = policy_history['loss'][-1] - self.previous_policy_loss
            self.previous_policy_loss = policy_history['loss'][-1]
            history['policy_ent'][i] = policy_history['ent'][-1]
            history['policy_kld'][i] = policy_history['kld'][-1]

            # Test the agent during training
            if test_env is not None and (i + 1) % test_freq == 0:
                test_metrics = self.test_in_train(test_env, episodes=test_episodes, max_episode_length=max_episode_length, logdir=logdir)
                test_index = (i + 1) // test_freq - 1
                history['test_mean_returns'][test_index] = test_metrics['mean_returns']
                history['test_min_returns'][test_index] = test_metrics['min_returns']
                history['test_max_returns'][test_index] = test_metrics['max_returns']
                history['test_std_returns'][test_index] = test_metrics['std_returns']

                if tb_writer is not None:
                    tb_writer.add_scalar('test_mean_returns', test_metrics['mean_returns'], global_step=i)
                    tb_writer.add_scalar('test_min_returns', test_metrics['min_returns'], global_step=i)
                    tb_writer.add_scalar('test_max_returns', test_metrics['max_returns'], global_step=i)
                    tb_writer.add_scalar('test_std_returns', test_metrics['std_returns'], global_step=i)

            if test_env is None and logdir is not None and (i + 1) % save_freq == 0: #only save all the policies when no test in train is performed
                self.save_policy_weights(logdir + "/policy-" + str(i + 1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i + 1) + ".h5")
                self.save_policy_network(logdir + "/network-" + str(i + 1) + ".pth")

            # Log epoch metrics
            metrics = TrainingMetrics(
                epoch=i + 1,
                mean_return=float(history['mean_returns'][i]),
                std_return=float(history['std_returns'][i]),
                mean_length=float(history['mean_ep_lens'][i]),
                policy_loss=float(history['delta_policy_loss'][i]) if not np.isnan(history['delta_policy_loss'][i]) else None,
                kld=float(history['policy_kld'][i]) if not np.isnan(history['policy_kld'][i]) else None,
                entropy=float(history['policy_ent'][i]) if not np.isnan(history['policy_ent'][i]) else None,
            )
            get_logger().epoch_metrics(metrics)

            if tb_writer is not None:
                tb_writer.add_scalar('mean_returns', history['mean_returns'][i], global_step=i)
                tb_writer.add_scalar('min_returns', history['min_returns'][i], global_step=i)
                tb_writer.add_scalar('max_returns', history['max_returns'][i], global_step=i)
                tb_writer.add_scalar('std_returns', history['std_returns'][i], global_step=i)
                tb_writer.add_scalar('mean_ep_lens', history['mean_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('min_ep_lens', history['min_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('max_ep_lens', history['max_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('std_ep_lens', history['std_ep_lens'][i], global_step=i)
                tb_writer.add_scalar('policy_updates', history['policy_updates'][i], global_step=i)
                tb_writer.add_scalar('delta_policy_loss', history['delta_policy_loss'][i], global_step=i)
                tb_writer.add_scalar('policy_ent', history['policy_ent'][i], global_step=i)
                tb_writer.add_scalar('policy_kld', history['policy_kld'][i], global_step=i)
                tb_writer.flush()
            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

            # Step learning rate schedulers if enabled
            if self.lr_schedule:
                policy_scheduler.step()
                value_scheduler.step()

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """Run an episode and return total reward and episode length.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        max_episode_length : int, optional
            The maximum number of interactions before the episode ends.
        buffer : TrajectoryBuffer object, optional
            If included, it will store the whole rollout in the given buffer.

        Returns
        -------
        (total_reward, episode_length) : (float, int)
            The total nondiscounted reward obtained in this episode and the
            episode length. In causal RL mode, returns the environment's actual
            reward (info['pn_reward']) instead of step rewards.

        """
        state = env.reset()
        if hasattr(env, "problem") and hasattr(env.problem, "causal_trace"):
            env.problem.causal_trace.flush()

        done = False
        episode_length = 0
        total_reward = 0
        info = {'pn_reward': 0}  # Initialize info
        while not done:
            action, logprob, logpis = self.act(state, return_logprob=True)
            if self.value_model is None:
                value = 0
            elif isinstance(self.value_model, str):
                value = env.value(strategy=self.value_model, gamma=self.gam)
            else:
                value = self.value(state)
            next_state, reward, done, truncated, info = env.step(action)#, action_index=self.buffer.end)
            if buffer is not None:
                buffer.store(state, action, reward, logprob, value, logpis, token_ids=info.get('produced_token_ids'))
            # After storing, apply any eligibility credits returned by the environment

            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state

        if buffer is not None:
            if 'eligibility_credits' in info and info['eligibility_credits'] is not None:
                buffer.finish(credits=info['eligibility_credits'], mode="replace")
            else:
                buffer.finish(credits=None)

        # Return actual environment reward (info['pn_reward']) which contains causal RL credits
        # In causal mode, step rewards are 0, so total_reward would be 0
        # info['pn_reward'] contains the true accumulated reward from causal redistribution
        actual_reward = info.get('pn_reward', total_reward)
        return actual_reward, episode_length

    def run_episodes(self, env, episodes=100, tot_steps=None, max_episode_length=None, store=False):
        """Run several episodes, store interaction in buffer, and return history.

        Parameters
        ----------
        env : environment
            The environment to interact with.
        episodes : int, optional
            The number of episodes to perform.
        tot_steps : int, optional
            The total number of steps to perform across all episodes, if episodes is None.
        max_episode_length : int, optional
            The maximum number of steps before the episode is terminated.
        store : bool, optional
            Whether or not to store the rollout in self.buffer.

        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """


        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}
        for i in range(episodes):
            R, L = self.run_episode(env, max_episode_length=max_episode_length, buffer=self.buffer)
            history['returns'][i] = R
            history['lengths'][i] = L
        return history

    def _fit_policy_model(self, dataloader, logpis, epochs=1):
        """Fit policy model using data from dataset.

        Parameters
        ----------
        dataloader : DataLoader
            The data loader for the dataset.
        logpis : list of Tensors
            The log probabilities of the actions taken in the dataset.
        epochs : int, optional
            The number of epochs to train the policy model.
        Returns

        -------
        dict
            Dictionary with loss, KLD, and entropy history for each epoch.

        """
        history = {'loss': [], 'kld': [], 'ent': []}


        for epoch in range(epochs):
            start = 0
            loss, kld, ent, batches = 0, 0, 0, 0
            for i, batch in enumerate(dataloader):
                lp = logpis[start:start + len(batch)]
                start += len(batch)
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(batch, lp)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1
            if batches == 0:
                get_logger().no_batches_warning()
                continue
            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            if self.kld_limit is not None and kld/batches > self.kld_limit:
                print(f'Early stopping at epoch {epoch+1} due to KLD divergence. The computed KLD was {kld/batches}.')
                return {k: np.array(v) for k, v in history.items()}
        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_model_step(self, batch, logpis):
        """Fit policy model on one batch of data.

        Parameters
        ----------
        batch : DataBatch
            The batch of data containing states, actions, advantages, etc.
        logpis : list of Tensors
            The log probabilities of the actions taken in the dataset.
        Returns
        -------
        loss : float
            The loss value for the policy model.
        kld : float
            The Kullback-Leibler divergence between the new and old policies.
        ent : float
            The entropy of the policy distribution.
        """
        self.policy_model.train()  # set model to training mode
        self.policy_optimizer.zero_grad()  # zero out gradients

        # Save the initial weights
        #initial_weights = {name: param.clone() for name, param in self.policy_model.named_parameters()}

        indexes = batch['a_transition'].batch.data
        states = batch
        actions = torch.tensor(batch.y)
        logprobs = batch.logprobs.clone()
        advantages = batch.advantage.clone()

        epsilon = 1e-7
        new_probs = self.policy_model(states)
        new_logpis = (new_probs + epsilon).log()

        #new_logprobs contains, for each unique index in indexes, the value in the slice of logpis corresponding
        #to the current index in indexes with index action[index]
        new_logprobs = torch.stack([new_logpis[indexes == index][actions[index]] for index in indexes.unique()]).squeeze(1)

        # Calculate batch size
        batch_size = len(indexes.unique())

        # Compute normalized entropy
        ent = -torch.sum(new_probs * new_logpis) / batch_size

        # Compute normalized KLD
        logpis = torch.cat(logpis, dim=0)
        kld = torch.sum(new_probs * (new_logpis - logpis)) / batch_size
        loss = torch.mean(self.policy_loss(new_logprobs, logprobs, advantages)) - self.ent_bonus * ent

        try:
            loss.backward(retain_graph=True)  # compute gradients
        except Exception as e:
            print("Invalid loss")

        # Clip gradients for stability - critical for PPO
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        print(f"KLD divergence: {kld.item()}")
        return loss.item(), kld.item(), ent.item()

    # Assuming `model` is your PyTorch model
    def check_gradient_norms(self, model):
        """Check and print the gradient norms for each parameter in the model.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to check gradients for.
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                print(f"Gradient norm for {name}: {grad_norm}")
            else:
                print(f"No gradient for {name}")

    def load_policy_weights(self, filename):
        """Load weights from filename into the policy model.

        Parameters
        ----------
        filename : str
            The path to the file from which the model weights will be loaded.
        """
        self.policy_model.load_weights(filename)

    def save_policy_weights(self, filename):
        """Save the current weights in the policy model to filename.

        Parameters
        ----------
        filename : str
            The path to the file where the model weights will be saved.
        """
        self.policy_model.save_weights(filename)

    def _fit_value_model(self, dataloader, epochs=1):
        """Fit value model using data from dataset.

        Parameters
        ----------
        dataloader : DataLoader
            The data loader for the dataset.
        logpis : list of Tensors
            The log probabilities of the actions taken in the dataset.
        epochs : int, optional
            The number of epochs to train the policy model.

        Returns
        -------
        dict
            Dictionary containing training history with key 'loss'.

        """
        if self.value_model is None or isinstance(self.value_model, str):
            epochs = 0
        history = {'loss': []}
        for epoch in range(epochs):
            loss, batches = 0, 0
            for batch in dataloader:
                #batch = batch[0]
                batch_loss = self._fit_value_model_step(batch)
                loss += batch_loss
                batches += 1
            if batches == 0:
                print("No complete batches to process.")
                continue
            history['loss'].append(loss / batches)
        return {k: np.array(v) for k, v in history.items()}

    def _fit_value_model_step(self, batch):
        """Fit value model on one batch of data."""
        self.value_model.train()

        #indexes = batch['a_transition'].batch.data
        states = batch
        values = batch.value.clone() # discounted returns

        pred_values = self.value_model(states).squeeze()
        loss = torch.mean(self.value_loss.forward(input=pred_values, target=values))

        self.value_optimizer.zero_grad()
        try:
            loss.backward(retain_graph=True)
        except Exception as e:
            print("Loss.backward produced an invalid output.")

        #torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 0.5) #as implemented in tianshou ppo
        self.value_optimizer.step()

        return loss.item()

    def test_in_train(self, env, episodes=100, max_episode_length=None, deterministic=True, logdir=None):
        """Evaluate the agent on a test environment during training.

        Parameters
        ----------
        env : environment
            The test environment to evaluate on.
        episodes : int, optional
            The number of episodes to run for evaluation.
        max_episode_length : int, optional
            The maximum number of steps in an episode.
        deterministic : bool, optional
            Whether to use a deterministic policy during testing.
        logdir : str, optional
            Directory to save the best policy.

        Returns
        -------
        test_metrics : dict
            Dictionary containing evaluation metrics (mean, min, max, std returns and lengths).
        """
        history = {'returns': np.zeros(episodes), 'lengths': np.zeros(episodes)}
        for i in range(episodes):
            state = env.reset()
            done = False
            episode_length = 0
            total_reward = 0
            info = {'pn_reward': 0}
            while not done:
                action = self.act(state, deterministic=deterministic)
                next_state, reward, done, truncated, info = env.step(action)
                episode_length += 1
                state = next_state

            if episode_length == 0:
                get_logger().warning("Episode length is zero - no valid action produced")
            history['returns'][i] = info['pn_reward']
            history['lengths'][i] = episode_length

        test_metrics = {
            'mean_returns': np.mean(history['returns']),
            'min_returns': np.min(history['returns']),
            'max_returns': np.max(history['returns']),
            'std_returns': np.std(history['returns']),
            'mean_ep_lens': np.mean(history['lengths']),
            'min_ep_lens': np.min(history['lengths']),
            'max_ep_lens': np.max(history['lengths']),
            'std_ep_lens': np.std(history['lengths']),
        }

        # Save the best policy if the current mean_returns is better
        if test_metrics['mean_returns'] > self.best_test_metric:
            self.best_test_metric = test_metrics['mean_returns']
            if logdir is not None:
                self.save_policy_network(f"{logdir}/best_policy.pth")
                get_logger().best_policy_saved(logdir, self.best_test_metric)

        # After testing or inference
        self.policy_model.train()
        self.value_model.train()


        return test_metrics

    def load_value_weights(self, filename):
        """Load weights from filename into the value model."""
        if self.value_model is not None and self.value_model != 'env':
            self.value_model.load_weights(filename)

    def save_value_weights(self, filename):
        """Save the current weights in the value model to filename."""
        if self.value_model is not None and not isinstance(self.value_model, str):
            self.value_model.save_weights(filename)

    def save_policy_network(self, filename):
        """Save the current policy to file.

        Parameters
        ----------
        filename : str
            The path to the file where the model will be saved.
        """

        torch.save(self.policy_model, filename)

    def load_policy_network(self, filename):
        """Load the current policy from file.

        Parameters
        ----------
        filename : str
            The path to the file from which the model will be loaded.
        """
        self.policy_model = torch.load(torch.load(filename))

    def _fit_policy_and_value_models(self, dataloader, epochs=1):
        """Fit both policy and value models simultaneously using data from dataset.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader containing batches of training data.
        epochs : int, optional
            Number of epochs to train for.

        Returns
        -------
        dict
            Dictionary containing training history with keys 'loss', 'kld', and 'ent'.
        """
        history = {'loss': [], 'kld': [], 'ent': []}
        for epoch in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            start = 0
            for batch in dataloader:
                start += len(batch)
                batch_loss, batch_kld, batch_ent = self._fit_policy_and_value_model_step(batch)#, lp)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1

            if batches == 0:
                get_logger().no_batches_warning()
                continue
            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            if self.kld_limit is not None and kld/batches > self.kld_limit:
                get_logger().debug(f'Early stopping due to KLD divergence: {kld/batches:.4f}')
                return {k: np.array(v) for k, v in history.items()}
        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_and_value_model_step(self, batch):
        self.policy_model.train()
        self.value_model.train()

        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        # Standardize new log-probs to be 1-D per node
        new_logpis = (new_probs + epsilon).log()
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)  # (num_nodes,) instead of (num_nodes,1)

        actions = torch.as_tensor(batch.y)
        advantages = batch.advantage.clone()
        old_logprob = batch.logprobs.clone()

        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)

        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0

        # Split new logits by node type in the same order as actions_dict
        new_logpis_a = new_logpis[:nA] if nA else None
        new_logpis_p = new_logpis[nA:nA + nP] if nP else None

        # Old logits (standardize them to 1-D up front)
        old_logpis_a = batch['a_transition'].logpis if has_a else None
        if old_logpis_a is not None and old_logpis_a.dim() == 2 and old_logpis_a.size(-1) == 1:
            old_logpis_a = old_logpis_a.squeeze(-1)

        old_logpis_p = None
        if has_p and hasattr(batch['postpone'], 'logpis'):
            old_logpis_p = batch['postpone'].logpis
            if old_logpis_p is not None and old_logpis_p.dim() == 2 and old_logpis_p.size(-1) == 1:
                old_logpis_p = old_logpis_p.squeeze(-1)

        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None

        unique_samples = (idx_a.unique() if has_a else idx_p.unique())

        sel_new, sel_old, sel_adv = [], [], []
        kld_terms = []

        for s in unique_samples:
            parts_new = []
            parts_old = []

            if has_a:
                mask_a = (idx_a == s)
                ns_a = new_logpis_a[mask_a]  # 1-D slice
                os_a = old_logpis_a[mask_a]  # 1-D slice
                # Flatten defensively (handles accidental (k,1))
                ns_a = ns_a.reshape(-1)
                os_a = os_a.reshape(-1)
                if ns_a.numel():
                    parts_new.append(ns_a)
                    parts_old.append(os_a)

            if has_p and new_logpis_p is not None:
                mask_p = (idx_p == s)
                ns_p = new_logpis_p[mask_p].reshape(-1)  # 1-D slice
                if old_logpis_p is not None:
                    os_p = old_logpis_p[mask_p].reshape(-1)  # 1-D slice
                else:
                    # Fallback if you didn’t store old logpis for postpone yet:
                    os_p = torch.zeros_like(ns_p)
                if ns_p.numel():
                    parts_new.append(ns_p)
                    parts_old.append(os_p)

            if len(parts_new) == 0:
                # No action nodes for this sample -> skip policy update; still train value below
                continue

            # Concatenate 1-D slices safely
            ns_cat = torch.cat(parts_new, dim=0)  # (A_s + P_s,)
            os_cat = torch.cat(parts_old, dim=0)  # same length

            a_idx = int(actions[s])
            #if a_idx < 0 or a_idx >= ns_cat.shape[0]:
            #    # out-of-range chosen index -> skip this sample
            #    continue

            sel_new.append(ns_cat[a_idx].reshape(()))
            sel_old.append(old_logprob[s].reshape(()))
            sel_adv.append(advantages[s].reshape(()))

            # KLD per sample: KL(old || new) = log(p_old) - log(p_new)
            # This measures how much the new policy diverges from the old policy for this action
            kld_sample = old_logprob[s].item() - ns_cat[a_idx].item()
            kld_terms.append(kld_sample)

        # ----- Value loss (always trained)
        pred_values = self.value_model(batch).squeeze()
        loss_value = torch.mean(self.value_loss.forward(input=pred_values, target=batch.value.clone()))

        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        #if len(sel_new) == 0:
        #    loss_value.backward(retain_graph=True)
        #    torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        #    self.value_optimizer.step()
        #    return float(loss_value.item()), 0.0, 0.0

        new_sel = torch.stack(sel_new)
        old_sel = torch.stack(sel_old)
        adv_sel = torch.stack(sel_adv)

        loss_policy_core = torch.mean(self.policy_loss(new_sel, old_sel, adv_sel))
        # Compute KLD as mean difference in log probabilities
        if len(kld_terms) > 0:
            kld = torch.tensor(kld_terms, device=new_probs.device).mean()
        else:
            kld = torch.tensor(0.0, device=new_probs.device)

        ent = -torch.mean(new_probs * new_logpis)  # coarse entropy over all nodes

        loss_total = loss_policy_core + self.vf_coeff * loss_value - self.ent_bonus * ent
        p0 = sum((p.data.norm() for p in self.policy_model.parameters()), torch.tensor(0.0))


        loss_total.backward(retain_graph=True)
        # Clip gradients for stability - critical for PPO
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.value_optimizer.step()
        self.policy_optimizer.step()

        p1 = sum((p.data.norm() for p in self.policy_model.parameters()), torch.tensor(0.0))
        get_logger().debug(f"Δ||θ|| = {(p1 - p0).item():.6f}")

        return loss_total.item(), kld.item(), ent.item()

def pg_surrogate_loss(new_logps, old_logps, advantages):
    """Return loss with gradient for policy gradient.

    Parameters
    ----------
    new_logps : Tensor (batch_dim,)
        The output of the current model for the chosen action.
    old_logps : Tensor (batch_dim,)
        The previous logged probability of the chosen action.
    advantages : Tensor (batch_dim,)
        The computed advantages.

    Returns
    -------
    loss : Tensor (batch_dim,)
        The loss for each interaction.

    """
    return -new_logps * advantages


class PGAgent(Agent):
    """A policy gradient agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.

    """

    def __init__(self, policy_network, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = pg_surrogate_loss


def ppo_surrogate_loss(method='clip', eps=0.2, c=0.01):
    """Return loss function with gradient for proximal policy optimization.

    Parameters
    ----------
    method : {'clip', 'penalty'}
        The specific loss for PPO.
    eps : float
        The clip ratio if using 'clip'.
    c : float
        The fixed KLD weight if using 'penalty'.

    """
    if method == 'clip':

        def loss(new_logps, old_logps, advantages):
            """Return loss with gradient for clipped PPO.

            Parameters
            ----------
            new_logps : Tensor (batch_dim,)
                The output of the current model for the chosen action.
            old_logps : Tensor (batch_dim,)
                The previous logged probability for the chosen action.
            advantages : Tensor (batch_dim,)
                The computed advantages.

            Returns
            -------
            loss : Tensor (batch_dim,)
                The loss for each interaction.
            """

            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            try:
                ret_loss = -torch.min(surr1, surr2)
            except Exception as e:
                print("Invalid loss detected.")
            return ret_loss
        return loss
    elif method == 'penalty':
        def loss(new_logps, old_logps, advantages):
            """Return loss with gradient for penalty PPO.

            Parameters
            ----------
            new_logps : Tensor (batch_dim,)
                The output of the current model for the chosen action.
            old_logps : Tensor (batch_dim,)
                The previous logged probability for the chosen action.
            advantages : Tensor (batch_dim,)
                The computed advantages.

            Returns
            -------
            loss : Tensor (batch_dim,)
                The loss for each interaction.
            """
            return -(torch.exp(new_logps - old_logps) * advantages - c * (old_logps - new_logps))
        return loss
    else:
        raise ValueError('unknown PPO method')


class PPOAgent(Agent):
    """Proximal Policy Optimization agent.

    Parameters
    ----------
    policy_network : network
        The network for the policy model.
    method : {'clip', 'penalty'}
        The loss type for PPO.
    eps : float
        The clip ratio if using 'clip'.
    c : float
        The fixed KLD weight if using 'penalty'.

    """

    def __init__(self, policy_network, method='clip', eps=0.2, c=0.01, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.policy_loss = ppo_surrogate_loss(method=method, eps=eps, c=c)
