"""Policy gradient agents that support changing state spaces, specifically for graph environments.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy gradient) and proximal policy optimization
agent.
"""
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from gympn.data import TrajectoryBuffer, print_status_bar

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
                 kld_limit=0.01, ent_bonus=0.01, test_in_train=True, vf_coeff=0.5):
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
            dataloader = self.buffer.get(normalize_advantages=self.normalize_advantages, batch_size=batch_size,
                                         sort=sort_states)

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
                print("Testing the agent during training...")
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
            episode length.

        """
        state = env.reset()
        done = False
        episode_length = 0
        total_reward = 0
        while not done:
            action, logprob, logpis = self.act(state, return_logprob=True)
            if self.value_model is None:
                value = 0
            elif isinstance(self.value_model, str):
                value = env.value(strategy=self.value_model, gamma=self.gam)
            else:
                value = self.value(state)
            next_state, reward, done, truncated, _ = env.step(action)
            if buffer is not None:
                buffer.store(state, action, reward, logprob, value, logpis)
            episode_length += 1
            total_reward += reward
            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state
        if buffer is not None:
            buffer.finish()
        return total_reward, episode_length

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
                print('Batch: ', batches + 1, ' of ', len(dataloader))
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(batch, lp)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1
            if batches == 0:
                print("No complete batches to process.")
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
            #self.check_gradient_norms(self.policy_model)
        except Exception as e:
            print("Invalid loss")

        #torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 0.5) #as implemented in tianshou ppo
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
                #if (max_episode_length is not None and episode_length > max_episode_length) or done or truncated:
                #    break
                episode_length += 1
                #total_reward += reward
                state = next_state

            if episode_length == 0:
                print("Warning: Episode length is zero, this episode did not produce any valid action.")
            history['returns'][i] = info['pn_reward'] #total_reward
            print(f"The recorded reward was: {info['pn_reward']}")
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
                print(f"New best policy saved with mean_returns: {self.best_test_metric}")

        # After testing or inference
        self.policy_model.train()
        self.value_model.train()

        print("Finished test in train")
        print('----------------------------------------------------')

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
                print('Batch: ', batches + 1, ' of ', len(dataloader))
                start += len(batch)
                batch_loss, batch_kld, batch_ent = self._fit_policy_and_value_model_step(batch)#, lp)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1

            if batches == 0:
                print("No complete batches to process.")
                continue
            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            if self.kld_limit is not None and kld/batches > self.kld_limit:
                print(f'Early stopping at epoch {epoch+1} due to KLD divergence. The computed KLD was {kld/batches}.')
                return {k: np.array(v) for k, v in history.items()}
        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_and_value_model_step(self, batch):#, logpis):
        """Perform one training step for both policy and value models.

        Parameters
        ----------
        batch : dict
            Batch of training data containing states, actions, advantages, etc.

        Returns
        -------
        tuple
            (policy_loss, kld, entropy) for the training step.
        """
        self.policy_model.train()  # set model to training mode
        self.value_model.train()  # set model to training mode

        indexes = batch['a_transition'].batch.data

        #handle postpone nodes if present
        if 'postpone' in batch.x_dict.keys():
            postpone_indexes = batch['postpone'].batch.data
            indexes = torch.cat((indexes, postpone_indexes), dim=0)
        else:
            print("No postpone nodes found in batch.")

        states = batch
        actions = torch.tensor(batch.y)
        logprobs = batch.logprobs.clone()
        advantages = batch.advantage.clone()
        logpis = batch.logpis.clone().unsqueeze(-1)

        epsilon = 1e-7
        new_probs = self.policy_model(states)
        new_logpis = (new_probs + epsilon).log()

        # new_logprobs contains, for each unique index in indexes, the value in the slice of logpis corresponding
        # to the current index in indexes with index action[index]
        try:
            new_logprobs = torch.stack(
                [new_logpis[indexes == index][actions[index]] for index in indexes.unique()]).squeeze(1)




        except IndexError:
            print("IndexError encountered while stacking new_logprobs. Check action indices and batch data.")
            raise

        # Compute the loss and gradients
        # Calculate batch size
        batch_size = len(indexes.unique())

        # Compute normalized entropy
        ent = -torch.sum(new_probs * new_logpis) / batch_size

        # Compute normalized KLD
        #logpis = torch.cat(logpis, dim=0)
        try:
            kld = torch.sum(new_probs * (new_logpis - logpis)) / batch_size
            #print(f'KLD divergence computed: {kld.item()}')
        except RuntimeError:
            print("RuntimeError encountered while computing KLD. Check dimensions of new_logpis and logpis.")
            raise

        loss_value = torch.mean(
            self.value_loss.forward(input=self.value_model(states).squeeze(), target=batch.value.clone()))

        loss_policy = torch.mean(self.policy_loss(new_logprobs, logprobs, advantages)) + self.vf_coeff*loss_value - self.ent_bonus * ent

        self.value_optimizer.zero_grad()  # zero out gradients
        self.policy_optimizer.zero_grad()  # zero out gradients

        try:
            loss_policy.backward(retain_graph=True)  # compute gradients
            #self.check_gradient_norms(self.policy_model)
        except Exception as e:
            print("Invalid loss")

        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1) #as implemented in tianshou ppo
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1)
        self.value_optimizer.step()
        self.policy_optimizer.step()

        return loss_policy.item(), kld.item(), ent.item()

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
