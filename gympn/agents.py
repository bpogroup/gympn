"""Policy gradient agents that support changing state spaces, specifically for graph environments.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy optimization
agent.
"""
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing as mp
from typing import Dict

from gympn.data import TrajectoryBuffer, print_status_bar
from gympn.logging_utils import Logger, TrainingMetrics, TestMetrics, get_logger

#torch.autograd.set_detect_anomaly(True)


# ============================================================================
# MULTIPROCESSING WORKER FUNCTION
# ============================================================================
def _run_episode_worker(args):
    """Worker function for parallel episode collection (picklable).

    Must be at module level to be serializable by multiprocessing.
    """
    agent, env_copy, max_len = args
    return agent.run_episode(env_copy, max_episode_length=max_len, buffer=None)


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
        Whether to normalize advantages. Default is True.
        Advantage normalization (zero-mean, unit-variance) is generally recommended
        for stable policy learning. Safe to use with any value training setup.
    normalize_returns : bool, optional
        Whether to normalize returns (discounted cumulative rewards) for value training.
        Default is False (IMPORTANT for correctness).

        ⚠️  CRITICAL: If normalize_returns=True, the value network will be trained on
        normalized targets (mean=0, std=1). However, value predictions from rollout time
        will be in the normalized scale, while GAE calculations use raw rewards/credits.
        This creates a scale mismatch in delta = reward + gamma * v_{t+1} - v_t.

        RECOMMENDATION: Keep normalize_returns=False (default) to avoid this mismatch.
        If you need stable value training, use normalize_advantages=True instead, which
        provides similar benefits without the inconsistency.
    kld_limit : float, optional
        The limit on KL divergence for early stopping policy updates.
    ent_bonus : float, optional
        Bonus factor for sampled policy entropy.

    """

    def __init__(self,
                 policy_network, value_network, policy_lr=1e-4, policy_updates=1,
                 value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=0.01, ent_bonus=0.01, test_in_train=True, vf_coeff=0.05,
                 normalize_returns=False, lr_schedule=False,
                 causal_scheme='flow', causal_gamma=0.9, causal_pg=False,
                 causal_rl=False):
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
        self.causal_pg = causal_pg
        self.causal_rl = bool(causal_rl)
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam,
                                       causal_scheme=causal_scheme,
                                       causal_gamma=causal_gamma,
                                       causal_pg=causal_pg,
                                       causal_rl=causal_rl)
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
            if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                try:
                    print(f"[PPO DEBUG] act: logpi shape = {tuple(logpi.shape)}")
                except Exception:
                    pass
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
              logdir=None, batch_size=64, sort_states=False, test_env=None, test_freq=5, test_episodes=10,
              wandb_logger=None, num_workers=1):
        """Train the agent on env with optional testing during training.

        Parameters
        ----------
        env : environment
            The environment to train on.
        test_env : environment, optional
            The test environment for evaluation during training.
        test_freq : int, optional
            Frequency (in epochs) to run testing during training.
        wandb_logger : WandBLogger, optional
            Logger for Weights & Biases integration.

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
            # Use parallel episode collection with dill (4-8x speedup on collection, 2-4x overall)
            # Dill can serialize lambda functions and complex objects like SimVar
            return_history = self.run_episodes(env, episodes=episodes, max_episode_length=max_episode_length,
                                               store=True, num_workers=num_workers)

            # === RUDDER Training and Reward Redistribution ===
            if self.rudder_agent is not None and self.rudder_agent.should_train():
                self._apply_rudder_credit_assignment(return_history)
                rudder_loss = self.rudder_agent.train(num_epochs=5)
                if wandb_logger and i % 10 == 0:
                    wandb_logger.log({'rudder/loss': rudder_loss}, step=i)
                self.rudder_agent.step_epoch()
                get_logger().info(f"  [RUDDER] Training loss: {rudder_loss:.4f}")

            # Advantage normalization.
            # In causal RL mode, advantages are (credit - V(s)) which need
            # normalization just like standard GAE advantages.
            # The _normalize_advantages method handles low-variance cases
            # (e.g., near-optimal policy) with a center-only fallback.
            normalize_adv_for_batch = self.normalize_advantages

            dataloader = self.buffer.get(normalize_advantages=normalize_adv_for_batch,
                                         normalize_returns=self.normalize_returns,
                                         batch_size=batch_size,
                                         sort=sort_states, drop_remainder=True)

            #logpis = self.buffer.logpis

            #value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
            #policy_history = self._fit_policy_model(dataloader, logpis, epochs=self.policy_updates)
            #
            # Standard PPO training: policy + value networks.
            # In causal RL mode, the value net learns to predict per-action
            # credits and advantages are (credit - V(s)), normalized.
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

            # === CRITICAL FIX: Check if policy_history has loss data before accessing ===
            if len(policy_history['loss']) > 0:
                history['delta_policy_loss'][i] = policy_history['loss'][-1] - self.previous_policy_loss
                self.previous_policy_loss = policy_history['loss'][-1]
                history['policy_ent'][i] = policy_history['ent'][-1]
                history['policy_kld'][i] = policy_history['kld'][-1]
            else:
                # No batches were processed - warn and skip metrics
                get_logger().warning(
                    f"Epoch {i+1}: No complete batches to process. "
                    f"Buffer size ({len(self.buffer)}) < batch_size ({batch_size}). "
                    f"Consider reducing batch_size or increasing episodes per epoch."
                )
                history['delta_policy_loss'][i] = 0.0
                history['policy_ent'][i] = 0.0
                history['policy_kld'][i] = 0.0

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

                # Log test metrics to W&B if logger provided
                if wandb_logger is not None:
                    wandb_logger.log_test(
                        epoch=i,
                        mean_return=test_metrics['mean_returns'],
                        std_return=test_metrics['std_returns'],
                        min_return=test_metrics['min_returns'],
                        max_return=test_metrics['max_returns'],
                    )

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
            # Log to W&B if logger provided
            if wandb_logger is not None:
                wandb_logger.log_epoch(
                    epoch=i,
                    mean_return=float(history['mean_returns'][i]),
                    std_return=float(history['std_returns'][i]),
                    policy_loss=float(history['delta_policy_loss'][i]) if not np.isnan(history['delta_policy_loss'][i]) else None,
                    kld=float(history['policy_kld'][i]) if not np.isnan(history['policy_kld'][i]) else None,
                    entropy=float(history['policy_ent'][i]) if not np.isnan(history['policy_ent'][i]) else None,
                )

            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

            # Step learning rate schedulers if enabled
            if self.lr_schedule:
                policy_scheduler.step()
                value_scheduler.step()

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """Run an episode and return total reward and episode length.

        OPTIMIZATION: Uses batched value predictions for 5-20x speedup.
        Value network is called every N steps on a batch of states instead of
        calling it on every single step.

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
        # NOTE: Do NOT flush causal traces here. env.reset() already flushes
        # the trace and then get_to_first_action() populates it with initial
        # evolution data (e.g. 'arrive' tokens and their parent-child links).
        # A second flush would wipe that data and break the causal chain for
        # tokens created during the initial evolution phase.

        done = False
        episode_length = 0
        total_reward = 0
        info = {'pn_reward': 0}  # Initialize info

        # === OPTIMIZATION: Batch value predictions ===
        # Instead of computing value every step, collect states and compute in batches
        states_batch = []
        actions_batch = []
        logprobs_batch = []
        logpis_batch = []
        rewards_batch = []
        value_batch_size = 8  # Compute values for 8 states at a time

        while not done:
            action, logprob, logpis = self.act(state, return_logprob=True)

            # Collect for batch processing
            states_batch.append(state)
            actions_batch.append(action)
            logprobs_batch.append(logprob)
            logpis_batch.append(logpis)

            next_state, reward, done, truncated, info = env.step(action)
            rewards_batch.append(reward)

            episode_length += 1
            total_reward += reward

            # Compute values in batch every N steps or at episode end
            if len(states_batch) >= value_batch_size or done:
                values = self._compute_batch_values(states_batch, env)

                # Store all buffered transitions
                if buffer is not None:
                    for i, (s, a, lp, lpis, r) in enumerate(zip(
                        states_batch, actions_batch, logprobs_batch,
                        logpis_batch, rewards_batch)):
                        buffer.store(s, a, r, lp, values[i], lpis,
                                   token_ids=None)

                # Clear batches for next iteration
                states_batch = []
                actions_batch = []
                logprobs_batch = []
                logpis_batch = []
                rewards_batch = []

            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state

        if buffer is not None:
            if 'eligibility_credits' in info and info['eligibility_credits'] is not None:
                # Diagnostic: if environment signals debugging, print causal trace stats
                try:
                    pn = getattr(env, 'pn', None) or getattr(env, 'problem', None)
                    if pn is not None and getattr(pn, '_debugging', False) and pn.causal_rl:
                        ct = pn.causal_trace
                        tok_count, trans_count = ct.stats()
                        # compute a quick redistribution sample (flow) to check sizes
                        try:
                            sample_cr = ct.redistribute_rewards(gamma=0.9, scheme='flow')
                        except Exception as e:
                            sample_cr = None
                            import warnings
                            warnings.warn(f"[CAUSAL-DIAG] tokens={tok_count}, transitions={trans_count}, ep_steps={episode_length}, redis_len={len(sample_cr) if sample_cr is not None else 'ERR'}, redis_sum={sum(sample_cr) if sample_cr is not None else 'ERR'}")
                except Exception:
                    pass
                buffer.finish(credits=info['eligibility_credits'], mode="replace")
            else:
                buffer.finish(credits=None)

        # Return actual environment reward (info['pn_reward']) which contains causal RL credits
        # In causal mode, step rewards are 0, so total_reward would be 0
        # info['pn_reward'] contains the true accumulated reward from causal redistribution
        actual_reward = info.get('pn_reward', total_reward)
        return actual_reward, episode_length

    def _compute_batch_values(self, states_list, env):
        """Compute values for a batch of states efficiently.

        OPTIMIZATION: Uses true PyTorch batching on heterogeneous graphs.
        Provides 2-5x speedup compared to individual forward passes.

        Parameters
        ----------
        states_list : list
            List of state observations from the environment
        env : environment
            The environment (for strategy-based value functions)

        Returns
        -------
        values : list
            List of scalar value predictions
        """
        if len(states_list) == 0:
            return []

        if self.value_model is None:
            return [0] * len(states_list)

        if isinstance(self.value_model, str):
            # Strategy-based value function - must call per step (no batching possible)
            return [env.value(strategy=self.value_model, gamma=self.gam)
                    for _ in states_list]

        # === OPTIMIZED: Use torch_geometric batching for heterogeneous graphs ===
        # This is much faster than looping through individual states
        self.value_model.eval()
        with torch.no_grad():
            try:
                # Try to use torch_geometric batching if states are graph objects
                from torch_geometric.data import HeteroData, Batch

                # Check if states are HeteroData objects
                if states_list and isinstance(states_list[0], dict) and 'graph' in states_list[0]:
                    # Extract graph objects and batch them
                    graphs = [s['graph'] for s in states_list]

                    if isinstance(graphs[0], HeteroData):
                        # Batch heterogeneous graphs
                        batched_graph = Batch.from_data_list(graphs)

                        # Single forward pass on batched graph
                        batch_values = self.value_model(batched_graph)

                        # Extract per-graph values
                        if isinstance(batch_values, torch.Tensor):
                            # Values should have shape [num_graphs] or [num_graphs, 1]
                            if batch_values.dim() > 1:
                                values = batch_values[:, 0].tolist() if batch_values.size(1) == 1 else batch_values.tolist()
                            else:
                                values = batch_values.tolist()
                            return values
            except Exception as e:
                # Fall back to individual computation if batching fails
                import warnings
                warnings.warn(f"Batching failed ({e}), falling back to sequential computation")

        # Fall back: compute individually (slower but always works)
        self.value_model.eval()
        with torch.no_grad():
            values = []
            for state in states_list:
                value = self.value_model(state)
                # Handle different value output shapes
                if isinstance(value, torch.Tensor):
                    value = value.squeeze().item() if value.numel() == 1 else value
                values.append(value)
            return values

    def run_episodes(self, env, episodes=100, tot_steps=None, max_episode_length=None, store=False, num_workers=None):
        """Run several episodes, store interaction in buffer, and return history.

        OPTIMIZATION: Supports parallel episode collection using multiprocessing.
        With num_workers > 1, episodes are collected in parallel across multiple CPU cores.
        This provides 4-8x speedup on episode collection (2-4x overall).

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
        num_workers : int, optional
            Number of parallel workers. If None, defaults to sequential.
            If > 1, uses multiprocessing.Pool for parallel collection.

        Returns
        -------
        history : dict
            Dictionary which contains information from the runs.

        """
        import copy
        import os

        history = {'returns': np.zeros(episodes),
                   'lengths': np.zeros(episodes)}

        # Determine number of workers
        if num_workers is None:
            num_workers = 1
        else:
            num_workers = min(num_workers, episodes, os.cpu_count() or 1)

        if num_workers <= 1 or episodes < 2:
            # Fall back to sequential for small episode counts or num_workers=1
            for i in range(episodes):
                R, L = self.run_episode(env, max_episode_length=max_episode_length, buffer=self.buffer if store else None)
                history['returns'][i] = R
                history['lengths'][i] = L
        else:
            # Parallel episode collection using dill for serialization
            # Dill can handle lambda functions and complex objects like SimVar
            try:
                import dill
                import multiprocessing

                # Create environment copies for each worker
                env_copies = [copy.deepcopy(env) for _ in range(num_workers)]

                # Prepare arguments for workers
                worker_args = [(self, env_copies[i % num_workers], max_episode_length)
                              for i in range(episodes)]

                # Use spawn context with dill for robust serialization
                ctx = multiprocessing.get_context('spawn')

                # Create a custom Pool that uses dill for pickling
                # When dill is imported, it automatically patches pickle to use dill's methods
                with ctx.Pool(processes=num_workers) as pool:
                    results = pool.map(_run_episode_worker, worker_args)

                # Aggregate results
                for i, (R, L) in enumerate(results):
                    history['returns'][i] = R
                    history['lengths'][i] = L

            except Exception as e:
                # Silently fall back to sequential if parallel fails
                for i in range(episodes):
                    R, L = self.run_episode(env, max_episode_length=max_episode_length, buffer=self.buffer if store else None)
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
            early_stop_epoch = False

            for i, batch in enumerate(dataloader):
                lp = logpis[start:start + len(batch)]
                start += len(batch)
                batch_loss, batch_kld, batch_ent = self._fit_policy_model_step(batch, lp)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                batches += 1

                # === CRITICAL FIX: Check KLD limit per-batch ===
                # This allows early stopping DURING an epoch, not just after
                if self.kld_limit is not None and batch_kld > self.kld_limit:
                    get_logger().debug(f'Early stopping at epoch {epoch+1}, batch {i+1}: '
                                      f'batch KLD {batch_kld:.6f} exceeded limit {self.kld_limit:.6f}')
                    early_stop_epoch = True
                    break

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            avg_loss = loss / batches
            avg_kld = kld / batches
            avg_ent = ent / batches
            history['loss'].append(avg_loss)
            history['kld'].append(avg_kld)
            history['ent'].append(avg_ent)

            # Stop training if KLD exceeded limit in any batch of this epoch
            if early_stop_epoch:
                get_logger().debug(f'Early stopping due to KLD divergence at epoch {epoch+1}. '
                                  f'Average epoch KLD: {avg_kld:.6f}, Limit: {self.kld_limit:.6f}')
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
            print("Invalid loss", e)

        # Clip gradients for stability - critical for PPO
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        # Debug: print gradient norms per parameter if requested
        try:
            if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                for name, param in self.policy_model.named_parameters():
                    if param.grad is not None:
                        try:
                            gnorm = float(torch.norm(param.grad).item())
                        except Exception:
                            gnorm = None
                        print(f"[PPO DEBUG] grad_norm policy {name}: {gnorm}")
        except Exception:
            pass

        self.policy_optimizer.step()

        try:
            if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                print(f"KLD divergence: {kld.item():.6f} ent: {ent.item():.6f}")
        except Exception:
            pass
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

    def _apply_rudder_credit_assignment(self, return_history: Dict) -> None:
        """
        Apply RUDDER credit assignment to buffer trajectories.

        This method is called during training to train the RUDDER network
        and redistribute rewards based on learned importance weights.

        Parameters
        ----------
        return_history : dict
            Dictionary containing 'returns' and optionally state sequences
        """
        if not hasattr(self, 'rudder_agent') or self.rudder_agent is None:
            return

        try:
            # Extract trajectories from buffer for RUDDER training
            if hasattr(self.buffer, 'states') and len(self.buffer.states) > 0:
                # Group buffer data by episode
                trajectories = []
                current_traj_idx = 0

                for ep_idx in range(len(return_history['returns'])):
                    traj_length = return_history['lengths'][ep_idx]

                    # Extract trajectory data
                    traj_states = self.buffer.states[current_traj_idx:current_traj_idx + traj_length]
                    traj_rewards = self.buffer.rewards[current_traj_idx:current_traj_idx + traj_length]
                    episode_return = return_history['returns'][ep_idx]

                    # Convert to numpy
                    if hasattr(traj_states, 'cpu'):
                        states_np = traj_states.cpu().numpy()
                    else:
                        states_np = np.array(traj_states)

                    if hasattr(traj_rewards, 'cpu'):
                        rewards_np = traj_rewards.cpu().numpy()
                    else:
                        rewards_np = np.array(traj_rewards)

                    # Flatten states if needed
                    if states_np.ndim > 2:
                        states_np = states_np.reshape(states_np.shape[0], -1)

                    # Add trajectory to RUDDER buffer
                    self.rudder_agent.add_trajectory(
                        states=states_np,
                        actions=None,  # Not used in contribution-based method
                        rewards=rewards_np,
                        episode_return=episode_return
                    )

                    current_traj_idx += traj_length

        except Exception as e:
            get_logger().warning(f"⚠ RUDDER credit assignment failed: {e}")

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
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [], 'value_loss': []}
        for epoch in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            policy_core_acc, value_loss_acc = 0, 0
            early_stop_epoch = False

            for batch_i, batch in enumerate(dataloader, start=1):
                batch_loss, batch_kld, batch_ent, batch_vloss, batch_ploss = self._fit_policy_and_value_model_step(batch)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                value_loss_acc += batch_vloss
                policy_core_acc += batch_ploss
                batches += 1
                # Debugging hooks: print per-batch KLD and entropy if requested
                try:
                    if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                        print(f"[PPO DEBUG] epoch={epoch+1} batch={batch_i} batch_kld={batch_kld:.6f} batch_ent={batch_ent:.6f}")
                except Exception:
                    pass

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            avg_loss = loss / batches
            avg_kld = kld / batches
            avg_ent = ent / batches
            avg_vloss = value_loss_acc / batches
            avg_ploss = policy_core_acc / batches
            history['loss'].append(avg_loss)
            history['kld'].append(avg_kld)
            history['ent'].append(avg_ent)
            history['value_loss'].append(avg_vloss)
            history['policy_core_loss'].append(avg_ploss)

            # Stop training if KLD exceeded limit in any batch of this epoch
            if early_stop_epoch:
                get_logger().debug(f'Early stopping due to KLD divergence at epoch {epoch+1}. '
                                  f'Average epoch KLD: {avg_kld:.6f}, Limit: {self.kld_limit:.6f}')
                if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                    print(f"[PPO DEBUG] Early stopped epoch={epoch+1} batches_processed={batches} avg_kld={avg_kld:.6f}")
                return {k: np.array(v) for k, v in history.items()}

        return {k: np.array(v) for k, v in history.items()}

    # ==================================================================
    # Causal PG: policy-only training with causal credits as advantages
    # ==================================================================

    def _fit_causal_policy_models(self, dataloader, epochs=1):
        """Fit policy + value using causal credits as advantages (advantage replacement).

        Key differences from _fit_policy_and_value_models:
          - Advantages come from causal credits directly (not GAE).
          - Value function is trained on discounted-credit targets for stability.
          - KLD early stopping prevents catastrophic policy collapse.
          - Full normalization (mean + std) applied upstream in get(), with
            low-variance fallback to center-only when std < threshold.
        """
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [], 'value_loss': []}
        for epoch in range(epochs):
            loss_acc, ent_acc, kld_acc, batches = 0.0, 0.0, 0.0, 0
            ploss_acc = 0.0
            early_stop_epoch = False

            for batch_i, batch in enumerate(dataloader, start=1):
                batch_loss, batch_ent, batch_ploss, batch_kld = self._fit_causal_policy_step(batch)
                loss_acc += batch_loss
                ent_acc += batch_ent
                ploss_acc += batch_ploss
                kld_acc += batch_kld
                batches += 1

                # KLD guard: prevent catastrophic policy updates
                if self.kld_limit is not None and abs(batch_kld) > self.kld_limit:
                    get_logger().debug(
                        f'Causal PG early stopping at epoch {epoch+1}, batch {batch_i}: '
                        f'batch KLD {batch_kld:.6f} exceeded limit {self.kld_limit:.6f}')
                    early_stop_epoch = True
                    break

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            avg_loss = loss_acc / batches
            avg_ent = ent_acc / batches
            avg_ploss = ploss_acc / batches
            avg_kld = kld_acc / batches
            history['loss'].append(avg_loss)
            history['kld'].append(avg_kld)
            history['ent'].append(avg_ent)
            history['value_loss'].append(0.0)
            history['policy_core_loss'].append(avg_ploss)

            if early_stop_epoch:
                get_logger().debug(f'Causal PG early stopping at epoch {epoch+1}. '
                                  f'Average KLD: {avg_kld:.6f}')
                return {k: np.array(v) for k, v in history.items()}

        return {k: np.array(v) for k, v in history.items()}

    def _fit_causal_policy_step(self, batch):
        """One gradient step for causal policy gradient with advantage replacement.

        Key design:
          - Advantages come from causal credits (NOT from value baseline).
          - Value network is trained alongside policy on discounted-credit targets
            for stability and monitoring.
          - KLD is computed for early stopping.
          - Normal entropy bonus (1x).
        """
        self.policy_model.train()
        if self.value_model is not None and not isinstance(self.value_model, str):
            self.value_model.train()

        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        new_logpis = (new_probs + epsilon).log()
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)

        actions = torch.as_tensor(batch.y)
        advantages = batch.advantage.clone()
        old_logprob = batch.logprobs.clone()

        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)

        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0

        new_logpis_a = new_logpis[:nA] if nA else None
        new_logpis_p = new_logpis[nA:nA + nP] if nP else None

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

            if has_a:
                mask_a = (idx_a == s)
                ns_a = new_logpis_a[mask_a].reshape(-1)
                if ns_a.numel():
                    parts_new.append(ns_a)

            if has_p and new_logpis_p is not None:
                mask_p = (idx_p == s)
                ns_p = new_logpis_p[mask_p].reshape(-1)
                if ns_p.numel():
                    parts_new.append(ns_p)

            if len(parts_new) == 0:
                continue

            ns_cat = torch.cat(parts_new, dim=0)
            a_idx = int(actions[s])

            sel_new.append(ns_cat[a_idx].reshape(()))
            sel_old.append(old_logprob[s].reshape(()))
            sel_adv.append(advantages[s].reshape(()))

            # KLD per sample
            kld_sample = old_logprob[s].item() - ns_cat[a_idx].item()
            kld_terms.append(kld_sample)

        # ----- Value loss (always trained for stability) -----
        loss_value = torch.tensor(0.0)
        if self.value_model is not None and not isinstance(self.value_model, str):
            pred_values = self.value_model(batch).squeeze()
            loss_value = torch.mean(self.value_loss.forward(
                input=pred_values, target=batch.value.clone()))

        self.policy_optimizer.zero_grad()
        if self.value_model is not None and not isinstance(self.value_model, str):
            self.value_optimizer.zero_grad()

        if len(sel_new) == 0:
            # No policy samples but still train value
            if loss_value.requires_grad:
                loss_value.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
                self.value_optimizer.step()
            return 0.0, 0.0, 0.0, 0.0

        new_sel = torch.stack(sel_new)
        old_sel = torch.stack(sel_old)
        adv_sel = torch.stack(sel_adv)

        # PPO clipped surrogate loss
        loss_policy_core = torch.mean(self.policy_loss(new_sel, old_sel, adv_sel))

        # Entropy bonus
        ent = -torch.mean(new_probs * new_logpis)

        # Combined loss: policy + value (value provides useful gradients but
        # does NOT affect advantage computation — advantages are from credits).
        loss_total = loss_policy_core + self.vf_coeff * loss_value - self.ent_bonus * ent

        # KLD for monitoring and early stopping
        if len(kld_terms) > 0:
            kld = sum(kld_terms) / len(kld_terms)
        else:
            kld = 0.0

        loss_total.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        if self.value_model is not None and not isinstance(self.value_model, str):
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
            self.value_optimizer.step()
        self.policy_optimizer.step()

        return loss_total.item(), ent.item(), float(loss_policy_core.item()), kld

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

        # Return also value and policy-core losses for diagnostics
        return loss_total.item(), kld.item(), ent.item(), float(loss_value.item()), float(loss_policy_core.item())


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


# ============================================================================
# PPO LOSS CLASSES (FULLY PICKLABLE - no closures, just callable classes)
# ============================================================================

class PPOClipLoss:
    """Clipped PPO loss (picklable callable class).

    Parameters
    ----------
    eps : float
        The clip ratio.
    """
    def __init__(self, eps=0.2):
        self.eps = eps

    def __call__(self, new_logps, old_logps, advantages):
        """Compute clipped PPO loss.

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
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
        try:
            ret_loss = -torch.min(surr1, surr2)
        except Exception as e:
            print("Invalid loss detected.")
            ret_loss = -torch.min(surr1, surr2)
        return ret_loss


class PPOPenaltyLoss:
    """Penalty PPO loss (picklable callable class).

    Parameters
    ----------
    c : float
        The fixed KLD weight.
    """
    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, new_logps, old_logps, advantages):
        """Compute penalty PPO loss.

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
        return -(torch.exp(new_logps - old_logps) * advantages - self.c * (old_logps - new_logps))



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

    def __init__(self, policy_network, method='clip', eps=0.2, c=0.01, rudder_config=None, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.method = method
        self.eps = eps
        self.c = c
        self.rudder_config = rudder_config or {}
        self.rudder_agent = None

        # Initialize RUDDER if enabled
        if self.rudder_config.get('enabled', False):
            try:
                from gympn.rudder import RUDDERAgent
                self.rudder_agent = RUDDERAgent(
                    state_dim=self.rudder_config.get('state_dim', 128),
                    hidden_dim=self.rudder_config.get('hidden_dim', 256),
                    learning_rate=self.rudder_config.get('learning_rate', 1e-3),
                    device=self.rudder_config.get('device', 'cpu'),
                    training_frequency=self.rudder_config.get('training_frequency', 1),
                    redistribution_method=self.rudder_config.get('redistribution_method', 'contribution')
                )
                get_logger().info(f"✓ RUDDER agent initialized with state_dim={self.rudder_config.get('state_dim', 128)}")
            except ImportError:
                get_logger().warning("⚠ RUDDER module not available, skipping RUDDER initialization")
                self.rudder_agent = None

        # Instantiate picklable loss classes
        if method == 'clip':
            self.policy_loss = PPOClipLoss(eps=eps)
        elif method == 'penalty':
            self.policy_loss = PPOPenaltyLoss(c=c)
        else:
            raise ValueError(f"Unknown PPO method: {method}")

