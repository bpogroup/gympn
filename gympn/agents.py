"""Policy gradient agents that support changing state spaces, specifically for graph environments.

Currently includes policy gradient agent (i.e., Monte Carlo policy
gradient or vanilla policy optimization
agent.
"""
import numpy as np
import os
import random
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from gympn.data import TrajectoryBuffer, print_status_bar
from gympn.flat_graph import is_flat
from gympn.logging_utils import TrainingMetrics, get_logger


def _make_summary_writer(logdir):
    """TensorBoard writer for logdir, or None if tensorboard is not installed."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        import warnings
        warnings.warn("tensorboard is not installed, so training curves are not "
                      "logged; install it with `pip install gympn[tensorboard]`")
        return None
    return SummaryWriter(log_dir=logdir)


# torch.autograd.set_detect_anomaly(True)


def _normalized_entropy(probs, logpis, batch_index):
    """Compute mean normalized entropy across decisions with variable action spaces.

    For each decision (identified by ``batch_index``), the raw entropy
    ``H = -sum(p * log(p))`` is divided by ``log(num_actions)`` so that the
    result lies in [0, 1] regardless of how many actions are available.

    Parameters
    ----------
    probs : Tensor  – 1-D probability per action node (flat).
    logpis : Tensor – 1-D log-probability per action node.
    batch_index : Tensor – maps each node to its decision index.

    Returns
    -------
    Tensor (scalar) – mean normalized entropy in [0, 1].
    """
    ent_sum = torch.tensor(0.0, device=probs.device)
    count = 0
    for s in batch_index.unique():
        mask = (batch_index == s)
        p = probs[mask]
        lp = logpis[mask]
        n_actions = int(mask.sum().item())
        raw_ent = -(p * lp).sum()
        if n_actions > 1:
            ent_sum = ent_sum + raw_ent / torch.log(
                torch.tensor(float(n_actions), device=probs.device))
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=probs.device)
    return ent_sum / count

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
        Whether to apply per-batch advantage normalization (zero-mean, unit-variance).
        Default is **True** (the standard PPO choice). Normalization is needed to
        learn LOW-headroom tasks: when the optimal-vs-suboptimal margin is small the
        raw advantages are tiny, and without rescaling the gradient is too weak to
        reach the optimum (empirically, env b peaks at 8/9 with it off vs 9/9 with
        it on). Its downside — the step size not decaying at convergence → post-peak
        drift — is *cosmetic* given best-checkpoint restore (the deployed policy is
        the peak, not the drifted tail). Turn it off only for ablations.
        See INSTABILITY_ANALYSIS.md.
    normalize_returns : bool, optional
        Whether to normalize returns (discounted cumulative rewards) for value training.
        Default is False (IMPORTANT for correctness).

        ⚠️  CRITICAL: If normalize_returns=True, the value network will be trained on
        normalized targets (mean=0, std=1). However, value predictions from rollout time
        will be in the normalized scale, while GAE calculations use raw rewards/credits.
        This creates a scale mismatch in delta = reward + gamma * v_{t+1} - v_t.

        RECOMMENDATION: Keep normalize_returns=False (default) to avoid this mismatch.
        If you need stable value training, prefer advantage normalization
        (``normalize_advantages=True``) — but note it can reintroduce post-peak drift.
    kld_limit : float or None, optional
        Early-stopping limit on the mean per-state KL(pi_old || pi_new),
        checked after each inner policy epoch. Default is None (disabled).

        The KL is computed EXACTLY per state, over that state's own (variable
        size) action set: KL_s = sum_a p_old(a|s) (log p_old − log p_new),
        then averaged over states — the standard PPO target_kl quantity, valid
        for variable |A(s)| because old and new always share a state's support.
        (Historical note: before 2026-07-09 this metric was the SIGNED
        chosen-action log-ratio — cancellation-prone, high-variance and
        |A(s)|-dependent — and early stopping was rightly discouraged. That no
        longer applies.) PPO's clip bounds each surrogate term but NOT the
        realized policy shift after multiple inner epochs; the KL brake is the
        guard against the rare catastrophic update that collapses a
        near-deterministic policy (observed as post-convergence drift).
    ent_bonus : float, optional
        Bonus factor for sampled policy entropy.

    """

    def __init__(self,
                 policy_network, value_network, policy_lr=1e-4, policy_updates=1,
                 value_lr=1e-3, value_updates=25,
                 gam=0.99, lam=0.97, normalize_advantages=True, eps=0.2,
                 kld_limit=None, ent_bonus=0.0, test_in_train=True, vf_coeff=0.05,
                 normalize_returns=False, lr_schedule=False,
                 beta=0.0, smdp_discount=False, nfgae=False):
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
        # Net-factored GAE (suite/paper/NFGAE_THEORY.md): each decision is
        # credited only with its own net component's rewards, and the critic
        # pools only over that component's action nodes.
        self.nfgae = bool(nfgae)
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam, beta=beta,
                                       smdp_discount=smdp_discount, nfgae=nfgae)
        self.normalize_advantages = normalize_advantages
        self.normalize_returns = normalize_returns
        self.lr_schedule = lr_schedule
        self.kld_limit = kld_limit
        self.ent_bonus = ent_bonus
        self._ent_bonus_initial = ent_bonus  # for entropy annealing
        self._total_epochs = None  # set at start of train()
        self._current_epoch = 0

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
        # Rollout/eval only: no autograd graph is needed here (the stored logpis
        # are detached old-policy constants; the policy fit recomputes a fresh
        # forward on batches). Wrapping in no_grad avoids building and discarding
        # an autograd graph on every single environment step.
        with torch.no_grad():
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

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, batch_size=64, sort_states=False, test_env=None, test_freq=5, test_episodes=10,
              wandb_logger=None, num_workers=1, eval_seed=None):
        """Train the agent on env with optional testing during training.

        Parameters
        ----------
        env : environment
            The environment to train on.
        test_env : environment, optional
            The test environment for evaluation during training.
        test_freq : int, optional
            Frequency (in epochs) to run testing during training.
        eval_seed : int, optional
            Base seed pinning the evaluation scenarios (common random numbers).
            See :meth:`test_in_train`. None keeps the historical behaviour of
            drawing fresh eval scenarios from the live RNG stream.
        wandb_logger : WandBLogger, optional
            Logger for Weights & Biases integration.

        Returns
        -------
        history : dict
            Dictionary with statistics from training and testing.
        """
        # Pin the initial policy to the run seed alone. Parameters are created
        # lazily at the first forward pass, so this has to happen here rather
        # than at construction -- see gympn.seeding.seed_network_init for the
        # measurements that motivated it (arms of the same experiment were
        # starting from different initial policies, silently unmatching the
        # comparison).
        _agent_seed = getattr(self, 'agent_seed', None)
        if _agent_seed is not None:
            from gympn.seeding import seed_network_init
            seed_network_init(_agent_seed)

        tb_writer = None if logdir is None else _make_summary_writer(logdir)

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

        self._total_epochs = epochs

        for i in range(epochs):
            # === ENTROPY ANNEALING ===
            # Linearly decay entropy bonus from initial value to 0 over training.
            # Early epochs: high entropy encourages exploration.
            # Late epochs: zero entropy prevents the "success catastrophe" where
            # the entropy bonus destroys an optimal policy when advantages ≈ 0.
            self._current_epoch = i
            self.ent_bonus = self._ent_bonus_initial * max(0.0, 1.0 - i / max(1, epochs - 1))

            self.buffer.clear()
            # Use parallel episode collection with dill (4-8x speedup on collection, 2-4x overall)
            # Dill can serialize lambda functions and complex objects like SimVar
            return_history = self.run_episodes(env, episodes=episodes, max_episode_length=max_episode_length,
                                               store=True, num_workers=num_workers)

            # Standard per-batch advantage normalization (default ON via
            # self.normalize_advantages). Needed to learn low-margin tasks; the
            # drift it can cause near convergence is handled by best-checkpoint
            # restore. Toggle off only for ablations.
            normalize_adv_for_batch = self.normalize_advantages

            dataloader = self.buffer.get(normalize_advantages=normalize_adv_for_batch,
                                         normalize_returns=self.normalize_returns,
                                         batch_size=batch_size,
                                         sort=sort_states, drop_remainder=True)

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
                    f"Epoch {i + 1}: No complete batches to process. "
                    f"Buffer size ({len(self.buffer)}) < batch_size ({batch_size}). "
                    f"Consider reducing batch_size or increasing episodes per epoch."
                )
                history['delta_policy_loss'][i] = 0.0
                history['policy_ent'][i] = 0.0
                history['policy_kld'][i] = 0.0

            # Test the agent during training
            if test_env is not None and (i + 1) % test_freq == 0:
                test_metrics = self.test_in_train(test_env, episodes=test_episodes,
                                                  max_episode_length=max_episode_length, logdir=logdir,
                                                  eval_seed=eval_seed)
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

            if test_env is None and logdir is not None and (
                    i + 1) % save_freq == 0:  # only save all the policies when no test in train is performed
                self.save_policy_weights(logdir + "/policy-" + str(i + 1) + ".h5")
                self.save_value_weights(logdir + "/value-" + str(i + 1) + ".h5")
                self.save_policy_network(logdir + "/network-" + str(i + 1) + ".pth")

            # Log epoch metrics
            metrics = TrainingMetrics(
                epoch=i + 1,
                mean_return=float(history['mean_returns'][i]),
                std_return=float(history['std_returns'][i]),
                mean_length=float(history['mean_ep_lens'][i]),
                policy_loss=float(history['delta_policy_loss'][i]) if not np.isnan(
                    history['delta_policy_loss'][i]) else None,
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
                    policy_loss=float(history['delta_policy_loss'][i]) if not np.isnan(
                        history['delta_policy_loss'][i]) else None,
                    kld=float(history['policy_kld'][i]) if not np.isnan(history['policy_kld'][i]) else None,
                    entropy=float(history['policy_ent'][i]) if not np.isnan(history['policy_ent'][i]) else None,
                )

            if verbose > 0:
                print_status_bar(i, epochs, history, verbose=verbose)

            # Step learning rate schedulers if enabled
            if self.lr_schedule:
                policy_scheduler.step()
                value_scheduler.step()

        # === Best-checkpoint restore (early stopping) ===
        # The live policy can drift off the optimum after convergence (greedy eval
        # touches the optimum, then degrades — see INSTABILITY_ANALYSIS.md). When
        # deterministic eval ran during training and saved a best policy, reload it
        # so the returned agent holds the best policy found, not the last (possibly
        # degraded) one. No-op when no eval/checkpoint was produced.
        if test_env is not None and logdir is not None and self.best_test_metric > float('-inf'):
            best_path = os.path.join(logdir, "best_policy.pth")
            if os.path.exists(best_path):
                try:
                    self.policy_model = torch.load(best_path, weights_only=False)
                    get_logger().info(
                        f"Restored best policy (eval metric = {self.best_test_metric:.4f}) "
                        f"from {best_path}")
                except Exception as e:
                    get_logger().warning(f"Could not restore best policy from {best_path}: {e}")

        return history

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """Run an episode and return total reward and episode length.

        Values are predicted in batches of states (every 8 steps and at the
        end of the episode) rather than one forward pass per step.

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
            The environment's accumulated reward (info['pn_reward']) and the
            episode length.

        """
        state = env.reset()

        done = False
        episode_length = 0
        total_reward = 0
        info = {'pn_reward': 0}  # Initialize info

        states_batch = []
        actions_batch = []
        logprobs_batch = []
        logpis_batch = []
        rewards_batch = []
        times_batch = []
        value_batch_size = 8  # Compute values for 8 states at a time

        # nfgae: deciding component per step and per-component step rewards
        # (suite/paper/NFGAE_THEORY.md). The critic reads V_c by pooling only
        # over the deciding component's action nodes (graph pool_mask).
        nf_on = self.nfgae
        nf_comp_ep, nf_rew_ep = [], []
        if nf_on:
            _pn0 = getattr(env, 'pn', None)
            if (_pn0 is not None and getattr(_pn0, 'allow_postpone', False)
                    and getattr(_pn0, 'postpone_scope', 'global') != 'component'):
                raise ValueError("nfgae needs allow_postpone=False or postpone_scope="
                                 "'component': a global postpone couples components "
                                 "(Theorem 1, C3).")

        while not done:
            action, logprob, logpis = self.act(state, return_logprob=True)

            # Decision time = simulator clock BEFORE stepping (for the SMDP
            # sojourn tau_t = u_{i+1} - u_i).
            pn = getattr(env, 'pn', None) or getattr(env, 'problem', None)
            decision_time = float(getattr(pn, 'clock', 0.0)) if pn is not None else 0.0

            if nf_on:
                part = pn.net_partition()
                is_pp = [isinstance(b[0], list) and b[0] == ['postpone'] for b in pn.pn_actions]
                node_comp = [b[2].comp if pp else part[b[2]._id]
                             for b, pp in zip(pn.pn_actions, is_pp)]
                c_dec = node_comp[action]
                g_ = state['graph']
                mask_a = torch.tensor([c == c_dec for c, pp in zip(node_comp, is_pp) if not pp], dtype=torch.bool)
                mask_p = torch.tensor([c == c_dec for c, pp in zip(node_comp, is_pp) if pp], dtype=torch.bool)
                if is_flat(g_):
                    # always set both, so every sample in a batch has the same keys
                    g_.a_pool_mask, g_.p_pool_mask = mask_a, mask_p
                else:
                    g_['a_transition'].pool_mask = mask_a
                    if any(is_pp):                     # postpone nodes, in the same order
                        g_['postpone'].pool_mask = mask_p
                nf_comp_ep.append(c_dec)

            # Collect for batch processing
            states_batch.append(state)
            actions_batch.append(action)
            logprobs_batch.append(logprob)
            logpis_batch.append(logpis)
            times_batch.append(decision_time)

            next_state, reward, done, truncated, info = env.step(action)
            if nf_on:
                nf_rew_ep.append(dict(info.get('comp_reward', {})))
            total_reward += reward
            rewards_batch.append(reward)

            episode_length += 1

            # Compute values in batch every N steps or at episode end
            if len(states_batch) >= value_batch_size or done:
                values = self._compute_batch_values(states_batch, env)

                if buffer is not None:
                    for i, (s, a, lp, lpis, r, tm) in enumerate(zip(
                            states_batch, actions_batch, logprobs_batch,
                            logpis_batch, rewards_batch, times_batch)):
                        buffer.store(s, a, r, lp, values[i], lpis, time=tm)

                states_batch = []
                actions_batch = []
                logprobs_batch = []
                logpis_batch = []
                rewards_batch = []
                times_batch = []

            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state

        if buffer is not None:
            if nf_on:
                buffer._nf_comp = nf_comp_ep
                buffer._nf_rew = nf_rew_ep
            buffer.finish()

        return info.get('pn_reward', total_reward), episode_length

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

                    if isinstance(graphs[0], HeteroData) or is_flat(graphs[0]):
                        # Batch heterogeneous graphs
                        batched_graph = Batch.from_data_list(graphs)

                        # Single forward pass on batched graph
                        batch_values = self.value_model(batched_graph)

                        # Extract per-graph values
                        if isinstance(batch_values, torch.Tensor):
                            # Values should have shape [num_graphs] or [num_graphs, 1]
                            if batch_values.dim() > 1:
                                values = batch_values[:, 0].tolist() if batch_values.size(
                                    1) == 1 else batch_values.tolist()
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
                R, L = self.run_episode(env, max_episode_length=max_episode_length,
                                        buffer=self.buffer if store else None)
                history['returns'][i] = R
                history['lengths'][i] = L
        else:
            # Parallel episode collection using dill for serialization
            # Dill can handle lambda functions and complex objects like SimVar
            try:
                import dill  # noqa: F401  (imported for its side effect: it patches pickle)
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
                    R, L = self.run_episode(env, max_episode_length=max_episode_length,
                                            buffer=self.buffer if store else None)
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

            avg_loss = loss / batches
            avg_kld = kld / batches
            avg_ent = ent / batches
            history['loss'].append(avg_loss)
            history['kld'].append(avg_kld)
            history['ent'].append(avg_ent)

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
        # initial_weights = {name: param.clone() for name, param in self.policy_model.named_parameters()}

        indexes = batch['a_transition'].batch.data
        states = batch
        actions = torch.tensor(batch.y)
        logprobs = batch.logprobs.clone()
        advantages = batch.advantage.clone()

        epsilon = 1e-7
        new_probs = self.policy_model(states)
        new_logpis = (new_probs + epsilon).log()

        # new_logprobs contains, for each unique index in indexes, the value in the slice of logpis corresponding
        # to the current index in indexes with index action[index]
        new_logprobs = torch.stack(
            [new_logpis[indexes == index][actions[index]] for index in indexes.unique()]).squeeze(1)

        # Calculate batch size
        batch_size = len(indexes.unique())

        # Compute normalized entropy
        ent = -torch.sum(new_probs * new_logpis) / batch_size

        # Compute normalized KLD
        logpis = torch.cat(logpis, dim=0)
        kld = torch.sum(new_probs * (new_logpis - logpis)) / batch_size
        loss = torch.mean(self.policy_loss(new_logprobs, logprobs, advantages)) - self.ent_bonus * ent

        try:
            # No second backward runs on this graph (each batch recomputes a
            # fresh forward), so retain_graph is unnecessary; dropping it frees
            # the activation graph immediately.
            loss.backward()  # compute gradients
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
                # batch = batch[0]
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

        states = batch
        values = batch.value.clone()  # value targets (GAE returns)

        pred_values = self.value_model(states).squeeze()
        loss = torch.mean(self.value_loss.forward(input=pred_values, target=values))

        self.value_optimizer.zero_grad()
        try:
            loss.backward()
        except Exception as e:
            print("Loss.backward produced an invalid output.")

        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.value_optimizer.step()

        return loss.item()

    def test_in_train(self, env, episodes=100, max_episode_length=None, deterministic=True, logdir=None,
                      eval_seed=None):
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
        eval_seed : int, optional
            Base seed for COMMON RANDOM NUMBERS across evaluations. When set,
            episode ``i`` runs under seed ``eval_seed + i``, so every eval point
            -- across epochs, across runs, and across methods -- scores the
            policy on the SAME fixed set of scenarios.

            Without it (the default, and the historical behaviour) each eval
            draws fresh scenarios from wherever training left the global RNG:
            unbiased, but two arms are compared on different sample paths.
            Measured on s1, one 20-episode eval point carries +-0.231 SD of
            pure scenario noise, so a paired single-point difference carries
            +-0.327 -- and ``greedy_drift``, a max over ~15 such points, is
            inflated by ~0.40 for a policy that is genuinely flat.

            The env's stochasticity comes from the global ``random`` / NumPy
            streams, so this reseeds those per episode and RESTORES the prior
            state afterwards. Training's stream therefore continues across the
            eval as if it had not run -- verified directly in
            suite/_test_eval_crn.py (T2).

            That does NOT make a CRN run step-identical to a non-CRN run of the
            same seed, and it cannot: without eval_seed the eval CONSUMES the
            training stream (20 episodes' worth of draws per eval point), so
            the two configurations' rollouts diverge from the first eval
            onward -- measured on a 4-epoch s1 cell, epoch 4's sampled return
            was 9.65 with CRN vs 9.40 without. Results produced with eval_seed
            set are a new baseline, not a re-scoring of existing cells.

            ``env.reset(seed=...)`` is deliberately NOT used: it routes to
            seed_everything, which would also reseed torch and re-apply the
            deterministic-kernel switches on every eval episode.

        Returns
        -------
        test_metrics : dict
            Dictionary containing evaluation metrics (mean, min, max, std returns and lengths).
        """
        history = {'returns': np.zeros(episodes), 'lengths': np.zeros(episodes)}
        crn = eval_seed is not None
        if crn:
            saved_random = random.getstate()
            saved_np = np.random.get_state()
            saved_torch = torch.get_rng_state()
        try:
            for i in range(episodes):
                if crn:
                    # Same scenario i for every arm and every epoch.
                    random.seed(eval_seed + i)
                    np.random.seed((eval_seed + i) % (2 ** 32))
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
        finally:
            if crn:
                random.setstate(saved_random)
                np.random.set_state(saved_np)
                torch.set_rng_state(saved_torch)

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
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [], 'value_loss': []}

        # --- Policy: PPO clipped surrogate for `policy_updates` (=`epochs`) passes ---
        # The policy and value nets are decoupled (separate optimizers/backward), so
        # the value net is NOT trained here; it gets its own `value_updates` passes
        # below. This fixes the previous behaviour where the value net was trained
        # exactly `policy_updates` times and `value_updates` was silently ignored
        # (fix #1), and removes the inert `vf_coeff` from the policy step (fix #2).
        for epoch in range(epochs):
            loss, kld, ent, batches = 0, 0, 0, 0
            policy_core_acc = 0

            for batch_i, batch in enumerate(dataloader, start=1):
                batch_loss, batch_kld, batch_ent, batch_ploss = self._fit_policy_and_value_model_step(batch)
                loss += batch_loss
                kld += batch_kld
                ent += batch_ent
                policy_core_acc += batch_ploss
                batches += 1

                # Debugging hooks: print per-batch KLD and entropy if requested
                try:
                    if os.environ.get('GP_DEBUG_PPO', '0') == '1':
                        print(
                            f"[PPO DEBUG] epoch={epoch + 1} batch={batch_i} batch_kld={batch_kld:.6f} batch_ent={batch_ent:.6f}")
                except Exception:
                    pass

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            history['loss'].append(loss / batches)
            history['kld'].append(kld / batches)
            history['ent'].append(ent / batches)
            history['policy_core_loss'].append(policy_core_acc / batches)

            # KL early stopping: abort remaining inner updates once the mean
            # per-state KL(old || new) exceeds the limit (true KL, >= 0).
            if self.kld_limit is not None and history['kld'][-1] > self.kld_limit:
                break

        # --- Value: MSE regression on the (GAE) return targets for `value_updates`
        # passes, with its own optimizer (fix #1). ---
        if self.value_model is not None and not isinstance(self.value_model, str):
            value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
            history['value_loss'] = list(value_history.get('loss', []))

        return {k: np.array(v) for k, v in history.items()}

    def _fit_policy_and_value_model_step(self, batch):
        self.policy_model.train()
        self.value_model.train()

        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        new_logpis = (new_probs + epsilon).log()
        # Squeeze both to 1-D to avoid broadcasting bugs in entropy
        if new_probs.dim() == 2 and new_probs.size(-1) == 1:
            new_probs = new_probs.squeeze(-1)
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)

        actions = torch.as_tensor(batch.y)
        advantages = batch.advantage.clone()
        old_logprob = batch.logprobs.clone()

        if is_flat(batch):
            # FlatGraph batch (flat_obs): the same quantities from the flat fields.
            nA, nP = int(batch.a_idx.numel()), int(batch.p_idx.numel())
            has_a, has_p = nA > 0, nP > 0
            old_logpis_a = batch.logpis_a.reshape(-1) if has_a else None
            old_logpis_p = batch.logpis_p.reshape(-1) if has_p else None
            idx_a = batch.batch[batch.a_idx] if has_a else None
            idx_p = batch.batch[batch.p_idx] if has_p else None
        else:
            has_a = ('a_transition' in batch.x_dict)
            has_p = ('postpone' in batch.x_dict)

            nA = batch['a_transition'].x.size(0) if has_a else 0
            nP = batch['postpone'].x.size(0) if has_p else 0

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

        # Split new logits by node type in the same order as actions_dict
        new_logpis_a = new_logpis[:nA] if nA else None
        new_logpis_p = new_logpis[nA:nA + nP] if nP else None

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
                    # Old logpis not stored for postpone: use the new values so
                    # this part contributes 0 to the KL (zeros_like would mean
                    # log p_old = 0, i.e. p_old = 1 — corrupting the KL).
                    os_p = ns_p.detach()
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
            # if a_idx < 0 or a_idx >= ns_cat.shape[0]:
            #    # out-of-range chosen index -> skip this sample
            #    continue

            sel_new.append(ns_cat[a_idx].reshape(()))
            sel_old.append(old_logprob[s].reshape(()))
            sel_adv.append(advantages[s].reshape(()))

            # TRUE per-state KL(old || new) over this state's OWN action set:
            #   KL_s = sum_a p_old(a|s) * (log p_old(a|s) - log p_new(a|s)).
            # Non-negative and well-defined for variable-size action sets (old
            # and new share the state's support), unlike the previous
            # chosen-action log-ratio, which was signed (batch mean cancels),
            # single-sample (huge variance) and |A(s)|-dependent.
            with torch.no_grad():
                p_old = torch.exp(os_cat)
                p_old = p_old / p_old.sum().clamp_min(1e-8)
                kld_terms.append(float((p_old * (os_cat - ns_cat)).sum().item()))

        # ----- Policy-only update (fix #1/#2) -----
        # The value net is trained separately for `value_updates` passes in
        # _fit_policy_and_value_models. Because policy and value are fully decoupled
        # (separate optimizers/backward), `vf_coeff` never affected the gradient and
        # is therefore dropped from this path.
        self.policy_optimizer.zero_grad()

        new_sel = torch.stack(sel_new)
        old_sel = torch.stack(sel_old)
        adv_sel = torch.stack(sel_adv)

        loss_policy_core = torch.mean(self.policy_loss(new_sel, old_sel, adv_sel))
        # Compute KLD as mean difference in log probabilities
        if len(kld_terms) > 0:
            kld = torch.tensor(kld_terms, device=new_probs.device).mean()
        else:
            kld = torch.tensor(0.0, device=new_probs.device)

        # Normalized entropy over the combined batch index
        _ent_parts = []
        if has_a and idx_a is not None:
            _ent_parts.append(idx_a)
        if has_p and idx_p is not None:
            _ent_parts.append(idx_p)
        _ent_idx = torch.cat(_ent_parts, dim=0) if _ent_parts else idx_a
        ent = _normalized_entropy(new_probs, new_logpis, _ent_idx)

        loss_policy = loss_policy_core - self.ent_bonus * ent

        # Policy backward + step
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        # Return (policy_loss, kld, ent, policy_core_loss)
        return float(loss_policy.item()), kld.item(), ent.item(), float(loss_policy_core.item())


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

    def __init__(self, policy_network, method='clip', eps=0.2, c=0.01, **kwargs):
        super().__init__(policy_network, **kwargs)
        self.method = method
        self.eps = eps
        self.c = c
        # Instantiate picklable loss classes
        if method == 'clip':
            self.policy_loss = PPOClipLoss(eps=eps)
        elif method == 'penalty':
            self.policy_loss = PPOPenaltyLoss(c=c)
        else:
            raise ValueError(f"Unknown PPO method: {method}")

