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
                 causal_scheme='lrq', causal_pg=False,
                 causal_rl=False, causal_beta=0.0, causal_mu=0.0,
                 smdp_discount=False, causal_aux_coef=0.5,
                 qoff_network=None, qlin_network=None, cf_config=None):
        self.policy_model = policy_network
        self.policy_loss = NotImplementedError
        self.policy_optimizer = torch.optim.Adam(params=list(policy_network.parameters()),
                                                 lr=policy_lr)
        self.policy_updates = policy_updates

        self.value_model = value_network
        self.value_loss = torch.nn.MSELoss()
        self.value_optimizer = torch.optim.Adam(params=list(value_network.parameters()), lr=value_lr)
        self.value_updates = value_updates

        # LRQ-v3 / LQI: learned per-action-node Q heads (see networks.
        # HeteroQOff). qoff = off-lineage component (lrq3, lqi); qlin =
        # lineage component (lqi only, where the FULL Q is learned and the
        # policy is improved MPO/AWR-style instead of via PPO advantages).
        self.causal_scheme = causal_scheme
        self.qoff_model = qoff_network
        self.qoff_optimizer = (torch.optim.Adam(qoff_network.parameters(), lr=value_lr)
                               if qoff_network is not None else None)
        self.qlin_model = qlin_network
        self.qlin_optimizer = (torch.optim.Adam(qlin_network.parameters(), lr=value_lr)
                               if qlin_network is not None else None)

        self.lam = lam
        self.gam = gam
        self.causal_pg = causal_pg
        self.causal_rl = bool(causal_rl)
        # LVA: weight of the critic's auxiliary lineage-credit regression in
        # the value loss (value_loss + coef * aux_loss). Only read when the
        # value network was built with aux_head=True (scheme 'lva').
        self.causal_aux_coef = float(causal_aux_coef)
        self.buffer = TrajectoryBuffer(gam=gam, lam=lam,
                                       causal_scheme=causal_scheme,
                                       causal_pg=causal_pg,
                                       causal_rl=causal_rl,
                                       causal_beta=causal_beta,
                                       causal_mu=causal_mu,
                                       smdp_discount=smdp_discount)
        self.normalize_advantages = normalize_advantages
        self.normalize_returns = normalize_returns  # New parameter
        self.lr_schedule = lr_schedule  # New parameter
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

        # G1 forked counterfactual preferences (gympn/counterfactual.py,
        # CAUSAL_LINEAGE_RETHINK.md §7.2). None => disabled (exact PPO floor:
        # no forks, no aux loss, nothing else changes). Keys: fork_prob,
        # reps, gate, lookahead, max_forks, coef, updates, beta.
        self.cf_config = cf_config
        self._cf_prefs = []
        self._cf_diag = []   # per-fork {'gap','se','passed'} mechanism telemetry
        self._cf_records = []  # decomposed mode: unresolved fork records

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
            self._cf_prefs = []
            self._cf_diag = []
            self._cf_records = []
            # Use parallel episode collection with dill (4-8x speedup on collection, 2-4x overall)
            # Dill can serialize lambda functions and complex objects like SimVar
            return_history = self.run_episodes(env, episodes=episodes, max_episode_length=max_episode_length,
                                               store=True, num_workers=num_workers)

            # === RUDDER: fit the return-predicting LSTM on this epoch's
            # trajectories (added per-episode in run_episode) ===
            if getattr(self, 'rudder_agent', None) is not None and self.rudder_agent.should_train():
                rudder_loss = self.rudder_agent.train(num_epochs=5)
                if wandb_logger and i % 10 == 0:
                    wandb_logger.log({'rudder/loss': rudder_loss}, step=i)
                self.rudder_agent.step_epoch()
                get_logger().info(f"  [RUDDER] Training loss: {rudder_loss:.4f}")

            # Standard per-batch advantage normalization (default ON via
            # self.normalize_advantages). Needed to learn low-margin tasks; the
            # drift it can cause near convergence is handled by best-checkpoint
            # restore. Toggle off only for ablations.
            normalize_adv_for_batch = self.normalize_advantages

            dataloader = self.buffer.get(normalize_advantages=normalize_adv_for_batch,
                                         normalize_returns=self.normalize_returns,
                                         batch_size=batch_size,
                                         sort=sort_states, drop_remainder=True)

            # LCV mechanism telemetry: the epoch's adaptive CV coefficient and
            # the fractional advantage-variance reduction it achieved (both
            # computed in buffer.get(); zeros for every other scheme).
            history.setdefault('cv_coef', np.zeros(epochs))
            history.setdefault('cv_var_reduction', np.zeros(epochs))
            history['cv_coef'][i] = getattr(self.buffer, 'last_cv_coef', 0.0)
            history['cv_var_reduction'][i] = getattr(self.buffer, 'last_cv_var_reduction', 0.0)

            # Route to appropriate training method:
            # - LQI (Q-native): fitted lineage-decomposed Q + advantage-
            #   weighted policy iteration (no PPO clip, no advantages from
            #   the buffer).
            # - Causal RL/PG: per-decision Q-sample advantages (LRQ family).
            # - Standard PPO: uses GAE advantages and discounted returns.
            if getattr(self, 'causal_scheme', None) == 'lqi' and self.causal_rl:
                policy_history = self._fit_lqi_models(dataloader)
            elif getattr(self, 'causal_scheme', None) == 'lcv' and self.causal_rl:
                # LCV: 100% standard PPO on the CV-adjusted advantages (the
                # adjustment happened in buffer.get()); plus the v_off state
                # head regressed on the measured off-lineage returns.
                policy_history = self._fit_policy_and_value_models(
                    dataloader, epochs=self.policy_updates)
                if getattr(self, 'qoff_model', None) is not None:
                    self._fit_voff_model(dataloader, epochs=self.value_updates)
            elif getattr(self, 'causal_scheme', None) == 'lva' and self.causal_rl:
                # LVA: 100% standard PPO (plain SMDP-GAE advantages from the
                # buffer, no CV, no credit advantages). The lineage enters only
                # inside _fit_value_model_step, as the critic aux head's
                # regression target (batch.qlin_target) — representation
                # shaping, policy gradient untouched.
                policy_history = self._fit_policy_and_value_models(
                    dataloader, epochs=self.policy_updates)
            elif self.causal_rl or self.causal_pg:
                policy_history = self._fit_causal_policy_models(dataloader, epochs=self.policy_updates)
            else:
                policy_history = self._fit_policy_and_value_models(dataloader, epochs=self.policy_updates)

            # G1: SNR-gated counterfactual preference aux pass (after the
            # PPO epochs; coefficient annealed to 0 over training = floor).
            if getattr(self, 'cf_config', None) is not None:
                cf_stats = self._fit_cf_preferences()
                # Mechanism telemetry: the paired SE and the gate-pass rate are
                # where the lineage-restriction claim lives (a tighter SE at
                # equal gap => more forks clear the gate for the same compute).
                diag = getattr(self, '_cf_diag', [])
                n_forks = len(diag)
                mean_se = float(np.mean([d['se'] for d in diag])) if diag else 0.0
                mean_gap = float(np.mean([abs(d['gap']) for d in diag])) if diag else 0.0
                pass_rate = float(np.mean([d['passed'] for d in diag])) if diag else 0.0
                for key in ('cf_prefs', 'cf_loss', 'cf_forks', 'cf_se',
                            'cf_gap', 'cf_pass_rate'):
                    history.setdefault(key, np.zeros(epochs))
                history['cf_prefs'][i] = cf_stats['n']
                history['cf_loss'][i] = cf_stats['loss']
                history['cf_forks'][i] = n_forks
                history['cf_se'][i] = mean_se
                history['cf_gap'][i] = mean_gap
                history['cf_pass_rate'][i] = pass_rate
                rs = getattr(self, '_cf_resolve_stats', None)
                if rs is not None:
                    # decomposed mode: the honest numbers are the regression's
                    # held-out R^2 (does the lineage feature actually predict
                    # the opportunity-cost channel?) and the resolved SE
                    history.setdefault('cf_r2', np.zeros(epochs))
                    history['cf_r2'][i] = (rs['r2'] if np.isfinite(rs['r2'])
                                           else 0.0)
                    history['cf_se'][i] = rs['se']
                    history['cf_pass_rate'][i] = rs['pass_rate']
                    get_logger().info(
                        f"  [CF] forks={n_forks} prefs={cf_stats['n']} "
                        f"mode={rs['mode']} r2={rs['r2']:.3f} "
                        f"nfit={rs.get('n_fit', 0)} "
                        f"pass={rs['pass_rate']:.0%} se={rs['se']:.4f} "
                        f"coef={cf_stats['coef']:.3f} loss={cf_stats['loss']:.4f}")
                else:
                    get_logger().info(
                        f"  [CF] forks={n_forks} prefs={cf_stats['n']} "
                        f"pass={pass_rate:.0%} |gap|={mean_gap:.3f} se={mean_se:.3f} "
                        f"coef={cf_stats['coef']:.3f} loss={cf_stats['loss']:.4f}")

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
                                                  max_episode_length=max_episode_length, logdir=logdir)
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
        times_batch = []
        value_batch_size = 8  # Compute values for 8 states at a time

        # RUDDER baseline: per-step (feature, reward) sequence for the LSTM
        # return predictor; consumed at episode end.
        rudder_on = getattr(self, 'rudder_agent', None) is not None and buffer is not None
        rudder_feats, rudder_rewards = [], []

        qoff_batch = []
        qoff_on = getattr(self, 'qoff_model', None) is not None and buffer is not None

        # G1 counterfactual forking: training rollouts only (buffer present),
        # capped per episode. Forks snapshot/restore env.pn, so the main
        # trajectory is untouched.
        cf_on = getattr(self, 'cf_config', None) is not None and buffer is not None
        cf_forks_done = 0

        while not done:
            action, logprob, logpis = self.act(state, return_logprob=True)

            # Decision time u_i = simulator clock BEFORE stepping (used for the
            # SMDP sojourn tau_t = u_{i+1} - u_i in causal time-discounting).
            pn = getattr(env, 'pn', None) or getattr(env, 'problem', None)
            decision_time = float(getattr(pn, 'clock', 0.0)) if pn is not None else 0.0

            if cf_on and cf_forks_done < self.cf_config['max_forks']:
                from gympn.counterfactual import maybe_fork
                forked, pref, diag = maybe_fork(self, env, state, action,
                                                logpis, self.cf_config)
                if forked:
                    cf_forks_done += 1
                if diag is not None:
                    self._cf_diag.append(diag)
                if pref is not None:
                    # decomposed mode yields fork RECORDS (resolved into
                    # preferences at epoch end, once the indirect-channel
                    # regression can be pooled); other modes yield preferences
                    if self.cf_config.get('decompose', False):
                        self._cf_records.append(pref)
                    else:
                        self._cf_prefs.append(pref)

            # Collect for batch processing
            states_batch.append(state)
            actions_batch.append(action)
            logprobs_batch.append(logprob)
            logpis_batch.append(logpis)
            times_batch.append(decision_time)
            if rudder_on:
                rudder_feats.append(self._rudder_features(state))
            if qoff_on:
                # Rollout-time auxiliary prediction (old parameters, frozen at
                # collection): q_off(s, a_taken) for lrq3/lqi; the state-only
                # centering v_off(s) for lcv (scalar HeteroCritic head).
                try:
                    with torch.no_grad():
                        qv = self.qoff_model(state).reshape(-1)
                    if getattr(self, 'causal_scheme', None) == 'lcv':
                        qoff_batch.append(float(qv[0]) if qv.numel() else 0.0)
                    else:
                        qoff_batch.append(float(qv[action]) if action < qv.numel() else 0.0)
                except Exception:
                    qoff_batch.append(0.0)

            next_state, reward, done, truncated, info = env.step(action)
            rewards_batch.append(reward)
            if rudder_on:
                rudder_rewards.append(float(reward))

            episode_length += 1
            total_reward += reward

            # Compute values in batch every N steps or at episode end
            if len(states_batch) >= value_batch_size or done:
                values = self._compute_batch_values(states_batch, env)

                # Store all buffered transitions
                if buffer is not None:
                    for i, (s, a, lp, lpis, r, tm) in enumerate(zip(
                            states_batch, actions_batch, logprobs_batch,
                            logpis_batch, rewards_batch, times_batch)):
                        buffer.store(s, a, r, lp, values[i], lpis,
                                     token_ids=None, time=tm,
                                     qoff=(qoff_batch[i] if qoff_on else None))

                # Clear batches for next iteration
                states_batch = []
                actions_batch = []
                logprobs_batch = []
                logpis_batch = []
                rewards_batch = []
                times_batch = []
                qoff_batch = []

            if max_episode_length is not None and episode_length > max_episode_length:
                break
            state = next_state

        if buffer is not None:
            if rudder_on and rudder_feats:
                # RUDDER baseline: add this episode to the LSTM's training set,
                # redistribute its rewards with the CURRENT predictor (early
                # epochs => near-uniform, standard RUDDER warm-up behaviour;
                # 'contribution' conserves the episode return exactly), and
                # feed the redistributed rewards through the ORDINARY GAE path.
                feats_np = np.asarray(rudder_feats, dtype=np.float32)
                rews_np = np.asarray(rudder_rewards, dtype=np.float32)
                self.rudder_agent.add_trajectory(
                    states=feats_np, actions=None, rewards=rews_np,
                    episode_return=float(rews_np.sum()))
                try:
                    red = self.rudder_agent.redistribute_rewards(feats_np, rews_np)
                    buffer.finish(credits=[float(x) for x in red],
                                  mode="replace_rewards")
                except Exception as e:
                    get_logger().warning(f"[RUDDER] redistribution failed ({e}); "
                                         f"falling back to raw rewards")
                    buffer.finish(credits=None)
            elif (self.causal_rl and 'eligibility_credits' in info
                  and info['eligibility_credits'] is not None):
                # NOTE the self.causal_rl guard: the ENV may record causal
                # traces while the AGENT stays on the standard SMDP-GAE path
                # (scheme 'cfpl' does exactly this — it needs the lineage DAG
                # for its forked counterfactual returns, but its policy
                # gradient must remain plain PPO). Behaviour-preserving for
                # every other method, where agent.causal_rl == env.causal_rl.
                # Diagnostic: if environment signals debugging, print causal trace stats
                try:
                    pn = getattr(env, 'pn', None) or getattr(env, 'problem', None)
                    if pn is not None and getattr(pn, '_debugging', False) and pn.causal_rl:
                        ct = pn.causal_trace
                        tok_count, trans_count = ct.stats()
                        # compute a quick credit sample to check sizes
                        try:
                            sample_cr = ct.redistribute_rewards(scheme='lrq')
                        except Exception as e:
                            sample_cr = None
                            import warnings
                            warnings.warn(
                                f"[CAUSAL-DIAG] tokens={tok_count}, transitions={trans_count}, ep_steps={episode_length}, redis_len={len(sample_cr) if sample_cr is not None else 'ERR'}, redis_sum={sum(sample_cr) if sample_cr is not None else 'ERR'}")
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

    # ==================================================================
    # LQI: Lineage-Q Iteration (Q-native consumer of the trace credits)
    # ==================================================================
    # Motivation (see CAUSAL_LQI_QNATIVE.md): PPO's clip + multi-epoch reuse
    # amplifies small-but-CONSISTENT advantages to epsilon-sized policy moves
    # -> premature commitment (measured on s1); LRQ-v3's cold-start race
    # corrupted advantages while its head was untrained. LQI removes the
    # policy-gradient advantage entirely: fit the decomposed Q from trace
    # targets FIRST each epoch, then improve the policy by advantage-weighted
    # regression (AWR): maximize E[w * log pi(a|s)], w = exp(A_std / TAU)
    # clipped at W_MAX, A = q(s,a) - mean_{a' available} q(s,a'). The
    # weighted-BC form is KL-regularized policy iteration - proportionate
    # moves, no clip saturation, and an untrained Q merely yields ~uniform
    # weights (harmless warm-up) instead of corrupted gradients.
    _LQI_TAU = 1.0        # temperature on per-batch STANDARDIZED advantages
    _LQI_WMAX = 20.0      # AWR weight clip

    def _fit_lqi_models(self, dataloader):
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [],
                   'value_loss': [], 'qlin_loss': [], 'qoff_loss': []}

        # 1. Q heads first (fresh Q before the policy is weighted by it).
        #    q_lin target = full mc_q sample - off target (= lrq2 credit).
        qlin_hist = self._fit_qhead(
            self.qlin_model, self.qlin_optimizer, dataloader,
            epochs=self.value_updates,
            get_target=lambda b: b.value - b.qoff_target)
        qoff_hist = self._fit_qhead(
            self.qoff_model, self.qoff_optimizer, dataloader,
            epochs=self.value_updates,
            get_target=lambda b: b.qoff_target)
        history['qlin_loss'] = qlin_hist['loss']
        history['qoff_loss'] = qoff_hist['loss']

        # 2. AWR policy epochs.
        for epoch in range(self.policy_updates):
            loss_acc = kld_acc = ent_acc = core_acc = 0.0
            batches = 0
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                b_loss, b_kld, b_ent, b_core = self._fit_lqi_policy_step(batch)
                loss_acc += b_loss
                kld_acc += b_kld
                ent_acc += b_ent
                core_acc += b_core
                batches += 1
            if batches == 0:
                get_logger().no_batches_warning()
                continue
            history['loss'].append(loss_acc / batches)
            history['kld'].append(kld_acc / batches)
            history['ent'].append(ent_acc / batches)
            history['policy_core_loss'].append(core_acc / batches)
            if self.kld_limit is not None and history['kld'][-1] > self.kld_limit:
                break

        return {k: np.array(v) for k, v in history.items()}

    def _fit_qhead(self, model, optimizer, dataloader, epochs, get_target):
        """Generic per-action-node regression: select the taken action's node
        output per sample and MSE against get_target(batch)."""
        history = {'loss': []}
        if model is None:
            return history
        for epoch in range(epochs):
            loss_acc, batches = 0.0, 0
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                model.train()
                out = model(batch)
                if out.dim() == 2 and out.size(-1) == 1:
                    out = out.squeeze(-1)
                sel, tgt = self._select_taken_nodes(batch, out, get_target(batch))
                if sel is None:
                    continue
                loss = torch.mean((sel - tgt) ** 2)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_acc += float(loss.item())
                batches += 1
            if batches:
                history['loss'].append(loss_acc / batches)
        return history

    @staticmethod
    def _per_sample_slices(batch, vec):
        """Yield (sample_id, concatenated per-node slice) in the policy's
        [a_transition; postpone] node order, per sample of the batch."""
        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)
        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0
        vec_a = vec[:nA] if nA else None
        vec_p = vec[nA:nA + nP] if nP else None
        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None
        for s in (idx_a.unique() if has_a else idx_p.unique()):
            parts = []
            if has_a:
                v = vec_a[idx_a == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if has_p and vec_p is not None:
                v = vec_p[idx_p == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if parts:
                yield int(s), torch.cat(parts, dim=0)

    def _select_taken_nodes(self, batch, vec, targets):
        """(taken-node outputs, aligned targets) across the batch's samples."""
        actions = torch.as_tensor(batch.y)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        sel, tgt = [], []
        for s, cat in self._per_sample_slices(batch, vec):
            a_idx = int(actions[s])
            if 0 <= a_idx < cat.numel():
                sel.append(cat[a_idx].reshape(()))
                tgt.append(targets[s].reshape(()))
        if not sel:
            return None, None
        return torch.stack(sel), torch.stack(tgt)

    def _fit_lqi_policy_step(self, batch):
        """One AWR step: w = exp(A_std / TAU) with A = q(s,a_taken) minus the
        per-state mean of q over the AVAILABLE actions (availability-aware
        centering), loss = -mean(w * log pi(a_taken|s)) - ent_bonus * H."""
        self.policy_model.train()
        epsilon = 1e-7
        new_probs = self.policy_model(batch)
        new_logpis = (new_probs + epsilon).log()
        if new_probs.dim() == 2 and new_probs.size(-1) == 1:
            new_probs = new_probs.squeeze(-1)
        if new_logpis.dim() == 2 and new_logpis.size(-1) == 1:
            new_logpis = new_logpis.squeeze(-1)

        with torch.no_grad():
            q_all = self.qlin_model(batch) + self.qoff_model(batch)
            if q_all.dim() == 2 and q_all.size(-1) == 1:
                q_all = q_all.squeeze(-1)

        actions = torch.as_tensor(batch.y)
        old_logprob = batch.logprobs.clone()

        sel_logp, advs, kld_terms = [], [], []
        # Old per-node logpis for the true per-state KL monitor.
        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)
        old_a = batch['a_transition'].logpis if has_a else None
        if old_a is not None and old_a.dim() == 2 and old_a.size(-1) == 1:
            old_a = old_a.squeeze(-1)
        old_p = None
        if has_p and hasattr(batch['postpone'], 'logpis'):
            old_p = batch['postpone'].logpis
            if old_p is not None and old_p.dim() == 2 and old_p.size(-1) == 1:
                old_p = old_p.squeeze(-1)
        old_vec = None
        if old_a is not None:
            old_vec = torch.cat([old_a, old_p], dim=0) if old_p is not None else old_a

        new_slices = dict(self._per_sample_slices(batch, new_logpis))
        q_slices = dict(self._per_sample_slices(batch, q_all))
        old_slices = (dict(self._per_sample_slices(batch, old_vec))
                      if old_vec is not None else {})

        for s, ns_cat in new_slices.items():
            q_cat = q_slices.get(s)
            if q_cat is None or q_cat.numel() != ns_cat.numel():
                continue
            a_idx = int(actions[s])
            if not (0 <= a_idx < ns_cat.numel()):
                continue
            sel_logp.append(ns_cat[a_idx].reshape(()))
            advs.append((q_cat[a_idx] - q_cat.mean()).reshape(()))
            os_cat = old_slices.get(s)
            if os_cat is not None and os_cat.numel() == ns_cat.numel():
                with torch.no_grad():
                    p_old = torch.exp(os_cat)
                    p_old = p_old / p_old.sum().clamp_min(1e-8)
                    kld_terms.append(float((p_old * (os_cat - ns_cat)).sum().item()))

        if not sel_logp:
            return 0.0, 0.0, 0.0, 0.0

        logp = torch.stack(sel_logp)
        A = torch.stack(advs)
        A = (A - A.mean()) / (A.std(unbiased=False) + 1e-8)
        w = torch.clamp(torch.exp(A / self._LQI_TAU), max=self._LQI_WMAX).detach()

        core = -torch.mean(w * logp)

        _ent_parts = []
        if has_a:
            _ent_parts.append(batch['a_transition'].batch.data)
        if has_p:
            _ent_parts.append(batch['postpone'].batch.data)
        ent = _normalized_entropy(new_probs, new_logpis, torch.cat(_ent_parts, dim=0))

        loss = core - self.ent_bonus * ent
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        kld = sum(kld_terms) / len(kld_terms) if kld_terms else 0.0
        return float(loss.item()), kld, float(ent.item()), float(core.item())

    def _fit_qoff_model(self, dataloader, epochs=1):
        """LRQ-v3: fit the off-lineage per-action-node head on trace-computed
        targets (mc_q credit − lrq2 credit of the TAKEN action per step)."""
        history = {'loss': []}
        for epoch in range(epochs):
            loss_acc, batches = 0.0, 0
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                step_loss = self._fit_qoff_model_step(batch)
                loss_acc += step_loss
                batches += 1
            if batches:
                history['loss'].append(loss_acc / batches)
        return history

    def _fit_qoff_model_step(self, batch):
        """One regression step: select the taken action's node output per
        sample (same [a_transition; postpone] ordering as the policy) and MSE
        it against the off-lineage target."""
        self.qoff_model.train()
        out = self.qoff_model(batch)
        if out.dim() == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)

        has_a = ('a_transition' in batch.x_dict)
        has_p = ('postpone' in batch.x_dict)
        nA = batch['a_transition'].x.size(0) if has_a else 0
        nP = batch['postpone'].x.size(0) if has_p else 0
        out_a = out[:nA] if nA else None
        out_p = out[nA:nA + nP] if nP else None
        idx_a = batch['a_transition'].batch.data if has_a else None
        idx_p = batch['postpone'].batch.data if has_p else None

        actions = torch.as_tensor(batch.y)
        targets = batch.qoff_target
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)

        sel, tgt = [], []
        for s in (idx_a.unique() if has_a else idx_p.unique()):
            parts = []
            if has_a:
                v = out_a[idx_a == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if has_p and out_p is not None:
                v = out_p[idx_p == s].reshape(-1)
                if v.numel():
                    parts.append(v)
            if not parts:
                continue
            cat = torch.cat(parts, dim=0)
            a_idx = int(actions[s])
            if 0 <= a_idx < cat.numel():
                sel.append(cat[a_idx].reshape(()))
                tgt.append(targets[s].reshape(()))
        if not sel:
            return 0.0

        loss = torch.mean((torch.stack(sel) - torch.stack(tgt)) ** 2)
        self.qoff_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qoff_model.parameters(), 1.0)
        self.qoff_optimizer.step()
        return float(loss.item())

    def _fit_voff_model(self, dataloader, epochs=1):
        """LCV: regress the state-only v_off head on the measured off-lineage
        returns (batch.qoff_target). Better centering => more of the CV's
        variance is removable; a bad head only shrinks c_hat, never biases."""
        for epoch in range(epochs):
            for batch in dataloader:
                if not hasattr(batch, 'qoff_target'):
                    continue
                self.qoff_model.train()
                pred = self.qoff_model(batch).squeeze()
                tgt = batch.qoff_target
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)
                if tgt.dim() == 0:
                    tgt = tgt.unsqueeze(0)
                if pred.numel() != tgt.numel():
                    continue
                loss = torch.mean((pred - tgt) ** 2)
                self.qoff_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.qoff_model.parameters(), 1.0)
                self.qoff_optimizer.step()

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
        """Fit value model on one batch of data.

        LVA: when the critic has a lineage aux head AND the batch carries
        qlin_target (both only exist for causal_scheme='lva'), the loss is
            MSE(V, gae_returns) + causal_aux_coef * MSE(V_aux, qlin_target).
        The aux head predicts the per-decision lineage credit from the shared
        encoding — an auxiliary representation task. V's own target stays the
        unbiased GAE return, so any lineage bias stops at the encoder."""
        self.value_model.train()

        # indexes = batch['a_transition'].batch.data
        states = batch
        values = batch.value.clone()  # discounted returns

        aux_on = (getattr(self.value_model, 'lineage_aux_head', None) is not None
                  and hasattr(batch, 'qlin_target'))
        if aux_on:
            pred_values, pred_aux = self.value_model.forward_with_aux(states)
            pred_values = pred_values.squeeze()
            pred_aux = pred_aux.squeeze()
            aux_tgt = batch.qlin_target
            if pred_aux.dim() == 0:
                pred_aux = pred_aux.unsqueeze(0)
            if aux_tgt.dim() == 0:
                aux_tgt = aux_tgt.unsqueeze(0)
            loss = torch.mean(self.value_loss.forward(input=pred_values, target=values))
            if pred_aux.numel() == aux_tgt.numel():
                loss = loss + self.causal_aux_coef * torch.mean((pred_aux - aux_tgt) ** 2)
        else:
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

    def _fit_cf_preferences(self):
        """G1 aux pass: pairwise logistic loss on the policy's log-probs for
        this epoch's SNR-gated counterfactual preferences.

        loss = -log sigmoid(logpi(winner) - logpi(loser)) per preference
        (the softmax normalization cancels in the difference, so this is the
        logit-difference DPO-style objective). Coefficient = cf coef *
        linear anneal to 0 over training: the floor is exact PPO by
        construction once the anneal completes. Runs AFTER the PPO policy
        epochs, so it is outside the KL early stop — kept safe by the small
        preference count, the gate, and the anneal.
        """
        cfg = self.cf_config
        # Decomposed mode: resolve this epoch's fork records into preferences
        # first — the indirect channel is a regression pooled over all of
        # them, so it cannot be decided fork-by-fork.
        self._cf_resolve_stats = None
        if cfg.get('decompose', False):
            from gympn.counterfactual import resolve_decomp_preferences
            recs = getattr(self, '_cf_records', [])
            # Rolling cross-epoch window: one epoch's forks cannot validate a
            # 5-parameter fit, and the occupancy->opportunity-cost relation
            # drifts slowly enough to pool.
            if not hasattr(self, '_cf_pool'):
                from collections import deque
                self._cf_pool = deque(maxlen=int(cfg.get('fit_window', 400)))
            self._cf_pool.extend(recs)
            resolved, rstats = resolve_decomp_preferences(
                recs, cfg['gate'], min_r2=cfg.get('min_r2', 0.05),
                fit_pool=self._cf_pool)
            self._cf_prefs = resolved
            self._cf_resolve_stats = rstats
        prefs = getattr(self, '_cf_prefs', [])
        if cfg.get('anneal', True):
            total = max(1, (self._total_epochs or 1) - 1)
            coef = cfg['coef'] * max(0.0, 1.0 - self._current_epoch / total)
        else:
            # X10 falsifier follow-up (mechanism probe): constant pressure,
            # floor knowingly sacrificed — tests the ceiling hypothesis
            # against the strongest version of G1.
            coef = cfg['coef']
        if not prefs or coef <= 0.0:
            return {'n': len(prefs), 'loss': 0.0, 'coef': coef}
        self.policy_model.train()
        last_loss = 0.0
        for _ in range(cfg['updates']):
            self.policy_optimizer.zero_grad()
            losses = []
            for p in prefs:
                pi = self.policy_model(p['state'])
                logp = pi.log().reshape(-1)
                if p['winner'] >= logp.numel() or p['loser'] >= logp.numel():
                    continue
                losses.append(-torch.nn.functional.logsigmoid(
                    logp[p['winner']] - logp[p['loser']]))
            if not losses:
                return {'n': len(prefs), 'loss': 0.0, 'coef': coef}
            loss = coef * torch.stack(losses).mean()
            loss.backward()
            self.policy_optimizer.step()
            last_loss = float(loss.item())
        return {'n': len(prefs), 'loss': last_loss, 'coef': coef}

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

    def _rudder_features(self, state):
        """Featurize a graph observation for the RUDDER LSTM: the marking
        vector (token count per node type, in the fixed metadata order set by
        the rudder_config's 'node_types'). Cheap, Markov-ish and constant-dim
        regardless of which places are currently occupied."""
        g = state['graph'] if isinstance(state, dict) and 'graph' in state else state
        feats = []
        for nt in getattr(self, '_rudder_node_types', []) or []:
            try:
                feats.append(float(g[nt].x.size(0)) if nt in g.node_types else 0.0)
            except Exception:
                feats.append(0.0)
        return feats

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

    # ==================================================================
    # Causal PG: policy-only training with causal credits as advantages
    # ==================================================================

    def _fit_causal_policy_models(self, dataloader, epochs=1):
        """Fit policy + value using causal credits with GAE(γ=1, λ).

        Mathematical basis — Credits-as-Rewards with GAE(γ=1, λ):
        ──────────────────────────────────────────────────────────
        In causal RL mode, the reward redistribution produces per-action credits
        c_t where Σ_t c_t = episode return.  These credits are treated as step
        rewards and processed with standard GAE but using γ=1 (no discounting
        on credits):

          • Value targets:  V_target(t) = Σ_{k≥t} c_k  (sum of future credits)
            With γ=1, V(s_t) = c_t + V(s_{t+1}) — the Bellman equation holds.

          • Advantages:  GAE(γ=1, λ)
            δ_t = c_t + V(s_{t+1}) − V(s_t), A_t = Σ_l λ^l δ_{t+l}
            Multi-step structure prevents oversmoothing where V(s_t) ≈ E[c_t|s_t]
            would kill the gradient signal prematurely.

          • Policy gradient:  ∇J ≈ Σ_t A_t ∇log π(a_t|s_t)

        Why GAE instead of pure credit-baseline (A_t = c_t − V(s_t)):
          With pure credit-baseline (equivalent to λ=0), V quickly learns to
          predict E[c_t|s_t], making advantages ≈ 0 and halting learning even
          for suboptimal policies.  With λ>0, GAE requires V to predict the
          entire future trajectory correctly before advantages vanish.
        """
        history = {'loss': [], 'kld': [], 'ent': [], 'policy_core_loss': [], 'value_loss': []}

        # --- Policy: PPO clipped surrogate on causal GAE advantages for
        # `policy_updates` (=`epochs`) passes. Value is trained separately below so
        # that `value_updates` is honoured (fix #1) and the inert `vf_coeff` is
        # dropped (fix #2) — mirrors the standard PPO path. ---
        for epoch in range(epochs):
            loss_acc, kld_acc, ent_acc, batches = 0.0, 0.0, 0.0, 0
            ploss_acc = 0.0

            for batch_i, batch in enumerate(dataloader, start=1):
                # 4-tuple: (policy_loss, kld, ent, policy_core_loss)
                batch_loss, batch_kld, batch_ent, batch_ploss = self._fit_causal_policy_step(batch)
                loss_acc += batch_loss
                kld_acc += batch_kld
                ent_acc += batch_ent
                ploss_acc += batch_ploss
                batches += 1

            if batches == 0:
                get_logger().no_batches_warning()
                continue

            history['loss'].append(loss_acc / batches)
            history['kld'].append(kld_acc / batches)
            history['ent'].append(ent_acc / batches)
            history['policy_core_loss'].append(ploss_acc / batches)

            # KL early stopping: abort remaining inner updates once the mean
            # per-state KL(old || new) exceeds the limit (true KL, >= 0).
            if self.kld_limit is not None and history['kld'][-1] > self.kld_limit:
                break

        # --- Value: MSE regression on the return-to-go-over-credits targets for
        # `value_updates` passes, with its own optimizer (fix #1). ---
        if self.value_model is not None and not isinstance(self.value_model, str):
            value_history = self._fit_value_model(dataloader, epochs=self.value_updates)
            history['value_loss'] = list(value_history.get('loss', []))

        # --- LRQ-v3: off-lineage head regression on (mc_q - lrq2) targets. ---
        if getattr(self, 'qoff_model', None) is not None:
            qoff_hist = self._fit_qoff_model(dataloader, epochs=self.value_updates)
            history['qoff_loss'] = list(qoff_hist.get('loss', []))

        return {k: np.array(v) for k, v in history.items()}

    def _fit_causal_policy_step(self, batch):
        """One gradient step for causal policy gradient with GAE(γ=1, λ).

        Credits-as-Rewards with GAE — no oversmoothing:
          • batch.advantage = GAE advantages (γ=1, λ)    [set by finish()]
          • batch.value     = Σ_{k≥t} c_k                [sum of future credits]
          • Policy loss: PPO clipped surrogate on GAE advantages
          • Value loss:  MSE(V(s_t), Σ_{k≥t} c_k) — learns expected remaining credit
          • Combined:    L = L_policy + vf_coeff * L_value − ent_bonus * H(π)

        Returns
        -------
        tuple of 5 floats: (loss_total, kld, ent, value_loss, policy_core_loss)
            Same signature as _fit_policy_and_value_model_step for consistency.
        """
        self.policy_model.train()
        if self.value_model is not None and not isinstance(self.value_model, str):
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
            parts_old = []

            if has_a:
                mask_a = (idx_a == s)
                ns_a = new_logpis_a[mask_a].reshape(-1)
                if ns_a.numel():
                    parts_new.append(ns_a)
                    if old_logpis_a is not None:
                        parts_old.append(old_logpis_a[mask_a].reshape(-1))
                    else:
                        parts_old.append(ns_a.detach())

            if has_p and new_logpis_p is not None:
                mask_p = (idx_p == s)
                ns_p = new_logpis_p[mask_p].reshape(-1)
                if ns_p.numel():
                    parts_new.append(ns_p)
                    if old_logpis_p is not None:
                        parts_old.append(old_logpis_p[mask_p].reshape(-1))
                    else:
                        # Missing old logpis contribute 0 to the KL (using the
                        # new values); zeros_like would mean p_old = 1.
                        parts_old.append(ns_p.detach())

            if len(parts_new) == 0:
                continue

            ns_cat = torch.cat(parts_new, dim=0)
            os_cat = torch.cat(parts_old, dim=0)
            a_idx = int(actions[s])

            sel_new.append(ns_cat[a_idx].reshape(()))
            sel_old.append(old_logprob[s].reshape(()))
            sel_adv.append(advantages[s].reshape(()))

            # TRUE per-state KL(old || new) over this state's OWN action set
            # (see _fit_policy_and_value_model_step for why the previous
            # chosen-action log-ratio was not a usable trust-region signal).
            with torch.no_grad():
                p_old = torch.exp(os_cat)
                p_old = p_old / p_old.sum().clamp_min(1e-8)
                kld_terms.append(float((p_old * (os_cat - ns_cat)).sum().item()))

        # ----- Policy-only update (fix #1/#2) -----
        # The value net is trained separately for `value_updates` passes in
        # _fit_causal_policy_models. Because policy and value are fully decoupled
        # (separate optimizers/backward), `vf_coeff` never affected the gradient and
        # is dropped from this path.
        self.policy_optimizer.zero_grad()

        if len(sel_new) == 0:
            # No policy samples this batch (value is trained separately below).
            return 0.0, 0.0, 0.0, 0.0

        new_sel = torch.stack(sel_new)
        old_sel = torch.stack(sel_old)
        adv_sel = torch.stack(sel_adv)

        # PPO clipped surrogate loss
        loss_policy_core = torch.mean(self.policy_loss(new_sel, old_sel, adv_sel))

        # Normalized entropy bonus (action-space invariant, in [0,1])
        _ent_parts = []
        if has_a and idx_a is not None:
            _ent_parts.append(idx_a)
        if has_p and idx_p is not None:
            _ent_parts.append(idx_p)
        _ent_idx = torch.cat(_ent_parts, dim=0) if _ent_parts else idx_a
        ent = _normalized_entropy(new_probs, new_logpis, _ent_idx)

        loss_policy = loss_policy_core - self.ent_bonus * ent

        # KLD for monitoring
        if len(kld_terms) > 0:
            kld = sum(kld_terms) / len(kld_terms)
        else:
            kld = 0.0

        # Policy backward + step
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.policy_optimizer.step()

        # Return (policy_loss, kld, ent, policy_core_loss)
        return float(loss_policy.item()), kld, ent.item(), float(loss_policy_core.item())

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

        # Normalized entropy using combined batch index (same as causal path)
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
                get_logger().info(
                    f"[RUDDER] agent initialized with state_dim={self.rudder_config.get('state_dim', 128)}")
            except ImportError:
                get_logger().warning("[RUDDER] module not available, skipping initialization")
                self.rudder_agent = None
        # Fixed node-type order for the marking-vector featurization
        # (must match state_dim; threaded by train.make_agent from metadata).
        self._rudder_node_types = self.rudder_config.get('node_types', [])

        # Instantiate picklable loss classes
        if method == 'clip':
            self.policy_loss = PPOClipLoss(eps=eps)
        elif method == 'penalty':
            self.policy_loss = PPOPenaltyLoss(c=c)
        else:
            raise ValueError(f"Unknown PPO method: {method}")

