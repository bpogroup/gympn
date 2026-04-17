# gympn/agents_dcl.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gympn.agents import Agent  # base class
from gympn.dcl_planner import PlannerConfig, compute_target_pi  # (or sequential_halving_target_pi)
from gympn.logging_utils import get_logger, TrainingMetrics


# --- Utilities ---------------------------------------------------------------

def _get_graph_obj(state) -> Any:
    """
    Return the graph object expected by the actor.
    Supports either a raw HeteroData or a dict with key 'graph'.
    """
    if isinstance(state, dict) and "graph" in state:
        return state["graph"]
    return state


def map_target_to_action_nodes(state_graph, enabled_ids: List[int], target_pi: np.ndarray) -> torch.Tensor:
    """
    Build a per-action-node vector aligned to the actor output.
    Zeros everywhere; 'target_pi' on enabled action indices (env positions).
    Handles optional 'postpone' node type if present.
    """
    g = _get_graph_obj(state_graph)

    if 'a_transition' not in g.node_types:
        raise AssertionError("Graph lacks 'a_transition' node type.")

    n_actions = g['a_transition'].num_nodes
    if 'postpone' in g.node_types and g['postpone'].num_nodes > 0:
        n_actions += g['postpone'].num_nodes

    out = torch.zeros(n_actions, dtype=torch.float32)
    if len(enabled_ids) > 0 and target_pi is not None and len(target_pi) == len(enabled_ids):
        idx = torch.tensor(enabled_ids, dtype=torch.long)
        out[idx] = torch.tensor(target_pi, dtype=torch.float32)
    return out


# --- DCL Agent ---------------------------------------------------------------

class DCLAgent(Agent):
    """
    Deep Controlled Learning Agent.

    - Per step: builds improved target policy (target_pi) via CRN-averaged short rollouts.
    - Stores target_pi (per-node) + q_first (diagnostic means) in buffer.
    - Trains with CE toward target_pi + optional value regression.

    Requirements on the environment:
      - env.get_state(), env.set_state(snapshot), env.set_seed(seed),
      - env.enabled_actions(state_graph), env.step(action) -> (state, reward, done, trunc, info)

    Buffer:
      - buffer.store(...) accepts target_pi and q_first (newer buffer);
        falls back gracefully if not available.
    """

    def __init__(
        self,
        policy_network,
        value_network=None,
        planner_cfg: Optional[PlannerConfig] = None,
        vf_coeff: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            policy_network=policy_network,
            value_network=value_network,
            **kwargs
        )
        self.vf_coeff = float(vf_coeff)
        self.planner_cfg = planner_cfg or PlannerConfig()
        self._eps = 1e-7  # small epsilon for safe logs

    @torch.no_grad()
    def _act_with_dcl(
        self,
        env,
        state_graph,
        lineage=None,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Compute target_pi using the DCL planner, choose an action, and
        build per-node target tensor + a q_first vector for logging/critic target.
        """
        target_pi, stats, enabled = compute_target_pi(
            env, state_graph, self.policy_model, self.planner_cfg, lineage=lineage
        )

        if len(enabled) == 0:
            return -1, torch.tensor([]), torch.tensor([])

        if len(enabled) == 1 or deterministic:
            idx_in_enabled = int(np.argmax(target_pi)) if len(enabled) > 1 else 0
            chosen_env_idx = int(enabled[idx_in_enabled])
        else:
            idx_in_enabled = int(np.random.choice(len(enabled), p=target_pi))
            chosen_env_idx = int(enabled[idx_in_enabled])

        tpi_node = map_target_to_action_nodes(state_graph, enabled, target_pi)
        q_first = torch.tensor(stats.get('means', [0.0]), dtype=torch.float32)
        return chosen_env_idx, tpi_node, q_first

    def run_episode(self, env, max_episode_length=None, buffer=None):
        """
        Executes an episode using DCL planning at each step.
        """
        state = env.reset()
        done = False
        episode_length = 0
        total_reward = 0.0

        lineage = getattr(env, "pn", None)

        while not done:
            # 1) DCL plan + action selection
            action_env_idx, tpi_node, q_first = self._act_with_dcl(
                env, state, lineage=lineage, deterministic=False
            )

            if action_env_idx < 0:
                break

            # 2) Policy forward (for logging)
            probs = self.policy_model(state)  # per action-node probs
            logpis = (probs + self._eps).log()

            # 3) Defensive bound check before stepping
            env_len = len(getattr(env.pn, "pn_actions", []))
            if not (0 <= action_env_idx < env_len):
                enabled_now = env.enabled_actions(state)
                if len(enabled_now) == 0:
                    break
                action_env_idx = int(enabled_now[0])

            # 4) Step env
            next_state, reward, done, truncated, info = env.step(action_env_idx)

            # 5) Store transition
            if buffer is not None:
                # Scalar value for logging/baseline; safe and detached
                val_scalar = 0.0
                if self.value_model is not None and not isinstance(self.value_model, str):
                    v = self.value(state)  # tensor
                    try:
                        val_scalar = float(v.detach().reshape(-1).mean().item())
                    except Exception:
                        val_scalar = 0.0

                flat = logpis.flatten()
                chosen_logprob = float(flat[action_env_idx].item()) if flat.numel() > action_env_idx >= 0 else 0.0

                try:
                    buffer.store(
                        state, action_env_idx, reward,
                        chosen_logprob, val_scalar,
                        logpis,
                        token_ids=info.get('produced_token_ids', []),
                        target_pi=tpi_node,
                        q_first=q_first
                    )
                except TypeError:
                    # Backward compatibility: buffer without target_pi/q_first
                    buffer.store(
                        state, action_env_idx, reward,
                        chosen_logprob, val_scalar,
                        logpis,
                        token_ids=info.get('produced_token_ids', [])
                    )

            episode_length += 1
            total_reward += float(reward)

            if max_episode_length is not None and episode_length > max_episode_length:
                break

            state = next_state

        # Finish trajectory (pass lineage credits if your env exposes them)
        if buffer is not None:
            credits = None
            if isinstance(state, dict) and 'eligibility_credits' in state:
                credits = state['eligibility_credits']
            buffer.finish(credits)

        return total_reward, episode_length

    # --- Training step: CE to target_pi + optional value loss ------------------

    def _fit_policy_and_value_model_step(self, batch):
        """
        One optimization step:
            loss_total = CE(target_pi, logprobs)
                         + vf_coeff * MSE(V(batch), mean(q_first))
                         - ent_bonus * entropy

        Assumptions:
          - batch.target_pi is a concatenated per-node vector aligned with model outputs.
          - batch.q_first is present (either vector per sample or scalar).
        """
        # Always return a triple to avoid NoneType unpack errors.
        try:
            self.policy_model.train()
            has_v = self.value_model is not None and not isinstance(self.value_model, str)
            if has_v:
                self.value_model.train()

            eps = self._eps

            # Forward policy over concatenated node batch
            probs = self.policy_model(batch)                  # per-node probabilities (concat across samples)
            logprobs = (probs + eps).log()

            # Required: target_pi in batch
            if not hasattr(batch, 'target_pi'):
                raise RuntimeError("Batch missing 'target_pi' for DCL training.")

            tpi = batch.target_pi
            # Align shapes for CE
            if tpi.shape != probs.shape:
                m = min(tpi.numel(), probs.numel())
                tpi = tpi.view(-1)[:m]
                logprobs = logprobs.view(-1)[:m]
                probs = probs.view(-1)[:m]

            # Cross-entropy toward the planner-improved target
            ce = -(tpi * logprobs).sum() / max(1.0, float(tpi.numel()))

            # Entropy regularization (normalized for variable action spaces)
            if hasattr(batch, 'batch') and batch.batch is not None:
                from gympn.agents import _normalized_entropy
                ent = _normalized_entropy(probs.view(-1), logprobs.view(-1), batch.batch.data)
            else:
                ent = -(probs * logprobs).sum() / max(1.0, float(probs.numel()))

            # Optional value regression to rollout target (mean of q_first if vector)
            loss_v = torch.tensor(0.0, dtype=torch.float32, device=probs.device)
            if has_v and hasattr(batch, 'q_first'):
                q = batch.q_first
                target_v = q.mean(dim=-1) if q.dim() > 1 else q
                # Shapes: pred_v -> (N,1) or (N,) depending on your value head
                pred_v = self.value_model(batch)

                # --- SHAPE FIX: turn both into 1-D vectors of same length ---
                pred_v = pred_v.reshape(-1)
                target_v = target_v.reshape(-1)
                m = min(pred_v.shape[0], target_v.shape[0])
                if m == 0:
                    loss_v = torch.tensor(0.0, dtype=torch.float32, device=probs.device)
                else:
                    loss_v = F.mse_loss(pred_v[:m], target_v[:m])

            loss_total = ce + self.vf_coeff * loss_v - self.ent_bonus * ent

            # Optimize
            self.policy_optimizer.zero_grad()
            if has_v:
                self.value_optimizer.zero_grad()

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            if has_v:
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)

            if has_v:
                self.value_optimizer.step()
            self.policy_optimizer.step()

            # KLD metric proxy (forward CE)
            kld_metric = ce.detach()
            return float(loss_total.item()), float(kld_metric.item()), float(ent.item())

        except Exception as e:
            # Make the error obvious but still avoid returning None
            print(f"[DCL] _fit_policy_and_value_model_step error: {e}")
            # Return something sensible so the training loop can continue/log
            return float('nan'), float('nan'), float('nan')

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, batch_size=64, sort_states=False, test_env=None, test_freq=5, test_episodes=10,
              wandb_logger=None, num_workers=4):
        """
        Train the DCL agent with optional testing during training.

        Parameters
        ----------
        env : environment
            The environment to train on.
        episodes : int
            Number of episodes per epoch.
        epochs : int
            Number of training epochs.
        max_episode_length : int, optional
            Maximum steps per episode.
        verbose : int
            Verbosity level.
        save_freq : int
            Frequency to save policy/value weights.
        logdir : str, optional
            Directory for logging.
        batch_size : int
            Batch size for training.
        sort_states : bool
            Whether to sort states in dataloader.
        test_env : environment, optional
            Environment for testing during training.
        test_freq : int
            Frequency (in epochs) to run testing.
        test_episodes : int
            Number of episodes for testing.
        wandb_logger : optional
            Logger for Weights & Biases integration.
        num_workers : int
            Number of parallel workers for episode collection.

        Returns
        -------
        history : dict
            Training history with metrics.
        """
        from torch.utils.tensorboard import SummaryWriter

        tb_writer = None if logdir is None else SummaryWriter(log_dir=logdir)
        history = {
            'mean_returns': np.zeros(epochs),
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
            'policy_kld': np.zeros(epochs),
        }

        if test_env is not None:
            history.update({
                'test_mean_returns': np.zeros(epochs // test_freq + 1),
                'test_min_returns': np.zeros(epochs // test_freq + 1),
                'test_max_returns': np.zeros(epochs // test_freq + 1),
                'test_std_returns': np.zeros(epochs // test_freq + 1),
            })

        for i in range(epochs):
            self.buffer.clear()
            return_history = self.run_episodes(
                env, episodes=episodes, max_episode_length=max_episode_length, store=True
            )

            dataloader = self.buffer.get(
                normalize_advantages=self.normalize_advantages,
                batch_size=batch_size,
                sort=sort_states,
                drop_remainder=False
            )

            # Training step: optimize policy and value
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

            # Handle case where no complete batches were processed
            if len(policy_history['loss']) > 0:
                history['delta_policy_loss'][i] = policy_history['loss'][-1] - self.previous_policy_loss
                self.previous_policy_loss = policy_history['loss'][-1]
                history['policy_ent'][i] = policy_history['ent'][-1]
                history['policy_kld'][i] = policy_history['kld'][-1]
            else:
                history['delta_policy_loss'][i] = 0.0
                history['policy_ent'][i] = 0.0
                history['policy_kld'][i] = 0.0

            # Test the agent during training
            if test_env is not None and (i + 1) % test_freq == 0:
                test_metrics = self.test_in_train(
                    test_env, episodes=test_episodes, max_episode_length=max_episode_length, logdir=logdir
                )
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

            # Save weights periodically
            if test_env is None and logdir is not None and (i + 1) % save_freq == 0:
                self.save_policy_weights(f"{logdir}/policy-{i + 1}.pth")
                self.save_value_weights(f"{logdir}/value-{i + 1}.pth")

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

        return {k: v for k, v in history.items()}

