"""Direction A agent — AlphaZero-over-AEPN (AEPN_NATIVE_LEARNING.md §3).

``MCTSAgent`` is a thin subclass of ``DCLAgent``: it swaps the DCL flat-rollout
planner for the PUCT tree search in ``mcts_planner`` and inherits everything
else — ``run_episode`` (plan → act → store target_pi/q_first), the distillation
step (cross-entropy toward the search-improved visit distribution = the
AlphaZero policy-improvement operator, plus value regression to the root
Q-means), and the whole ``train`` loop. That reuse is the point: the search is
the only new moving part.

The Direction B conflict graph is built once from the env and passed to the
planner so it can collapse structurally-forced (all-commuting) nodes and spend
its simulation budget on genuine contested decisions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from gympn.agents_dcl import DCLAgent, map_target_to_action_nodes
from gympn.mcts_planner import MCTSConfig, mcts_target_pi, build_conflict_adjacency


class MCTSAgent(DCLAgent):
    def __init__(self, policy_network, value_network=None,
                 mcts_cfg: Optional[MCTSConfig] = None, **kwargs):
        # DCLAgent wants a planner_cfg; we don't use it, but pass a default so
        # its __init__ is happy, then override the planning call below.
        super().__init__(policy_network, value_network=value_network, **kwargs)
        self.mcts_cfg = mcts_cfg or MCTSConfig()
        self._conflict_adj = None  # built lazily from the env's static net

    def _ensure_conflict_adj(self, env):
        if self._conflict_adj is None:
            try:
                self._conflict_adj = build_conflict_adjacency(env.pn)
            except Exception:
                self._conflict_adj = {}
        return self._conflict_adj

    @torch.no_grad()
    def _act_with_dcl(self, env, state_graph, lineage=None,
                      deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Same signature/return as DCLAgent so run_episode is inherited, but
        the improved target comes from PUCT search instead of flat rollouts."""
        adj = self._ensure_conflict_adj(env)
        critic = (self.value_model
                  if (self.value_model is not None
                      and not isinstance(self.value_model, str)) else None)
        target_pi, stats, enabled = mcts_target_pi(
            env, state_graph, self.policy_model, self.mcts_cfg,
            conflict_adj=adj, critic=critic)
        self._last_plan_stats = stats

        if len(enabled) == 0:
            return -1, torch.tensor([]), torch.tensor([])

        if len(enabled) == 1 or deterministic:
            idx_in_enabled = int(np.argmax(target_pi)) if len(enabled) > 1 else 0
        else:
            idx_in_enabled = int(np.random.choice(len(enabled), p=target_pi))
        chosen_env_idx = int(enabled[idx_in_enabled])

        tpi_node = map_target_to_action_nodes(state_graph, enabled, target_pi)
        q_first = torch.tensor(stats.get("means", [0.0]), dtype=torch.float32)
        return chosen_env_idx, tpi_node, q_first