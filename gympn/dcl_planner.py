
# gympn/planners/dcl_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch

@dataclass
class PlannerConfig:
    horizon: int = 5
    rollouts_per_action: int = 32
    gamma: float = 1.0
    temperature: float = 1.0    # softmax temp for target_pi
    use_crn: bool = True
    use_lineage: bool = True

def _shape_reward(reward: float, info: Dict[str, Any], lineage: Optional[object]) -> float:
    # If you prefer episodic redistribution in buffer.finish(), just return reward here.
    if lineage is None:
        return float(reward)
    if isinstance(info, dict) and 'eligibility_credits' in info:
        #disabled for now
        pass
        #return float(reward) + float(info['eligibility_credits'])
    return float(reward)


# gympn/planners/dcl_planner.py
@torch.no_grad()
def compute_target_pi(env, state_graph, actor, cfg: PlannerConfig, lineage=None):
    enabled = env.enabled_actions(state_graph)  # positions in pn.pn_actions
    A = len(enabled)
    if A == 0:
        return np.array([]), {'means': [], 'stds': []}, enabled
    if A == 1:
        return np.array([1.0], dtype=np.float32), {'means': [0.0], 'stds': [0.0]}, enabled

    seeds = [np.random.randint(0, 2**31-1) for _ in range(cfg.rollouts_per_action)] if cfg.use_crn else [None] * cfg.rollouts_per_action
    means = np.zeros(A, dtype=np.float32)
    stds  = np.zeros(A, dtype=np.float32)
    snapshot = env.get_state() if hasattr(env, "get_state") else None

    for i, env_idx0 in enumerate(enabled):
        rets = []
        for sd in seeds:
            if snapshot is not None and hasattr(env, "set_state"):
                env.set_state(snapshot)
            if sd is not None and hasattr(env, "set_seed"):
                env.set_seed(sd)

            # Step 0: take the candidate env position
            s, rwd, done, trunc, info = env.step(env_idx0)
            total = _shape_reward(rwd, info, lineage)
            disc  = cfg.gamma

            # Steps 1..H-1: continue H-1 times using CURRENT POLICY or uniform over enabled positions
            for t in range(1, cfg.horizon):
                if done or trunc:
                    break

                enabled_t = env.enabled_actions(s)  # positions in pn_actions
                if len(enabled_t) == 0:
                    break

                # --- robust choice (uniform over enabled positions) ---
                a_pos = int(np.random.choice(enabled_t))

                s, rwd, done, trunc, info = env.step(a_pos)
                total += disc * _shape_reward(rwd, info, lineage)
                disc  *= cfg.gamma

            rets.append(total)

        means[i] = float(np.mean(rets))
        stds[i]  = float(np.std(rets))

    logits = means / max(cfg.temperature, 1e-8)
    logits -= logits.max()
    exps = np.exp(logits)
    target_pi = exps / (exps.sum() if exps.sum() > 0 else 1.0)

    return target_pi.astype(np.float32), {'means': means.tolist(), 'stds': stds.tolist()}, enabled


# ----- Optional: Sequential Halving flavor for large action sets -----

@torch.no_grad()
def sequential_halving_target_pi(env, state_graph, actor, cfg: PlannerConfig,
                                 rounds: int = 3, lineage: Optional[object] = None
                                ) -> Tuple[np.ndarray, Dict[str, Any], List[int]]:
    enabled: List[int] = env.enabled_actions(state_graph)
    if len(enabled) <= 1:
        return (np.array([1.0], dtype=np.float32) if enabled else np.array([], dtype=np.float32),
                {'rounds': []}, enabled)

    S: List[int] = list(enabled)  # survivors
    round_stats: List[Dict[str, Any]] = []
    snapshot = env.get_state() if hasattr(env, "get_state") else None

    for r in range(rounds):
        seeds = [np.random.randint(0, 2**31 - 1) for _ in range(cfg.rollouts_per_action)]
        means: List[float] = []
        for a0 in S:
            rets: List[float] = []
            for sd in seeds:
                if snapshot is not None:
                    env.set_state(snapshot)
                if hasattr(env, "set_seed"):
                    env.set_seed(sd)

                s, rwd, done, trunc, info = env.step(a0)
                total = _shape_reward(rwd, info, lineage)
                disc = cfg.gamma
                for t in range(1, cfg.horizon):
                    if done or trunc: break
                    probs = actor(s)
                    enabled_t = env.enabled_actions(s)
                    if len(enabled_t) == 0: break
                    p_vec = torch.clamp(probs.flatten()[enabled_t], min=1e-8)
                    p_vec = (p_vec / p_vec.sum()).cpu().numpy()
                    a_t = int(np.random.choice(enabled_t, p=p_vec))
                    s, rwd, done, trunc, info = env.step(a_t)
                    total += disc * _shape_reward(rwd, info, lineage)
                    disc  *= cfg.gamma
                rets.append(float(total))
            means.append(float(np.mean(rets)))

        round_stats.append({'round': r, 'actions': S.copy(), 'means': means.copy()})
        # keep top half
        idx = np.argsort(means)[::-1]
        keep = max(1, int(np.ceil(len(S) / 2)))
        S = [S[i] for i in idx[:keep]]
        if len(S) == 1:
            break

    # final target over last survivors
    last_means = np.array(round_stats[-1]['means'], dtype=np.float32)
    logits = last_means / max(cfg.temperature, 1e-8)
    logits -= logits.max()
    exps = np.exp(logits)
    target = exps / (exps.sum() if exps.sum() > 0 else 1.0)
    return target.astype(np.float32), {'rounds': round_stats}, S
