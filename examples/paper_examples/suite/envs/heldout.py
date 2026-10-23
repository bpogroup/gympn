"""Scoring on fixed sets of days, the way in-training evaluation scores a policy.

Every day i of a set runs under seed base + i (Python random and NumPy reseeded,
then env.reset()), so a policy, the heuristic and the random policy all face
the same scenarios: the scores are paired. Two sets are used:

- EVAL days (EVAL_SEED, 20 days): the days of the in-training evaluation. The
  learning curves and greedy_final are scored on them, and best_policy.pth is
  SELECTED on them, so a best checkpoint's score there is optimistic. The
  anchors on these days make greedy_final's normalisation paired.
- HELD-OUT days (HELDOUT_SEED, 100 days): never used for training, tuning or
  selection. The final and best checkpoints are scored here, with anchors on
  the same days.

The random policy draws its choices from its own generator (seeded by the
day), so it does not consume the scenario's random stream.
"""
import random

import numpy as np

EVAL_SEED, EVAL_DAYS = 555_000, 20
HELDOUT_SEED, HELDOUT_DAYS = 888_000, 100


def make_env(build, n, length, allow_postpone=False, postpone_scope='global', flat_obs=True,
             global_view='edges', marking_certificate=False):
    """An AEPN_Env set up like run_bpm's training and test environments."""
    from gympn.environment import AEPN_Env
    pn = build(n, allow_postpone=allow_postpone)
    pn.postpone_scope = postpone_scope
    pn.length = length
    pn.flat_obs = flat_obs
    pn.global_view = global_view
    pn.marking_certificate = marking_certificate
    return AEPN_Env(pn)


def _index_of(B, sel):
    if isinstance(sel, int):
        return sel
    for i, b in enumerate(B):
        if b is sel:
            return i
    if sel == 'postpone' or (isinstance(sel, tuple) and sel and sel[0] == ['postpone']):
        for i, b in enumerate(B):
            if isinstance(b[0], list) and b[0] == ['postpone']:
                return i
    for i, b in enumerate(B):           # an equal binding (same tokens, time, transition)
        if b == sel:
            return i
    raise ValueError("the heuristic returned a binding that is not offered")


def score_solver(env, base, days, kind, heuristic=None):
    """Returns per day of the random policy (kind='random') or of a heuristic
    (kind='heuristic', a run_bpm heuristic function)."""
    from gympn.solvers import HeuristicSolver
    solver = HeuristicSolver(heuristic) if kind == 'heuristic' else None
    out = []
    for i in range(days):
        seed = base + i
        random.seed(seed); np.random.seed(seed % 2 ** 32)
        rng = np.random.default_rng(seed)
        env.reset()
        done, info = False, {'pn_reward': 0.0}
        while not done and env.pn.pn_actions:
            B = env.pn.pn_actions
            if kind == 'random':
                a = int(rng.integers(len(B)))
            else:
                a = _index_of(B, solver.solve(env.pn.get_heuristic_observation(), B))
            _, _, done, _, info = env.step(a)
        out.append(float(info['pn_reward']))
    return out


def score_policy(env, policy, base, days, batch=32):
    """Greedy returns per day of a policy network, batched like test_in_train
    (Agent._test_lockstep: same scenarios as one day at a time)."""
    from gympn.agents import Agent

    class _Shim:
        pass
    shim = _Shim()
    shim.policy_model, shim.rollout_batch = policy, batch
    state = (random.getstate(), np.random.get_state())
    try:
        returns, _ = Agent._test_lockstep(shim, env, days, base, deterministic=True)
    finally:
        random.setstate(state[0]); np.random.set_state(state[1])
    return [float(r) for r in returns]


def anchors(env, heuristic, base, days):
    r = score_solver(env, base, days, 'random')
    h = score_solver(env, base, days, 'heuristic', heuristic)
    return {"seed": base, "days": days, "random_mean": float(np.mean(r)), "heuristic_mean": float(np.mean(h)),
            "random": r, "heuristic": h}
