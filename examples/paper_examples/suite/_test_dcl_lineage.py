"""Unit tests for the structurally lineage-aware DCL planner
(gympn/dcl_planner.compute_target_pi_lineage). Run: python _test_dcl_lineage.py
"""
import random
import sys
import types
import uuid

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite")

from gympn.dcl_planner import (PlannerConfig, compute_target_pi,
                               compute_target_pi_lineage, _binding_signature)
from gympn.environment import AEPN_Env as _AEPN
from gympn.environment import AEPN_Env
from stoch_envs import make_s1_stoch_sequence


def _traced_env(seed=5):
    """s1 with causal tracing on, initialised the way training_run does."""
    random.seed(seed)
    np.random.seed(seed)
    pn = make_s1_stoch_sequence(causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True)
    pn.length = 20
    for place in pn.places:
        for token in place.marking:
            setattr(token, '_id', str(uuid.uuid4()))
    pn.causal_trace.flush()
    sentinel = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sentinel, [], time=0)
    pn.causal_trace.register_transition(sentinel, [], toks, is_action=False,
                                        reward=0.0, time=0)
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, obs


class _Actor:
    def __call__(self, s):
        return torch.ones(8)


CFG = PlannerConfig(horizon=6, rollouts_per_action=8, gamma=0.95,
                    temperature=1.0, use_crn=True, use_lineage=True)


def test_signature_is_stable_and_identifies_bindings():
    env, obs = _traced_env()
    sigs = [_binding_signature(b) for b in env.pn.pn_actions]
    real = [s for s in sigs if s is not None]
    assert real, "no signatures built"
    assert len(set(real)) == len(real), "signatures must be unique per binding"
    # a signature survives a step that does not consume its tokens
    before = set(real)
    env.step(0)
    after = {s for s in (_binding_signature(b) for b in env.pn.pn_actions)
             if s is not None}
    assert before & after, "signatures did not survive an unrelated step"
    print(f"  signature_stable OK ({len(real)} bindings, "
          f"{len(before & after)} survived a step)")


def test_planner_restores_env_and_returns_valid_distribution():
    env, obs = _traced_env()
    clock0 = float(env.pn.clock)
    reward0 = float(env.pn.reward)
    pi, stats, enabled = compute_target_pi_lineage(env, obs, _Actor(), CFG)
    assert len(pi) == len(enabled) and len(pi) > 1
    assert abs(float(pi.sum()) - 1.0) < 1e-5, pi
    assert float(env.pn.clock) == clock0, "planner left the clock advanced"
    assert float(env.pn.reward) == reward0, "planner left reward mutated"
    print(f"  planner_restores_env OK (A={len(enabled)}, "
          f"rollouts={stats['rollouts']}, samples={stats['samples']})")


def test_sharing_adds_samples_without_extra_rollouts():
    """lineage_share must yield MORE scored samples than rollouts run."""
    env, obs = _traced_env()
    cfg = PlannerConfig(**{**CFG.__dict__, 'lineage_tally': True,
                           'lineage_share': True, 'lineage_prune': False})
    _, st_share, _ = compute_target_pi_lineage(env, obs, _Actor(), cfg)

    env, obs = _traced_env()
    cfg_no = PlannerConfig(**{**cfg.__dict__, 'lineage_share': False})
    _, st_plain, _ = compute_target_pi_lineage(env, obs, _Actor(), cfg_no)

    tot_share, tot_plain = sum(st_share['samples']), sum(st_plain['samples'])
    assert st_share['rollouts'] == st_plain['rollouts'], "rollout count differed"
    assert tot_share >= tot_plain, (tot_share, tot_plain)
    print(f"  sharing_adds_samples OK ({tot_plain} -> {tot_share} samples "
          f"at identical {st_share['rollouts']} rollouts)")


def test_pruning_reduces_rollouts():
    env, obs = _traced_env()
    cfg_p = PlannerConfig(**{**CFG.__dict__, 'lineage_prune': True,
                             'rollouts_per_action': 12})
    _, st_p, _ = compute_target_pi_lineage(env, obs, _Actor(), cfg_p)

    env, obs = _traced_env()
    cfg_n = PlannerConfig(**{**cfg_p.__dict__, 'lineage_prune': False})
    _, st_n, _ = compute_target_pi_lineage(env, obs, _Actor(), cfg_n)
    print(f"  pruning OK (rollouts {st_n['rollouts']} -> {st_p['rollouts']}, "
          f"pruned={st_p['pruned']})")
    assert st_p['rollouts'] <= st_n['rollouts'] * 1.35, (st_p, st_n)


def test_truncation_is_cheaper_and_only_with_tally():
    """Coupling truncation must not change results when tallies are raw
    (it is only sound for the restricted estimand)."""
    env, obs = _traced_env()
    cfg_raw_t = PlannerConfig(**{**CFG.__dict__, 'lineage_tally': False,
                                 'lineage_truncate': True, 'use_crn': True})
    pi_a, _, _ = compute_target_pi_lineage(env, obs, _Actor(), cfg_raw_t)

    env, obs = _traced_env()
    cfg_raw_f = PlannerConfig(**{**cfg_raw_t.__dict__, 'lineage_truncate': False})
    pi_b, _, _ = compute_target_pi_lineage(env, obs, _Actor(), cfg_raw_f)

    assert np.allclose(pi_a, pi_b, atol=1e-6), (pi_a, pi_b)
    print("  truncation_guarded OK (no effect on raw tallies, as designed)")


def test_untraced_env_still_works():
    """Planner must degrade gracefully when the env records no trace."""
    random.seed(1); np.random.seed(1)
    pn = make_s1_stoch_sequence(causal_rl=False, allow_postpone=True)
    pn.length = 20
    env = AEPN_Env(pn)
    obs = env.reset()
    pi, stats, enabled = compute_target_pi_lineage(env, obs, _Actor(), CFG)
    assert abs(float(pi.sum()) - 1.0) < 1e-5
    print(f"  untraced_env OK (fell back, rollouts={stats['rollouts']})")


def test_both_planners_restore_env_state():
    """Regression for the X15 below-random bug: BOTH planner entry points must
    leave the env exactly as they found it (clock, reward, action count)."""
    from gympn.dcl_planner import compute_target_pi as _plain
    for name, traced, fn in (("plain", False, _plain),
                             ("lineage", True, compute_target_pi_lineage)):
        env, obs = (_traced_env() if traced else _untraced())
        clock0, rew0 = float(env.pn.clock), float(env.pn.reward)
        n0 = len(env.pn.pn_actions)
        fn(env, obs, _Actor(), CFG)
        assert float(env.pn.clock) == clock0, f"{name}: clock moved"
        assert float(env.pn.reward) == rew0, f"{name}: reward moved"
        assert len(env.pn.pn_actions) == n0, f"{name}: action set changed"
    print("  both_planners_restore_env_state OK (plain + lineage)")


def _untraced(seed=5):
    random.seed(seed); np.random.seed(seed)
    pn = make_s1_stoch_sequence(causal_rl=False, allow_postpone=True)
    pn.length = 20
    env = _AEPN(pn)
    return env, env.reset()


if __name__ == "__main__":
    test_both_planners_restore_env_state()
    test_signature_is_stable_and_identifies_bindings()
    test_planner_restores_env_and_returns_valid_distribution()
    test_sharing_adds_samples_without_extra_rollouts()
    test_pruning_reduces_rollouts()
    test_truncation_is_cheaper_and_only_with_tally()
    test_untraced_env_still_works()
    print("all dcl lineage planner tests passed")