"""Unit tests for cfpk's coupling truncation
(gympn/counterfactual.py's `_paired_coupled_suffixes`, wired into `maybe_fork`
via `cfg['coupling_truncate']`). Run: python _test_coupling_truncation.py

Addresses CFPK_EXPLAINED.md's documented-but-unbuilt speedup: "stop early
once both playouts have clearly reconverged" (measured ~3.6x cost vs the
intended <2x). Mirrors gympn/mcts_planner.py's state_fingerprint transposition
mechanism (reused directly, not reinvented).
"""
import sys, os, types, uuid
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import torch

import gympn
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.environment import AEPN_Env
from gympn.counterfactual import maybe_fork
import gympn.counterfactual as cf_mod


def make_symmetric(causal_rl=True, reward=5.0):
    """Two actions (go_a/go_b) competing for the SAME input place, both
    producing a token in the SAME downstream place with the SAME value and
    delay -- so the two forked branches are GUARANTEED to reconverge to an
    identical PN state (same marking, same clock) immediately after firing
    their first, distinguishing action. Purpose-built for a controlled
    coupling-truncation test, not a realistic env.

    A "corridor" of two more REAL decisions (step2_x/y, step3_x/y -- each a
    genuine 2-binding choice, not an auto-resolved single-binding event)
    follows `mid`, so a deterministic continuation policy takes several
    GENUINE lockstep rounds before the episode ends -- without this, the net
    completes within the fork's own first env.step() call and there is
    nothing left for coupling truncation to actually save (confirmed by
    running the single-corridor-step version of this net first)."""
    ag = GymProblem(allow_postpone=False, causal_rl=causal_rl)
    start = ag.add_var("start", var_attributes=['id'])
    mid = ag.add_var("mid", var_attributes=['id'])
    p2 = ag.add_var("p2", var_attributes=['id'])
    p3 = ag.add_var("p3", var_attributes=['id'])
    done = ag.add_var("done", var_attributes=['id'])
    start.put({'id': 1})
    ag.add_action([start], [mid], behavior=lambda c: [SimToken(c, delay=1)], name='go_a')
    ag.add_action([start], [mid], behavior=lambda c: [SimToken(c, delay=1)], name='go_b')
    ag.add_action([mid], [p2], behavior=lambda c: [SimToken(c, delay=1)], name='step2_x')
    ag.add_action([mid], [p2], behavior=lambda c: [SimToken(c, delay=1)], name='step2_y')
    ag.add_action([p2], [p3], behavior=lambda c: [SimToken(c, delay=1)], name='step3_x')
    ag.add_action([p2], [p3], behavior=lambda c: [SimToken(c, delay=1)], name='step3_y')
    ag.add_event([p3], [done], lambda p: [SimToken(p)], name='finish',
                 reward_function=lambda b: reward)
    return ag


class _StubAgent:
    """Deterministic (always index 0) continuation policy + zero critic --
    matches _test_cfp.py's pattern. Deterministic act() means both branches'
    post-reconvergence continuations are trivially identical, reinforcing
    (not creating) the reconvergence this test is actually checking."""
    def act(self, state, deterministic=False, return_logprob=False):
        return 0

    value_model = staticmethod(lambda obs: torch.zeros(1))


def _fresh_env(seed=7, **rw):
    random.seed(seed); np.random.seed(seed)
    pn = make_symmetric(causal_rl=True, **rw)
    pn.length = 20
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, obs


CFG_BASE = {'fork_prob': 1.0, 'reps': 3, 'gate': 0.0, 'lookahead': 6.0,
           'max_forks': 5, 'coef': 1.0, 'updates': 2, 'beta': 0.0}


def test_coupling_matches_uncoupled_gap():
    """The paired gap/se must be numerically identical whether or not
    coupling truncation is on -- coupling changes HOW the tail is computed
    (once, shared) not WHAT the branches' expected returns are."""
    env1, obs1 = _fresh_env()
    n = len(obs1['actions_dict'])
    logpis = torch.log(torch.ones(n) / n)
    _, _, diag_off = maybe_fork(_StubAgent(), env1, obs1, 0, logpis,
                               dict(CFG_BASE, coupling_truncate=False))

    env2, obs2 = _fresh_env()
    _, _, diag_on = maybe_fork(_StubAgent(), env2, obs2, 0, logpis,
                              dict(CFG_BASE, coupling_truncate=True))

    assert diag_off is not None and diag_on is not None
    assert abs(diag_off['gap'] - diag_on['gap']) < 1e-6, (diag_off, diag_on)
    assert abs(diag_off['se'] - diag_on['se']) < 1e-6, (diag_off, diag_on)
    print(f"  coupling_matches_uncoupled_gap OK (gap={diag_on['gap']:.4f}, "
          f"se={diag_on['se']:.4f})")


def test_coupling_actually_triggers_and_saves_steps():
    """On the symmetric net, reconvergence is guaranteed at round 0 (right
    after the two branches' first action) -- confirm the mechanism telemetry
    reports it firing every rep, AND that env.step() is called fewer times
    with coupling on than off (the actual cost claim)."""
    env1, obs1 = _fresh_env()
    n = len(obs1['actions_dict'])
    logpis = torch.log(torch.ones(n) / n)

    steps = {'n': 0}
    real_step = type(env1).step

    def _counting_step(self, *a, **kw):
        steps['n'] += 1
        return real_step(self, *a, **kw)

    type(env1).step = _counting_step
    try:
        steps['n'] = 0
        _, _, diag_off = maybe_fork(_StubAgent(), env1, obs1, 0, logpis,
                                   dict(CFG_BASE, coupling_truncate=False))
        steps_off = steps['n']
    finally:
        type(env1).step = real_step

    env2, obs2 = _fresh_env()
    type(env2).step = _counting_step
    try:
        steps['n'] = 0
        _, _, diag_on = maybe_fork(_StubAgent(), env2, obs2, 0, logpis,
                                  dict(CFG_BASE, coupling_truncate=True))
        steps_on = steps['n']
    finally:
        type(env2).step = real_step

    assert diag_on.get('coupled_reps') == CFG_BASE['reps'], diag_on
    assert all(c == 0 for c in diag_on['coupled_rounds']), diag_on
    assert steps_on < steps_off, (steps_on, steps_off)
    print(f"  coupling_actually_triggers_and_saves_steps OK "
          f"(coupled_reps={diag_on['coupled_reps']}/{CFG_BASE['reps']}, "
          f"env.step calls: off={steps_off} on={steps_on})")


def test_coupling_disabled_by_default():
    """No 'coupling_truncate' key in cfg -> old sequential path, unaffected."""
    env, obs = _fresh_env()
    n = len(obs['actions_dict'])
    logpis = torch.log(torch.ones(n) / n)
    forked, pref, diag = maybe_fork(_StubAgent(), env, obs, 0, logpis, CFG_BASE)
    assert forked and diag is not None
    assert 'coupled_reps' not in diag
    print("  coupling_disabled_by_default OK")


def test_lineage_mode_ignores_coupling_truncate():
    """coupling_truncate=True combined with lineage=True must not crash and
    must silently fall back to the sequential (uncoupled) lineage path --
    coupling truncation is explicitly out of scope for lineage's per-branch
    trace accounting (see _paired_coupled_suffixes' docstring)."""
    random.seed(3); np.random.seed(3)
    pn = make_symmetric(causal_rl=True)
    pn.length = 20
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_postpone_tokenflow = True
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn)
    obs = env.reset()
    n = len(obs['actions_dict'])
    logpis = torch.log(torch.ones(n) / n)

    forked, pref, diag = maybe_fork(_StubAgent(), env, obs, 0, logpis,
                                   dict(CFG_BASE, lineage=True, coupling_truncate=True))
    assert forked and diag is not None
    assert 'coupled_reps' not in diag, "coupling telemetry must not appear in lineage mode"
    print("  lineage_mode_ignores_coupling_truncate OK")


if __name__ == "__main__":
    test_coupling_matches_uncoupled_gap()
    test_coupling_actually_triggers_and_saves_steps()
    test_coupling_disabled_by_default()
    test_lineage_mode_ignores_coupling_truncate()
    print("all coupling-truncation tests passed")
