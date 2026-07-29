"""Tests for the Direction A MCTS planner (gympn/mcts_planner.py,
AEPN_NATIVE_LEARNING.md §3)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.mcts_planner import (  # noqa: E402
    MCTSConfig, mcts_target_pi, build_conflict_adjacency,
    _sig, _is_forced, _conflicts, _POSTPONE,
)


# --------------------------------------------------------------------------- #
# Unit: the conflict gate logic (pure, no env)
# --------------------------------------------------------------------------- #
def test_forced_vs_decision_node():
    # two bindings of DISTINCT, structurally-independent transitions, disjoint
    # tokens -> commute -> forced (not a decision)
    a = ("startA", ("t1",))
    b = ("startB", ("t2",))
    assert _is_forced([a, b], adj={}) is True
    # shared token -> conflict -> decision node
    c = ("startB", ("t1",))
    assert _is_forced([a, c], adj={}) is False
    # static shared-input-place edge -> conflict
    assert _conflicts(a, b, adj={"startA": {"startB"}}) is True
    # postpone is always a real choice
    assert _conflicts(a, _POSTPONE, adj={}) is True
    assert _is_forced([a, b, _POSTPONE], adj={}) is False


# --------------------------------------------------------------------------- #
# Integration on the deterministic E1 one-shot foreclosing choice
# --------------------------------------------------------------------------- #
def _build_e1_at_decision(stages=2):
    """Drive a fresh E1b one-shot env to its single real decision state
    (start_A{stages} on the shared employee vs start_B) and return
    (env, obs, positions) where positions maps 'chain'/'shortcut' -> env idx."""
    from e1_chain_env import make_e1b_oneshot
    pn = make_e1b_oneshot(causal_rl=False, allow_postpone=False, stages=stages)
    # Horizon 2 is the designed one-shot foreclosing window: only ONE of the
    # two tasks fits, so serving the chain (12) forecloses the shortcut (7).
    # With more horizon both complete (total 19) and the choice is moot.
    pn.length = 2
    env = AEPN_Env(pn)
    obs = env.reset()

    for _ in range(50):
        binds = env.pn.pn_actions
        real = [j for j, b in enumerate(binds) if _sig(b) not in (None, _POSTPONE)]
        trs = {j: _sig(binds[j])[0] for j in real}
        has_chain = any(t.startswith("start_A") for t in trs.values())
        has_short = any(t.startswith("start_B") for t in trs.values())
        if has_chain and has_short and len(real) >= 2:
            pos = {"chain": next(j for j, t in trs.items() if t.startswith("start_A")),
                   "shortcut": next(j for j, t in trs.items() if t.startswith("start_B"))}
            return env, obs, pos
        if not real:
            obs, _, done, _, _ = env.step(0)
            if done:
                break
        else:
            obs, _, done, _, _ = env.step(real[0])
            if done:
                break
    raise AssertionError("did not reach the A-vs-B decision state")


def test_search_prefers_chain_over_shortcut():
    """The foreclosing choice: chain completion pays 12, shortcut pays 7.
    Pure lookahead (uniform prior, no critic) must put more visit mass on the
    chain — the discrimination scalar credit methods failed and cfpk nailed."""
    env, obs, pos = _build_e1_at_decision(stages=2)
    cfg = MCTSConfig(n_simulations=64, max_depth=4, beta=0.5,
                     value_bootstrap=False, conflict_gate=True)
    adj = build_conflict_adjacency(env.pn)
    target, stats, enabled = mcts_target_pi(env, obs, actor=None, cfg=cfg,
                                            conflict_adj=adj, critic=None)
    ci = enabled.index(pos["chain"])
    si = enabled.index(pos["shortcut"])
    assert target[ci] > target[si], (target.tolist(), stats["visits"])
    # the search's own value estimate should also rank the chain higher
    assert stats["means"][ci] > stats["means"][si]


def test_search_restores_env_state():
    """The planner must leave the real episode exactly where it found it
    (the X15 below-random corruption guard)."""
    env, obs, _ = _build_e1_at_decision(stages=2)
    before_clock = float(env.pn.clock)
    before_tokens = sum(len(p.marking) for p in env.pn.places)
    cfg = MCTSConfig(n_simulations=32, max_depth=4, beta=0.5, value_bootstrap=False)
    mcts_target_pi(env, obs, actor=None, cfg=cfg,
                   conflict_adj=build_conflict_adjacency(env.pn), critic=None)
    assert float(env.pn.clock) == before_clock
    assert sum(len(p.marking) for p in env.pn.places) == before_tokens


# --------------------------------------------------------------------------- #
# Coupling truncation
# --------------------------------------------------------------------------- #
def _reconverging_env(name="a_sequence_joint"):
    """A real deterministic suite env driven to its first multi-binding
    decision. These have genuine reconvergence at decision states (different
    assignment orders reach the same aggregate marking), which the trivial
    two-task net does not — the env auto-resolves single-binding states, so a
    reconvergence there lands on a terminal rather than a tabled decision."""
    from envs import make_env
    pn = make_env(name, allow_postpone=False)
    pn.length = 10
    env = AEPN_Env(pn)
    obs = env.reset()
    for _ in range(20):
        if len(env.pn.pn_actions) >= 2:
            return env, obs
        obs, _, done, _, _ = env.step(0)
        if done:
            break
    raise AssertionError(f"{name}: never reached a multi-binding decision")


def test_coupling_reuses_reconvergent_states():
    """On an env with genuine reconvergence, node-sharing registers many
    transposition hits (branches that coincide reuse one node)."""
    env, obs = _reconverging_env()
    cfg = MCTSConfig(n_simulations=64, lookahead=10.0, beta=0.5,
                     value_bootstrap=False, conflict_gate=False,
                     coupling_truncate=True)
    _, stats, _ = mcts_target_pi(env, obs, actor=None, cfg=cfg,
                                 conflict_adj=None, critic=None)
    assert stats["transpositions"] > 0, stats


def test_couple_min_visits_cuts_env_steps():
    """The value-reuse short-circuit is the actual cost win: re-entering a
    resolved coupled state returns its cached value instead of re-descending,
    so it fires truncations AND executes strictly fewer env.step calls."""
    common = dict(n_simulations=64, lookahead=10.0, beta=0.5,
                  value_bootstrap=False, conflict_gate=False,
                  coupling_truncate=True)
    env, obs = _reconverging_env()
    snap = env.get_state()
    _, s0, _ = mcts_target_pi(env, obs, actor=None,
                              cfg=MCTSConfig(couple_min_visits=0, **common),
                              conflict_adj=None, critic=None)
    env.set_state(snap)
    _, s1, _ = mcts_target_pi(env, obs, actor=None,
                              cfg=MCTSConfig(couple_min_visits=2, **common),
                              conflict_adj=None, critic=None)
    assert s1["truncations"] > 0, s1
    assert s1["steps"] < s0["steps"], (s1["steps"], s0["steps"])


def test_coupling_preserves_decision():
    """Node-sharing (couple_min_visits=0) must not change the search result on
    the E1 foreclosing choice."""
    for couple in (False, True):
        env, obs, pos = _build_e1_at_decision(stages=2)
        cfg = MCTSConfig(n_simulations=64, lookahead=8.0, beta=0.5,
                         value_bootstrap=False, conflict_gate=True,
                         coupling_truncate=couple)
        target, _, enabled = mcts_target_pi(env, obs, actor=None, cfg=cfg,
                                             conflict_adj=build_conflict_adjacency(env.pn),
                                             critic=None)
        assert target[enabled.index(pos["chain"])] > target[enabled.index(pos["shortcut"])]


# --------------------------------------------------------------------------- #
# Lineage-attributed backup (the paper contribution)
# --------------------------------------------------------------------------- #
def _causal_e1_at_decision(stages=2):
    """E1b one-shot with causal tracing on, driven to the A-vs-B decision.
    Mirrors the token-id init training_run does before building the env."""
    import types
    import uuid
    from e1_chain_env import make_e1b_oneshot
    pn = make_e1b_oneshot(causal_rl=True, allow_postpone=False, stages=stages)
    pn.length = 2
    for place in pn.places:
        for token in place.marking:
            setattr(token, "_id", str(uuid.uuid4()))
    pn.causal_trace.flush()
    sentinel = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sentinel, [], time=0)
    pn.causal_trace.register_transition(sentinel, [], toks, is_action=False,
                                        reward=0.0, time=0)
    env = AEPN_Env(pn)
    obs = env.reset()
    for _ in range(50):
        binds = env.pn.pn_actions
        real = [j for j, b in enumerate(binds) if _sig(b) not in (None, _POSTPONE)]
        trs = {j: _sig(binds[j])[0] for j in real}
        if (any(t.startswith("start_A") for t in trs.values())
                and any(t.startswith("start_B") for t in trs.values())):
            pos = {"chain": next(j for j, t in trs.items() if t.startswith("start_A")),
                   "shortcut": next(j for j, t in trs.items() if t.startswith("start_B"))}
            return env, obs, pos
        obs, _, done, _, _ = env.step(real[0] if real else 0)
        if done:
            break
    raise AssertionError("did not reach the A-vs-B decision (causal env)")


def test_lineage_backup_prefers_chain():
    """Lineage-attributed backup credits start_A2 by the chain reward (12, its
    descendant) and start_B by the shortcut reward (7, its descendant), so the
    sibling comparison prefers the chain — the discrimination done via per-edge
    causal credit rather than value bootstrap."""
    env, obs, pos = _causal_e1_at_decision(stages=2)
    cfg = MCTSConfig(n_simulations=48, lookahead=8.0, beta=0.5,
                     conflict_gate=True, lineage_backup=True)
    target, stats, enabled = mcts_target_pi(env, obs, actor=None, cfg=cfg,
                                             conflict_adj=build_conflict_adjacency(env.pn),
                                             critic=None)
    ci, si = enabled.index(pos["chain"]), enabled.index(pos["shortcut"])
    assert target[ci] > target[si], (target.tolist(), stats["visits"])
    assert stats["means"][ci] > stats["means"][si]


def test_whole_and_lineage_agree_on_e1():
    """E1 is direct-dominated (each reward IS the deciding action's descendant),
    so whole return-to-go and lineage-restricted backup must agree: both prefer
    the chain. The lineage advantage shows on concurrency/noise envs, not here —
    this is the do-no-harm check."""
    for lin in (False, True):
        env, obs, pos = _causal_e1_at_decision(stages=2)
        cfg = MCTSConfig(n_simulations=48, lookahead=8.0, beta=0.5,
                         conflict_gate=True, rollout_backup=True, lineage_backup=lin)
        target, _, enabled = mcts_target_pi(env, obs, actor=None, cfg=cfg,
                                            conflict_adj=build_conflict_adjacency(env.pn),
                                            critic=None)
        assert target[enabled.index(pos["chain"])] > target[enabled.index(pos["shortcut"])]


def test_lineage_backup_requires_trace():
    """Rollout/lineage backup on a non-traced env must fail loudly, not
    silently degrade."""
    env, obs, _ = _build_e1_at_decision(stages=2)  # causal_rl=False
    cfg = MCTSConfig(n_simulations=8, lineage_backup=True)
    with pytest.raises(RuntimeError, match="causal_rl=True"):
        mcts_target_pi(env, obs, actor=None, cfg=cfg, conflict_adj=None, critic=None)


def test_lineage_backup_restores_state():
    env, obs, _ = _causal_e1_at_decision(stages=2)
    before_clock = float(env.pn.clock)
    before_tokens = sum(len(p.marking) for p in env.pn.places)
    cfg = MCTSConfig(n_simulations=24, lookahead=8.0, beta=0.5, lineage_backup=True)
    mcts_target_pi(env, obs, actor=None, cfg=cfg,
                   conflict_adj=build_conflict_adjacency(env.pn), critic=None)
    assert float(env.pn.clock) == before_clock
    assert sum(len(p.marking) for p in env.pn.places) == before_tokens


def test_contract_matches_dcl_planner():
    """Return shape must match compute_target_pi so DCLAgent consumes it:
    target_pi over enabled, stats with 'means', enabled positions list."""
    env, obs, _ = _build_e1_at_decision(stages=2)
    cfg = MCTSConfig(n_simulations=16, max_depth=4, beta=0.5, value_bootstrap=False)
    target, stats, enabled = mcts_target_pi(env, obs, actor=None, cfg=cfg,
                                            conflict_adj=None, critic=None)
    assert len(target) == len(enabled)
    assert len(stats["means"]) == len(enabled)
    assert abs(float(np.sum(target)) - 1.0) < 1e-5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))