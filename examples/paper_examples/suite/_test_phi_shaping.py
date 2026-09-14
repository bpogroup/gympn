r"""Unit + smoke tests for potential-based reward shaping (gympn/potential.py +
its injection into gympn.agents.Agent.run_episode). Run: python _test_phi_shaping.py

Covers, in order: (1) the place-weight BFS on a known topology; (2) the
terminal-Phi=0 convention, both for natural completion and for
max_episode_length truncation; (3) THE PRIMARY GATE -- policy invariance: with
shaping on, the relative SMDP-discounted-return gap between two forced action
choices must be numerically identical to the unshaped gap, on both a
natural-completion and a truncated episode; (4) a real end-to-end smoke train
via run_suite.train_cell, for plain PPO (full effect) and for a lineage-Q
scheme (documented no-op, must not crash).
"""
import sys, os, math, types, uuid
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import gympn
from gympn.environment import AEPN_Env
from gympn.data import TrajectoryBuffer
from gympn.agents import Agent
import gympn.agents as agents_mod
from gympn.potential import _place_reward_hops, get_place_weights, topology_potential
from assembly_probe import make_shared_r


# --------------------------------------------------------------------------- #
# Forced-trajectory harness through the REAL Agent/TrajectoryBuffer/AEPN_Env  #
# --------------------------------------------------------------------------- #

def _make_forced_agent(env, choice_name, phi_coef, phi_decay=0.9, causal_beta=0.3, phi_cap=None):
    """A real Agent (via object.__new__, the same pattern _test_cfp.py already
    uses) with act() overridden to force the named action whenever it is
    available (mirrors assembly_probe.run_forced's binding-matching), and
    value_model=None so _compute_batch_values takes its trivial [0]*len path
    -- no network machinery needed since we only care about the reward
    stream, not learning."""
    # smdp_discount=False: finish()'s own SMDP-GAE branch is not what this
    # test checks (it independently recomputes the return from rewards_raw/
    # times via _smdp_return) -- and M2's two decisions can legitimately both
    # be decided at the same simulator clock instant (sojourn=0 until the
    # delayed behaviors actually elapse), which finish()'s smdp_discount path
    # correctly refuses to silently discount by. causal_beta is still used
    # directly by the shaping formula itself (self.buffer.causal_beta, read
    # in run_episode's injected block) regardless of this flag.
    buffer = TrajectoryBuffer(gam=0.99, lam=0.95, causal_scheme='lrq',
                              causal_rl=False, causal_beta=causal_beta,
                              smdp_discount=False)
    agent = object.__new__(Agent)
    agent.value_model = None
    agent.causal_rl = False
    agent.causal_scheme = 'lrq'
    agent.cf_config = None
    agent.rudder_agent = None
    agent.qoff_model = None
    agent.phi_coef = float(phi_coef)
    agent.phi_decay = float(phi_decay)
    agent.phi_cap = phi_cap
    agent._ls_hca_hhat = {}
    agent._ls_hca_records = []
    agent.buffer = buffer

    def _act(state, return_logprob=False, deterministic=False):
        acts = env.pn.pn_actions
        idx = 0
        for i, a in enumerate(acts):
            tname = ''
            if isinstance(a, tuple) and len(a) > 2 and a[2] is not None:
                tname = getattr(a[2], '_id', getattr(a[2], 'name', ''))
            if choice_name in str(tname):
                idx = i
                break
        n = max(1, len(acts))
        logpis = torch.zeros(n)
        logprob = torch.tensor(0.0)
        if return_logprob:
            return idx, logprob, logpis
        return idx
    agent.act = _act
    return agent, buffer


def _run_forced(choice_name, phi_coef, phi_decay=0.9, causal_beta=0.3,
                max_episode_length=None, length=12, seed=0, phi_cap=None,
                pn_factory=make_shared_r, pn_kwargs=None, **rw):
    """ENV built with causal_rl=True so single-binding decisions (M2's d2/
    route2) are NOT auto-resolved away inside one env.step() call -- without
    this the whole net collapses into a single agent-visible step and the
    invariance check never exercises the interesting cross-step Phi(s')-
    Phi(s) telescoping. AGENT stays causal_rl=False (standard SMDP-GAE path,
    where shaping has its full documented effect) -- the same env-traces/
    agent-stays-standard split already used by this project's 'cfpl' scheme.
    `choice_name` that matches no transition (e.g. "__NONE__") gives a fixed
    "always pick index 0" policy -- used for the longer-horizon truncation
    check, where forcing a specific NAMED action isn't the point."""
    gympn.seed_everything(seed)
    pn = pn_factory(causal_rl=True, **(pn_kwargs or {}), **rw)
    pn.length = length
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
    agent, buffer = _make_forced_agent(env, choice_name, phi_coef, phi_decay, causal_beta, phi_cap)
    buffer.clear()
    total_R, L = agent.run_episode(env, max_episode_length=max_episode_length, buffer=buffer)
    rewards = [float(r) for r in buffer.rewards_raw[:L].tolist()]
    times = [float(t) for t in buffer.times[:L]]
    return rewards, times, L


def _smdp_return(rewards, times, beta):
    """Independent reimplementation of the SMDP-discounted return-to-go
    (NOT calling smdp_gae itself, so this check isn't circular): tau_t =
    times[t+1]-times[t], tau_last = 0 (matches TrajectoryBuffer's own
    _taus_from_times convention), G_t = r_t + disc_t*G_{t+1}, G_n = 0."""
    n = len(rewards)
    taus = [times[t + 1] - times[t] for t in range(n - 1)] + [0.0]
    discounts = [math.exp(-beta * tau) for tau in taus]
    G = 0.0
    for t in reversed(range(n)):
        G = rewards[t] + discounts[t] * G
    return G


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

def test_place_weights_basic():
    pn = make_shared_r(causal_rl=False, r_join=5, r1=5, r2=10)
    decay = 0.9
    hops = _place_reward_hops(pn)
    weights = get_place_weights(pn, decay=decay)

    assert hops['sa1'] == 1, hops
    assert abs(weights['sa1'] - decay ** 1) < 1e-9

    assert hops['jw1'] == 1 and hops['busy1'] == 2, hops
    assert weights['busy1'] < weights['jw1'], "weight must decrease with hop count"

    # 'done' is a pure sink (output of every reward transition, input to none)
    # and 'busypriv' is an unused/unreferenced place -- both must be absent
    # (weight 0), never spuriously reachable.
    assert 'done' not in weights
    assert 'busypriv' not in weights
    print(f"  place_weights_basic OK (hops={hops})")


def test_cap_bounds_queue_contribution():
    """A place accumulating many tokens (a queue-like backlog) contributes
    unboundedly to the uncapped Phi but is bounded under phi_cap -- the whole
    point of the cap (see topology_potential's docstring: exogenous-arrival
    queue-depth variance). Manually swell one place's marking to simulate a
    backlog, independent of any particular env's arrival dynamics."""
    from gympn.simulator import GymProblem
    from simpn.simulator import SimToken
    pn = make_shared_r(causal_rl=False, r_join=5, r1=5, r2=10)
    part1 = next(p for p in pn.places if p._id == 'part1')
    for _ in range(9):  # part1 already has 1 token -> 10 total
        part1.put({'id': 99})

    uncapped = topology_potential(pn, decay=0.9, cap=None)
    capped1 = topology_potential(pn, decay=0.9, cap=1)
    capped3 = topology_potential(pn, decay=0.9, cap=3)

    w_part1 = get_place_weights(pn, decay=0.9)['part1']
    assert abs(uncapped - capped1) > 1e-9, "cap=1 must actually reduce the total vs uncapped"
    assert capped1 < capped3 < uncapped, (capped1, capped3, uncapped)
    # part1 alone contributes w*10 uncapped vs w*1 at cap=1 -- exactly w*9 of
    # the difference must be attributable to part1's swollen marking.
    assert abs((uncapped - capped1) - w_part1 * 9) < 1e-9
    print(f"  cap_bounds_queue_contribution OK (uncapped={uncapped:.3f} "
          f"cap=1:{capped1:.3f} cap=3:{capped3:.3f})")


def test_terminal_zero_convention():
    """topology_potential must be called exactly twice per non-terminal step
    (phi_s before stepping, phi_next after) and exactly ONCE on the terminal
    step (phi_next forced to 0.0 without calling the function). Checked on
    s1_stoch_sequence (29 steps under an "always index 0" policy) with a
    max_episode_length of 7 -- genuinely shorter than the natural episode, so
    this actually exercises the truncation clause of `is_last_step`
    independently of `done` (M2's 2-decision motif can't: its break check
    can never fire before `done` does, so a truncated run there is
    indistinguishable from natural completion -- see _run_forced's docstring
    and the truncation-identity test below for the same reason). 7, not some
    other short value: run_episode batches buffer.store() calls in groups of
    value_batch_size=8 (flushed when the batch fills OR `done`); a
    max_episode_length below 8 with `done` still False can hit `break` before
    ANY flush happens, losing steps from the buffer entirely (a genuine
    PRE-EXISTING run_episode/TrajectoryBuffer interaction, confirmed by hand
    -- out of scope to fix here). max_episode_length=7 makes the break land
    exactly on an 8-step flush boundary, the smallest value that is safe."""
    from stoch_envs import make_s1_stoch_sequence

    calls = {'n': 0}
    real = agents_mod.topology_potential

    def _counting(pn, decay=0.9, cap=None):
        calls['n'] += 1
        return real(pn, decay=decay, cap=cap)

    agents_mod.topology_potential = _counting
    try:
        calls['n'] = 0
        _, _, L = _run_forced('__NONE__', phi_coef=1.0, causal_beta=0.3, length=20,
                              pn_factory=make_s1_stoch_sequence,
                              pn_kwargs=dict(allow_postpone=True, causal_postpone_tokenflow=True))
        assert calls['n'] == 2 * L - 1, (calls['n'], L, 'natural completion')

        calls['n'] = 0
        _, _, L2 = _run_forced('__NONE__', phi_coef=1.0, causal_beta=0.3, length=20,
                               max_episode_length=7,
                               pn_factory=make_s1_stoch_sequence,
                               pn_kwargs=dict(allow_postpone=True, causal_postpone_tokenflow=True))
        assert L2 < L, (L2, L, 'truncation must genuinely cut the episode short')
        assert calls['n'] == 2 * L2 - 1, (calls['n'], L2, 'truncated')
    finally:
        agents_mod.topology_potential = real
    print(f"  terminal_zero_convention OK (natural L={L} calls={2*L-1}; "
          f"truncated L={L2} calls={2*L2-1})")


def test_policy_invariance_no_shaping_bias():
    """PRIMARY GATE, part 1: on M2 (make_shared_r, natural completion), the
    SMDP-discounted-return GAP between the two forced d1 choices must be
    numerically identical with and without shaping -- shaping may move each
    return's absolute magnitude but never the relative ranking between
    choices."""
    phi_coef, phi_decay, beta = 1.0, 0.9, 0.3
    kwargs = dict(causal_beta=beta, r_join=5, r1=5, r2=10)

    r_raw_A, t_raw_A, _ = _run_forced('use_R', phi_coef=0.0, **kwargs)
    r_raw_B, t_raw_B, _ = _run_forced('standalone', phi_coef=0.0, **kwargs)
    r_shp_A, t_shp_A, _ = _run_forced('use_R', phi_coef=phi_coef, phi_decay=phi_decay, **kwargs)
    r_shp_B, t_shp_B, _ = _run_forced('standalone', phi_coef=phi_coef, phi_decay=phi_decay, **kwargs)

    G_raw_A = _smdp_return(r_raw_A, t_raw_A, beta)
    G_raw_B = _smdp_return(r_raw_B, t_raw_B, beta)
    G_shp_A = _smdp_return(r_shp_A, t_shp_A, beta)
    G_shp_B = _smdp_return(r_shp_B, t_shp_B, beta)

    gap_raw = G_raw_A - G_raw_B
    gap_shp = G_shp_A - G_shp_B
    assert abs(gap_shp - gap_raw) < 1e-5, (gap_raw, gap_shp)
    # mechanism sanity: shaping must actually have DONE something, else the
    # invariance check above would be vacuous.
    assert abs(G_shp_A - G_raw_A) > 1e-6, 'shaping had no effect on A'
    assert abs(G_shp_B - G_raw_B) > 1e-6, 'shaping had no effect on B'
    print(f"  policy_invariance_no_shaping_bias OK: gap_raw={gap_raw:+.4f} "
          f"gap_shaped={gap_shp:+.4f}  (G_raw A/B={G_raw_A:.3f}/{G_raw_B:.3f}, "
          f"G_shaped A/B={G_shp_A:.3f}/{G_shp_B:.3f})")


def test_truncation_identity():
    """PRIMARY GATE, part 2: the single-trajectory identity
    G_shaped - G_raw == -phi_coef * Phi(s_0) must hold EXACTLY (terminal Phi
    forced to 0 regardless of why the episode ended). Sign check: shaping adds
    phi_coef*(disc_t*Phi_{t+1} - Phi_t) at every step; telescoping the
    discounted sum over the whole episode collapses to
    D_T*Phi_T - D_0*Phi_0 = 0 - Phi_0 = -Phi_0 (D_0=1, Phi_T:=0), so the net
    effect on the full return is a SUBTRACTION of Phi(s_0), not an addition
    (matches the M2 pairwise test's own numbers -- G_raw=15.0 -> G_shaped=
    12.48, a decrease of exactly Phi(s_0)=2.52 -- which passed without this
    sign ever being checked explicitly, since a pairwise gap cancels it).
    Checked on
    s1_stoch_sequence both at natural completion (~29 steps under an
    "always index 0" policy) and truncated at max_episode_length=7 (the
    smallest value safe from the value_batch_size=8 flush-loss edge case --
    see test_terminal_zero_convention's docstring) -- a genuinely shorter,
    different terminal state, which M2's 2-decision motif cannot produce (see
    _run_forced's docstring). This directly targets the terminal-Phi
    convention: if truncation leaked a nonzero Phi(s_T) instead of zeroing
    it, this identity would fail specifically (and only) on the truncated
    run, since natural completion and truncation reach different terminal
    markings with generally different Phi values."""
    from stoch_envs import make_s1_stoch_sequence
    phi_coef, phi_decay, beta = 1.0, 0.9, 0.3
    pn_kwargs = dict(allow_postpone=True, causal_postpone_tokenflow=True)

    # Phi(s_0) must be measured from the ACTUAL first call inside a real run,
    # not reconstructed from a freshly-built, never-reset net: env.reset()
    # calls pn.get_to_first_action() (environment.py:118), which can run
    # initial evolutions (e.g. an 'arrive' event) before the agent's first
    # decision, so the marking at t=0 already differs from the pristine
    # construction. Capture it via the same monkeypatch pattern as
    # test_terminal_zero_convention.
    first_calls = {}
    real = agents_mod.topology_potential

    def _capturing(pn, decay=0.9, cap=None):
        v = real(pn, decay=decay, cap=cap)
        first_calls.setdefault('phi0', v)
        return v

    for mel, tag in ((None, "natural completion"), (7, "truncated at 7 steps")):
        common = dict(causal_beta=beta, length=20, max_episode_length=mel,
                      pn_factory=make_s1_stoch_sequence, pn_kwargs=pn_kwargs)
        r_raw, t_raw, L_raw = _run_forced('__NONE__', phi_coef=0.0, **common)

        first_calls.clear()
        agents_mod.topology_potential = _capturing
        try:
            r_shp, t_shp, L_shp = _run_forced('__NONE__', phi_coef=phi_coef,
                                              phi_decay=phi_decay, **common)
        finally:
            agents_mod.topology_potential = real
        phi0 = first_calls['phi0']

        assert L_raw == L_shp, (tag, L_raw, L_shp, 'shaping must not change episode length')
        G_raw = _smdp_return(r_raw, t_raw, beta)
        G_shp = _smdp_return(r_shp, t_shp, beta)
        expected = -phi_coef * phi0
        assert abs((G_shp - G_raw) - expected) < 1e-4, \
            (tag, G_shp - G_raw, expected)
        print(f"  truncation_identity[{tag}] OK: L={L_raw}  "
              f"G_shaped-G_raw={G_shp-G_raw:+.4f}  phi_coef*Phi(s0)={expected:+.4f}")


def test_end_to_end_smoke_train():
    from config import SuiteConfig
    from pathlib import Path
    from run_suite import train_cell
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="phi_smoke_"))

    cfg = SuiteConfig(envs=["d_parallel_disjoint"], methods=["ppo_clip"], seeds=1,
                      epochs=3, episodes_per_epoch=4, test_freq=1, batch_size=8,
                      phi_coef=0.5, output_dir=out)
    m = train_cell("d_parallel_disjoint", "ppo_clip", 0, cfg, str(out / "train"))
    assert m is not None and 'env' in m
    assert all(v == v for v in m.get('sampled_curve', [])), "NaN in sampled_curve (ppo_clip+phi)"
    print(f"  end_to_end_smoke_train[ppo_clip+phi] OK (sampled_curve={m.get('sampled_curve')})")

    cfg2 = SuiteConfig(envs=["d_parallel_disjoint"], methods=["lrq"], seeds=1,
                       epochs=3, episodes_per_epoch=4, test_freq=1, batch_size=8,
                       phi_coef=0.5, output_dir=out)
    m2 = train_cell("d_parallel_disjoint", "lrq", 0, cfg2, str(out / "train"))
    assert m2 is not None and 'env' in m2
    assert all(v == v for v in m2.get('sampled_curve', [])), "NaN in sampled_curve (lrq+phi)"
    print(f"  end_to_end_smoke_train[lrq+phi, documented no-op] OK "
          f"(sampled_curve={m2.get('sampled_curve')})")


if __name__ == "__main__":
    test_place_weights_basic()
    test_cap_bounds_queue_contribution()
    test_terminal_zero_convention()
    test_policy_invariance_no_shaping_bias()
    test_truncation_identity()
    test_end_to_end_smoke_train()
    print("all phi-shaping tests passed")
