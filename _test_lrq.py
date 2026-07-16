"""Validation tests for the LRQ (Lineage-Restricted Q) scheme.

LRQ is the fix proposed in CAUSAL_LRQ_PROPOSAL.md for REC's return-to-go /
chain-dilution bias (CAUSAL_REC_CRITICAL_REVIEW.md §3): each decision receives
the FULL discounted value of every future reward in whose causal lineage it
sits (no 1/|P| split, postpone included), and data.py consumes it as
A_t = Q_t - V(s_t) with NO GAE chaining over credits.

Run: python _test_lrq.py
"""
import math
from types import SimpleNamespace

import torch

from gympn.causal_traces import CausalTraces
from gympn.data import TrajectoryBuffer


def tok(i):
    return SimpleNamespace(_id=f"t{i}")


def tr(name):
    return SimpleNamespace(_id=name)


# --------------------------------------------------------------------------- #
# Trace builders
# --------------------------------------------------------------------------- #

def build_choice(choice):
    """The REC counterexample (see _counterexample_rec_returntogo.py).

    Decision 1 (action A @1) starts a two-stage chain. Decision 2 either fires
    X consuming the chain token -> reward 12 @3 (lineage {A, X}), or fires Y
    consuming a fresh initial token -> reward 7 @3 (lineage {Y})."""
    ct = CausalTraces()
    root, spare = tok(0), tok("spare")
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_token(spare, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root, spare], is_action=False, reward=0.0, time=0)
    tA = tok("A")
    ct.register_token(tA, tr("A"), [root], time=1)
    ct.register_transition(tr("A"), [root], [tA], is_action=True, reward=0.0, time=1)
    if choice == "X":
        tX = tok("X")
        ct.register_token(tX, tr("X"), [tA], time=2)
        ct.register_transition(tr("X"), [tA], [tX], is_action=True, reward=0.0, time=2)
        tE = tok("E")
        ct.register_token(tE, tr("E"), [tX], time=3)
        ct.register_transition(tr("E"), [tX], [tE], is_action=False, reward=12.0, time=3)
    else:
        tY = tok("Y")
        ct.register_token(tY, tr("Y"), [spare], time=2)
        ct.register_transition(tr("Y"), [spare], [tY], is_action=True, reward=0.0, time=2)
        tE = tok("E")
        ct.register_token(tE, tr("E"), [tY], time=3)
        ct.register_transition(tr("E"), [tY], [tE], is_action=False, reward=7.0, time=3)
    return ct


def build_postpone(wait, reward_now=10.0, reward_wait=10.0, tokenflow=True):
    """If wait: root -> [postpone @1] -> tP -> [X @2] -> tA -> [E @3, r=reward_wait].
    Else:      root -> [X @1] -> tA -> [E @2, r=reward_now]."""
    ct = CausalTraces()
    ct.postpone_tokenflow = tokenflow  # what the simulator flag would set
    root = tok(0)
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root], is_action=False, reward=0.0, time=0)
    if wait:
        tP = tok("P")
        ct.register_token(tP, tr("postpone_1"), [root], time=1)
        ct.register_transition(tr("postpone_1"), [root], [tP], is_action=True, reward=0.0, time=1)
        src, tx, te, r = tP, 2, 3, reward_wait
    else:
        src, tx, te, r = root, 1, 2, reward_now
    tA = tok("A")
    ct.register_token(tA, tr("X"), [src], time=tx)
    ct.register_transition(tr("X"), [src], [tA], is_action=True, reward=0.0, time=tx)
    tB = tok("B")
    ct.register_token(tB, tr("E"), [tA], time=te)
    ct.register_transition(tr("E"), [tA], [tB], is_action=False, reward=r, time=te)
    return ct


def build_exogenous():
    """Action X @1 plus a reward @2 with NO decision in its lineage (fires on an
    untouched initial-marking token). Under LRQ it must enter no one's Q."""
    ct = CausalTraces()
    root, r2 = tok(0), tok("r2")
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_token(r2, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root, r2], is_action=False, reward=0.0, time=0)
    tA = tok("A")
    ct.register_token(tA, tr("X"), [root], time=1)
    ct.register_transition(tr("X"), [root], [tA], is_action=True, reward=0.0, time=1)
    tC = tok("C")
    ct.register_token(tC, tr("Eexo"), [r2], time=2)
    ct.register_transition(tr("Eexo"), [r2], [tC], is_action=False, reward=5.0, time=2)
    return ct


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_counterexample_resolved():
    """The headline: decision 2 must rank the chain completion (full 12) above
    the exclusive alternative (7). The removed credits-as-rewards schemes
    ranked them 6 < 7 (the chain-dilution bias)."""
    qx = build_choice("X").redistribute_rewards(scheme="lrq", beta=0.0)
    qy = build_choice("Y").redistribute_rewards(scheme="lrq", beta=0.0)
    assert abs(qx[1] - 12.0) < 1e-9, ("decision 2 must see the FULL 12", qx)
    assert abs(qy[1] - 7.0) < 1e-9, qy
    assert qx[1] > qy[1], "LRQ must prefer completing the chain (12 > 7)"
    print("test_counterexample_resolved OK: lrq X=%s Y=%s" % (qx, qy))


def test_no_dilution_full_mass_per_lineage_member():
    """Every decision in a k-stage chain sees the full reward (Q-sample), and
    Sigma_t Q_t = k*r by design — LRQ is NOT a redistribution."""
    q = build_choice("X").redistribute_rewards(scheme="lrq", beta=0.0)
    assert abs(q[0] - 12.0) < 1e-9 and abs(q[1] - 12.0) < 1e-9, q
    assert abs(sum(q) - 24.0) < 1e-9, ("multi-counting is intended", q)
    print("test_no_dilution OK:", q)


def test_discount_from_decision_time():
    """Q_t discounts each lineage reward from the DECISION's own clock."""
    beta = 0.3
    q = build_choice("X").redistribute_rewards(scheme="lrq", beta=beta)
    assert abs(q[0] - 12.0 * math.exp(-beta * (3 - 1))) < 1e-9, q  # A @1, reward @3
    assert abs(q[1] - 12.0 * math.exp(-beta * (3 - 2))) < 1e-9, q  # X @2, reward @3
    print("test_discount_from_decision_time OK:", q)


def test_postpone_timing_penalty():
    """Same case, act now vs wait: acting now beats postponing by exactly the
    discount factor e^{-beta*wait}; postpone is IN the lineage (direct signal)."""
    beta = 0.3
    q_now = build_postpone(wait=False).redistribute_rewards(scheme="lrq", beta=beta)
    q_wait = build_postpone(wait=True).redistribute_rewards(scheme="lrq", beta=beta)
    # decision 0 alternatives: X@1 (reward @2) vs postpone@1 (reward @3)
    q_act = q_now[0]
    q_pp = q_wait[0]
    assert q_pp > 0.0, ("postpone is a lineage member under token-flow", q_wait)
    assert abs(q_act - 10.0 * math.exp(-beta * 1)) < 1e-9, q_now
    assert abs(q_pp - 10.0 * math.exp(-beta * 2)) < 1e-9, q_wait
    assert abs(q_pp / q_act - math.exp(-beta)) < 1e-9, (q_pp, q_act)
    print("test_postpone_timing_penalty OK: act=%.4f > postpone=%.4f" % (q_act, q_pp))


def test_postpone_batching_benefit():
    """A beneficial wait (waiting enables a LARGER reward) must show up as a
    larger Q for postpone — the direct benefit channel REC lacked."""
    beta = 0.3
    q_now = build_postpone(wait=False, reward_now=10.0).redistribute_rewards(scheme="lrq", beta=beta)
    q_wait = build_postpone(wait=True, reward_wait=20.0).redistribute_rewards(scheme="lrq", beta=beta)
    assert q_wait[0] > q_now[0], ("waiting for 20 must beat acting for 10", q_wait, q_now)
    print("test_postpone_batching_benefit OK: postpone=%.4f > act=%.4f"
          % (q_wait[0], q_now[0]))


def test_postpone_requires_tokenflow():
    """Sink-less postpone would make Q_postpone == 0 identically; LRQ refuses."""
    ct = build_postpone(wait=True, tokenflow=False)
    try:
        ct.redistribute_rewards(scheme="lrq", beta=0.3)
    except ValueError as e:
        assert "token-flow" in str(e)
        print("test_postpone_requires_tokenflow OK:", str(e)[:60], "...")
        return
    raise AssertionError("LRQ must raise without token-flow postpone")


def test_exogenous_ignored():
    """Uncontrollable rewards enter no one's Q (no smearing, no fallback)."""
    q = build_exogenous().redistribute_rewards(scheme="lrq", beta=0.0)
    assert abs(q[0] - 0.0) < 1e-12, ("exogenous reward must be ignored", q)
    print("test_exogenous_ignored OK: lrq=", q)


def test_mcq_ablation():
    """mc_q = LRQ without the lineage test: every causally-prior decision sees
    the FULL discounted return-to-go, caused or not."""
    # Trajectory Y: reward 7's lineage is {Y} only, but mc_q credits A too.
    qy_lrq = build_choice("Y").redistribute_rewards(scheme="lrq", beta=0.0)
    qy_mcq = build_choice("Y").redistribute_rewards(scheme="mc_q", beta=0.0)
    assert qy_lrq == [0.0, 7.0], qy_lrq
    assert qy_mcq == [7.0, 7.0], ("no lineage filter => A sees 7 too", qy_mcq)
    # On the chain the two coincide (both decisions are in the lineage).
    qx_mcq = build_choice("X").redistribute_rewards(scheme="mc_q", beta=0.0)
    assert qx_mcq == [12.0, 12.0], qx_mcq
    # Discounting is from each decision's own clock (A@1, X@2, reward@3).
    qx_b = build_choice("X").redistribute_rewards(scheme="mc_q", beta=0.3)
    assert abs(qx_b[0] - 12.0 * math.exp(-0.6)) < 1e-9, qx_b
    assert abs(qx_b[1] - 12.0 * math.exp(-0.3)) < 1e-9, qx_b
    # Postpone gets return-to-go credit too (no null-player, no tokenflow need).
    q_pp = build_postpone(wait=True, tokenflow=False).redistribute_rewards(
        scheme="mc_q", beta=0.0)
    assert q_pp[0] == 10.0, ("postpone sees the future return under mc_q", q_pp)
    print("test_mcq_ablation OK: lrq", qy_lrq, "vs mc_q", qy_mcq)


def test_lrq2_postpone_excluded_from_credit():
    """v2: production actions keep their lineage Q-samples; postpone gets NO
    lineage credit (its advantage becomes SMDP-TD in data.py), and the walk
    still passes through postpone's re-emitted tokens to upstream producers."""
    # Chain (no postpone): v2 == v1.
    q1 = build_choice("X").redistribute_rewards(scheme="lrq", beta=0.0)
    q2 = build_choice("X").redistribute_rewards(scheme="lrq2", beta=0.0)
    assert q1 == q2 == [12.0, 12.0], (q1, q2)
    # Postpone trace (token-flow): v1 credits postpone, v2 gives it 0 but the
    # downstream action still sees the full reward THROUGH the postpone hop.
    v1 = build_postpone(wait=True, tokenflow=True).redistribute_rewards(
        scheme="lrq", beta=0.0)
    v2 = build_postpone(wait=True, tokenflow=True).redistribute_rewards(
        scheme="lrq2", beta=0.0)
    assert v1 == [10.0, 10.0], v1
    assert v2 == [0.0, 10.0], ("postpone 0, production unchanged", v2)
    # v2 must not require token-flow postpone (no guard).
    v2_nf = build_postpone(wait=True, tokenflow=False).redistribute_rewards(
        scheme="lrq2", beta=0.0)
    assert v2_nf == [0.0, 10.0], v2_nf
    print("test_lrq2_postpone_excluded OK:", v1, "->", v2)


def test_lrq2_buffer_postpone_td():
    """data.py lrq2 branch: postpone steps get A = e^{-beta*tau}V(s')-V(s) and
    value target e^{-beta*tau}V(s'); production steps keep Q - V."""
    ct = build_postpone(wait=True, tokenflow=True)  # step0=postpone@1, step1=X@2
    beta = 0.5
    buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lrq2',
                           causal_rl=True, causal_beta=beta)
    values = [2.0, 4.0]
    for t, v in zip((1.0, 2.0), values):
        buf.store({}, 0, 0.0, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None, time=float(t))
    buf.finish(credits=ct, mode="replace")
    q = ct.redistribute_rewards(scheme="lrq2", beta=beta)
    d0 = math.exp(-beta * 1.0)  # tau step0->step1 = 1
    # postpone step: A = d0*V(s1) - V(s0); target = d0*V(s1)
    exp_adv = torch.tensor([d0 * values[1] - values[0], q[1] - values[1]])
    exp_ret = torch.tensor([d0 * values[1], q[1]])
    assert torch.allclose(buf.advantages_, exp_adv.float(), atol=1e-6), \
        (buf.advantages_, exp_adv)
    assert torch.allclose(buf.returns_, exp_ret.float(), atol=1e-6), \
        (buf.returns_, exp_ret)
    print("test_lrq2_buffer_postpone_td OK: adv", buf.advantages_.tolist())


def test_lrq3_decomposition():
    """v3: A = c_lineage + q_off_pred - V; V targets = full mc_q sample;
    q_off regression targets = mc_q - lrq2, all aligned per step."""
    beta = 0.5
    ct = build_postpone(wait=True, tokenflow=True)  # postpone@1, X@2, r=10@3
    lin = ct.redistribute_rewards(scheme="lrq2", beta=beta)     # [0, 10e^-b]
    full = ct.redistribute_rewards(scheme="mc_q", beta=beta)    # [10e^-2b, 10e^-b]
    assert abs(lin[0]) < 1e-9 and abs(full[0] - 10 * math.exp(-2 * beta)) < 1e-9

    buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lrq3',
                           causal_rl=True, causal_beta=beta)
    values = [2.0, 4.0]
    qoffs = [0.5, 0.2]
    for t, (v, q) in enumerate(zip(values, qoffs)):
        buf.store({}, 0, 0.0, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None, time=float(t + 1), qoff=q)
    buf.finish(credits=ct, mode="replace")

    exp_adv = torch.tensor([lin[0] + qoffs[0] - values[0],
                            lin[1] + qoffs[1] - values[1]])
    exp_ret = torch.tensor(full)
    exp_qt = torch.tensor([full[0] - lin[0], full[1] - lin[1]])
    assert torch.allclose(buf.advantages_, exp_adv.float(), atol=1e-6), \
        (buf.advantages_, exp_adv)
    assert torch.allclose(buf.returns_, exp_ret.float(), atol=1e-6), \
        (buf.returns_, exp_ret)
    assert torch.allclose(buf.qoff_targets_, exp_qt.float(), atol=1e-6), \
        (buf.qoff_targets_, exp_qt)
    print("test_lrq3_decomposition OK: adv", [round(x, 3) for x in buf.advantages_.tolist()])


def test_lcv_floor_and_adjustment():
    """LCV: A = A_GAE - c_hat*(cv - mean(cv)); perfect centering (cv == 0)
    must recover plain SMDP-GAE EXACTLY (the floor property)."""
    from gympn.data import smdp_gae
    beta = 0.5
    values = [2.0, 4.0]
    rewards = [1.0, 3.0]

    def build_buf(qoff_preds):
        ct = build_postpone(wait=True, tokenflow=True)
        buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lcv',
                               causal_rl=True, causal_beta=beta)
        for t, (v, r, q) in enumerate(zip(values, rewards, qoff_preds)):
            buf.store({'graph': None}, 0, r, torch.tensor(0.0), torch.tensor(v),
                      torch.zeros(1), token_ids=None, time=float(t + 1), qoff=q)
        return ct, buf

    # Expected GAE base (same math as finish): taus [1, 0], disc e^{-b*tau}
    r = torch.tensor(rewards)
    v = torch.tensor(values)
    disc = torch.exp(-beta * torch.tensor([1.0, 0.0]))
    dones = torch.tensor([False, True])
    adv_base = smdp_gae(r, v, disc, 1.0, dones=dones)

    # Perfect centering: qoff_pred == R_off per step -> cvterms == 0 -> floor.
    ct = build_postpone(wait=True, tokenflow=True)
    lin = ct.redistribute_rewards(scheme='lrq2', beta=beta)
    full = ct.redistribute_rewards(scheme='mc_q', beta=beta)
    roff = [f - l for f, l in zip(full, lin)]
    ct2, buf = build_buf(qoff_preds=roff)
    buf.finish(credits=ct2, mode='replace')
    assert torch.allclose(buf.advantages_, adv_base, atol=1e-6), \
        (buf.advantages_, adv_base)
    assert torch.allclose(buf.cvterms_, torch.zeros(2), atol=1e-6), buf.cvterms_
    # returns are the standard GAE targets (adv + V), not credit targets
    assert torch.allclose(buf.returns_, adv_base + v, atol=1e-6), buf.returns_
    print("test_lcv floor OK: adv == plain SMDP-GAE when cv == 0")

    # Imperfect centering: cvterms nonzero; get() must apply c_hat in [0, 2].
    ct3, buf2 = build_buf(qoff_preds=[0.0, 0.0])
    buf2.finish(credits=ct3, mode='replace')
    assert torch.allclose(buf2.cvterms_, torch.tensor(roff).float(), atol=1e-6)
    print("test_lcv adjustment OK: cvterms =", [round(x, 3) for x in buf2.cvterms_.tolist()])


def test_removed_schemes_raise():
    """Every non-LRQ scheme was removed and must fail loudly, not silently."""
    ct = build_choice("X")
    for scheme in ("flow_dag", "shapley_dag", "rec", "flow", "exponential",
                   "linear", "uniform", "depth", "hybrid"):
        try:
            ct.redistribute_rewards(scheme=scheme)
        except ValueError:
            continue
        raise AssertionError(f"removed scheme {scheme!r} must raise ValueError")
    print("test_removed_schemes_raise OK")


def test_determinism():
    a = build_choice("X").redistribute_rewards(scheme="lrq", beta=0.3)
    b = build_choice("X").redistribute_rewards(scheme="lrq", beta=0.3)
    assert a == b, (a, b)
    print("test_determinism OK:", a)


def test_buffer_lrq_branch():
    """data.py consumption: returns = Q (value targets), A = Q - V, no GAE
    chaining over credits; hybrid mu mixes in SMDP-GAE on raw rewards."""
    def fill(buf):
        values = [1.0, 2.0, 3.0]
        for t, v in enumerate(values):
            buf.store({}, 0, 0.0, torch.tensor(0.0), torch.tensor(v),
                      torch.zeros(1), token_ids=None, time=float(t))
        return torch.tensor(values)

    credits = [4.0, 0.0, 6.0]

    buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lrq',
                           causal_rl=True, causal_beta=0.0, causal_mu=0.0)
    values = fill(buf)
    buf.finish(credits=credits, mode="replace")
    expected_adv = torch.tensor(credits) - values
    assert torch.allclose(buf.returns_, torch.tensor(credits)), buf.returns_
    assert torch.allclose(buf.advantages_, expected_adv), buf.advantages_

    # mu = 1.0 -> pure SMDP-GAE on raw rewards (all zero here) with lam=1:
    # A_t = sum of deltas = V-telescope = -V(s_t) + V(s_T bootstrap dropped)
    buf2 = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lrq',
                            causal_rl=True, causal_beta=0.0, causal_mu=1.0)
    values2 = fill(buf2)
    buf2.finish(credits=credits, mode="replace")
    assert torch.allclose(buf2.advantages_, -values2), buf2.advantages_
    print("test_buffer_lrq_branch OK: adv=", buf.advantages_.tolist(),
          "hybrid(mu=1)=", buf2.advantages_.tolist())


def test_smdp_discount_matches_constant_gamma():
    """smdp_gae with a CONSTANT discount d must equal compute_advantages with
    gamma=d (certifies the two GAE implementations differ only in how the
    discount varies over time, not in the recursion itself)."""
    from gympn.data import smdp_gae, compute_advantages
    rewards = torch.tensor([1.0, 2.0, 0.5, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5, 2.0])
    dones = torch.tensor([False, False, False, True])
    gamma, lam = 0.9, 0.8
    discounts = torch.full((4,), gamma)
    a_smdp = smdp_gae(rewards, values, discounts, lam, dones=dones)
    a_const = compute_advantages(rewards, values, gamma, lam, dones=dones)
    assert torch.allclose(a_smdp, a_const, atol=1e-6), (a_smdp, a_const)
    print("test_smdp_discount_matches_constant_gamma OK")


def test_smdp_discount_flag_standard_path():
    """TrajectoryBuffer(smdp_discount=True, causal_rl=False) -- the lcv0
    method -- must match smdp_gae called directly with e^{-beta*tau} from the
    buffered decision times, and must NOT use `gam` at all."""
    from gympn.data import smdp_gae
    beta = 0.5
    times = [0.0, 1.0, 3.0]     # taus = [1, 2, 0]
    rewards = [1.0, 0.0, 4.0]
    values = [2.0, 1.0, 3.0]

    buf = TrajectoryBuffer(gam=0.99, lam=0.95, causal_rl=False,
                           causal_beta=beta, smdp_discount=True)
    for t, v, r in zip(times, values, rewards):
        buf.store({'graph': None}, 0, r, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None, time=t)
    buf.finish(credits=None)

    r_t, v_t = torch.tensor(rewards), torch.tensor(values)
    taus = torch.tensor([1.0, 2.0, 0.0])
    disc = torch.exp(-beta * taus)
    dones = torch.tensor([False, False, True])
    expected_adv = smdp_gae(r_t, v_t, disc, 0.95, dones=dones)

    assert torch.allclose(buf.advantages_, expected_adv, atol=1e-6), \
        (buf.advantages_, expected_adv)
    assert torch.allclose(buf.returns_, expected_adv + v_t, atol=1e-6), buf.returns_

    # gam=0.5 instead of 0.99 must not change the result at all (unused path).
    buf_gam_check = TrajectoryBuffer(gam=0.5, lam=0.95, causal_rl=False,
                                     causal_beta=beta, smdp_discount=True)
    for t, v, r in zip(times, values, rewards):
        buf_gam_check.store({'graph': None}, 0, r, torch.tensor(0.0), torch.tensor(v),
                            torch.zeros(1), token_ids=None, time=t)
    buf_gam_check.finish(credits=None)
    assert torch.allclose(buf_gam_check.advantages_, expected_adv, atol=1e-6)
    print("test_smdp_discount_flag_standard_path OK: adv =",
          [round(x, 3) for x in buf.advantages_.tolist()])


def test_smdp_discount_lcv0_equals_lcv_floor():
    """The [ppo -> lcv0 -> lcv] chain, factored: lcv0 (standard path,
    smdp_discount=True) must produce the SAME advantages as the lcv branch
    with perfect centering (cv==0, the floor case already covered by
    test_lcv_floor_and_adjustment) -- both are plain SMDP-GAE on identical
    rewards/values/times/beta, just reached via different code paths."""
    from gympn.data import smdp_gae
    beta = 0.5
    values = [2.0, 4.0]
    rewards = [1.0, 3.0]

    buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_rl=False,
                           causal_beta=beta, smdp_discount=True)
    for t, (v, r) in enumerate(zip(values, rewards)):
        buf.store({'graph': None}, 0, r, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None, time=float(t + 1))
    buf.finish(credits=None)

    r_t, v_t = torch.tensor(rewards), torch.tensor(values)
    disc = torch.exp(-beta * torch.tensor([1.0, 0.0]))
    dones = torch.tensor([False, True])
    adv_base = smdp_gae(r_t, v_t, disc, 1.0, dones=dones)

    assert torch.allclose(buf.advantages_, adv_base, atol=1e-6), \
        (buf.advantages_, adv_base)
    print("test_smdp_discount_lcv0_equals_lcv_floor OK")


def test_lva_buffer_floor_and_aux_targets():
    """LVA: the policy path must be EXACTLY plain SMDP-GAE on raw rewards
    (same numbers as the lcv floor and the lcv0 standard path -- closing the
    lcv==lcv0==lva floor triangle), with the lrq2 lineage credits stored as
    the critic's auxiliary regression targets and value targets = adv + V."""
    from gympn.data import smdp_gae
    beta = 0.5
    values = [2.0, 4.0]
    rewards = [1.0, 3.0]

    ct = build_postpone(wait=True, tokenflow=True)
    buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lva',
                           causal_rl=True, causal_beta=beta)
    for t, (v, r) in enumerate(zip(values, rewards)):
        buf.store({'graph': None}, 0, r, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None, time=float(t + 1))
    buf.finish(credits=ct, mode='replace')

    r_t, v_t = torch.tensor(rewards), torch.tensor(values)
    disc = torch.exp(-beta * torch.tensor([1.0, 0.0]))
    dones = torch.tensor([False, True])
    adv_base = smdp_gae(r_t, v_t, disc, 1.0, dones=dones)

    assert torch.allclose(buf.advantages_, adv_base, atol=1e-6), \
        (buf.advantages_, adv_base)
    assert torch.allclose(buf.returns_, adv_base + v_t, atol=1e-6), buf.returns_

    lin = build_postpone(wait=True, tokenflow=True).redistribute_rewards(
        scheme='lrq2', beta=beta)
    assert torch.allclose(buf.qlin_targets_, torch.tensor(lin).float(), atol=1e-6), \
        (buf.qlin_targets_, lin)
    # No CV terms on this path (aux-only consumption).
    assert buf.cvterms_.numel() == 0, buf.cvterms_
    print("test_lva_buffer_floor_and_aux_targets OK: adv =",
          [round(x, 3) for x in buf.advantages_.tolist()],
          "qlin =", [round(x, 3) for x in buf.qlin_targets_.tolist()])


def test_smdp_discount_silent_fallback_guard():
    """smdp_discount=True with causal_beta>0 on an episode where decision
    times were never recorded (all default to 0.0) must raise loudly instead
    of silently degenerating to an undiscounted (beta-inert) GAE."""
    buf = TrajectoryBuffer(gam=0.99, lam=0.95, causal_rl=False,
                           causal_beta=0.5, smdp_discount=True)
    for v, r in zip([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]):
        buf.store({'graph': None}, 0, r, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None)  # time= omitted -> defaults to 0.0
    try:
        buf.finish(credits=None)
    except RuntimeError as e:
        assert "smdp_discount" in str(e)
        print("test_smdp_discount_silent_fallback_guard OK:", e)
        return
    raise AssertionError("expected RuntimeError for unrecorded decision times")


if __name__ == "__main__":
    test_counterexample_resolved()
    test_no_dilution_full_mass_per_lineage_member()
    test_discount_from_decision_time()
    test_postpone_timing_penalty()
    test_postpone_batching_benefit()
    test_postpone_requires_tokenflow()
    test_exogenous_ignored()
    test_mcq_ablation()
    test_lrq2_postpone_excluded_from_credit()
    test_lrq2_buffer_postpone_td()
    test_lrq3_decomposition()
    test_lcv_floor_and_adjustment()
    test_removed_schemes_raise()
    test_determinism()
    test_buffer_lrq_branch()
    test_smdp_discount_matches_constant_gamma()
    test_smdp_discount_flag_standard_path()
    test_smdp_discount_lcv0_equals_lcv_floor()
    test_lva_buffer_floor_and_aux_targets()
    test_smdp_discount_silent_fallback_guard()
    print("\nAll LRQ tests passed.")
