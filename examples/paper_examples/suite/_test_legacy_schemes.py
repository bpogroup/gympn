"""Validate the resurrected legacy schemes against their known-good values.

Expected values come from the deleted _test_rec.py / the REC counterexample
(both verified in-session before the schemes were removed):
  - rec, two-stage chain, reward 12       -> [6, 6]   (equal split)
  - rec, counterexample X vs Y            -> G@dec2: 6 (X) vs 7 (Y)  (the bias)
  - lrq still routes to the CURRENT implementation -> [12, 12] on the chain
  - legacy buffer consumption: returns = smdp return-to-go over credits.

Run: python _test_legacy_schemes.py
"""
import math
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import legacy_schemes
from gympn.causal_traces import CausalTraces
from gympn.data import TrajectoryBuffer

legacy_schemes.install(gamma=0.9, self_credit=0.5, legacy_lam=0.0)


def tok(i):
    return SimpleNamespace(_id=f"t{i}")


def tr(name):
    return SimpleNamespace(_id=name)


def build_chain(choice="X"):
    """The REC counterexample trace: A@1 starts a chain; decision 2 fires X
    (chain completion, reward 12, lineage {A,X}) or Y (exclusive, reward 7)."""
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


def test_rec_equal_split():
    c = build_chain("X").redistribute_rewards(scheme="rec", beta=0.0)
    assert len(c) == 2 and abs(c[0] - 6.0) < 1e-9 and abs(c[1] - 6.0) < 1e-9, c
    print("rec equal split OK:", c)


def test_rec_return_equivalence():
    """S1: discounted credit-return == discounted reward-return, any beta."""
    for beta in (0.0, 0.3, 1.0):
        ct = build_chain("X")
        c = ct.redistribute_rewards(scheme="rec", beta=beta)
        u = [a.get("time") for a in ct.transition_history.get_action_transitions()]
        lhs = sum(math.exp(-beta * (u[t] - u[0])) * c[t] for t in range(len(c)))
        rhs = 12.0 * math.exp(-beta * (3 - u[0]))
        assert abs(lhs - rhs) < 1e-9, (beta, lhs, rhs)
    print("rec return-equivalence OK")


def test_rec_counterexample_bias():
    """The documented failure REC must still exhibit (it is the paper's E1):
    return-to-go at decision 2 ranks the 7-shortcut above the 12-chain."""
    cx = build_chain("X").redistribute_rewards(scheme="rec", beta=0.0)
    cy = build_chain("Y").redistribute_rewards(scheme="rec", beta=0.0)
    assert cx[1] < cy[1], ("rec must show the 6<7 dilution bias", cx, cy)
    print(f"rec counterexample bias reproduced: G@dec2 chain={cx[1]} < shortcut={cy[1]}")


def test_flow_dag_runs_and_conserves():
    c = build_chain("X").redistribute_rewards(scheme="flow_dag", beta=0.0)
    assert len(c) == 2 and all(v >= 0 for v in c) and sum(c) <= 12.0 + 1e-9, c
    print("flow_dag OK (mass-bounded):", c)


def test_lrq_unaffected():
    c = build_chain("X").redistribute_rewards(scheme="lrq", beta=0.0)
    assert abs(c[0] - 12.0) < 1e-9 and abs(c[1] - 12.0) < 1e-9, c
    print("lrq still routes to current implementation OK:", c)


def test_legacy_buffer_consumption():
    """rec + buffer: returns must be the smdp return-to-go over credits
    (NOT the LRQ Q-targets), advantages the smdp-gae at legacy_lam=0."""
    buf = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='rec',
                           causal_rl=True, causal_beta=0.0)
    values = [1.0, 2.0]
    for t, v in enumerate(values):
        buf.store({}, 0, 0.0, torch.tensor(0.0), torch.tensor(v),
                  torch.zeros(1), token_ids=None, time=float(t + 1))
    buf.finish(credits=build_chain("X"), mode="replace")
    # credits [6,6]; beta=0 -> returns-to-go [12, 6]; TD(0) adv:
    # A_0 = 6 + V1 - V0 = 6+2-1 = 7 ; A_1 = 6 + 0 - 2 = 4
    assert torch.allclose(buf.returns_, torch.tensor([12.0, 6.0])), buf.returns_
    assert torch.allclose(buf.advantages_, torch.tensor([7.0, 4.0])), buf.advantages_
    # and an lrq buffer is untouched by the patch:
    buf2 = TrajectoryBuffer(gam=1.0, lam=1.0, causal_scheme='lrq',
                            causal_rl=True, causal_beta=0.0)
    for t, v in enumerate(values):
        buf2.store({}, 0, 0.0, torch.tensor(0.0), torch.tensor(v),
                   torch.zeros(1), token_ids=None, time=float(t + 1))
    buf2.finish(credits=build_chain("X"), mode="replace")
    assert torch.allclose(buf2.returns_, torch.tensor([12.0, 12.0])), buf2.returns_
    assert torch.allclose(buf2.advantages_, torch.tensor([11.0, 10.0])), buf2.advantages_
    print("legacy buffer consumption OK: rec returns", buf.returns_.tolist(),
          "adv", buf.advantages_.tolist(), "| lrq untouched")


if __name__ == "__main__":
    test_rec_equal_split()
    test_rec_return_equivalence()
    test_rec_counterexample_bias()
    test_flow_dag_runs_and_conserves()
    test_lrq_unaffected()
    test_legacy_buffer_consumption()
    print("\nAll legacy-scheme tests passed.")
