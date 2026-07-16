"""Validation tests for the flow_dag causal redistribution scheme.

Run: python _test_flow_dag.py
"""
from types import SimpleNamespace
from gympn.causal_traces import CausalTraces


def tok(i):
    return SimpleNamespace(_id=f"t{i}")


def tr(name):
    return SimpleNamespace(_id=name)


def build_basic():
    """root -> [action X] -> tA -> [evolution E, reward=10] consumes tA."""
    ct = CausalTraces()
    root = tok(0)
    # register root as initial
    ct.register_token(root, tr("__initial__"), parent_tokens=[], time=0)
    ct.register_transition(tr("__initial__"), [], [root], is_action=False, reward=0.0, time=0)

    # action X consumes root, produces tA
    tA = tok("A")
    actX = tr("X")
    ct.register_token(tA, actX, [root], time=1)
    ct.register_transition(actX, [root], [tA], is_action=True, reward=0.0, time=1)

    # evolution E consumes tA, produces tB, yields reward 10
    tB = tok("B")
    evE = tr("E")
    ct.register_token(tB, evE, [tA], time=2)
    ct.register_transition(evE, [tA], [tB], is_action=False, reward=10.0, time=2)
    return ct


def test_basic_attribution():
    ct = build_basic()
    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag")
    # one action transition (X) -> index 0 should get all the reward
    assert len(credit) == 1, credit
    assert abs(credit[0] - 10.0) < 1e-9, credit
    print("test_basic_attribution OK:", credit)


def test_gamma_leak():
    """With gamma<1, credit decays across intermediate pass-through hops; the
    remainder is left unassigned (uncontrollable)."""
    ct = CausalTraces()
    root = tok(0)
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root], is_action=False, reward=0.0, time=0)

    # action X consumes root -> tA (sink, attr=1 for X)
    tA = tok("A")
    ct.register_token(tA, tr("X"), [root], time=1)
    ct.register_transition(tr("X"), [root], [tA], is_action=True, reward=0.0, time=1)

    # pass-through evolution E1: tA -> tB  (one hop of decay)
    tB = tok("B")
    ct.register_token(tB, tr("E1"), [tA], time=2)
    ct.register_transition(tr("E1"), [tA], [tB], is_action=False, reward=0.0, time=2)

    # reward evolution E2 consumes tB -> credit = 10 * gamma^1 = 5 at gamma=0.5
    tC = tok("C")
    ct.register_token(tC, tr("E2"), [tB], time=3)
    ct.register_transition(tr("E2"), [tB], [tC], is_action=False, reward=10.0, time=3)

    credit = ct.redistribute_rewards(gamma=0.5, scheme="flow_dag")
    assert abs(credit[0] - 5.0) < 1e-9, credit  # 10 * 0.5; other 5 unassigned
    print("test_gamma_leak OK:", credit)


def test_conservation():
    ct = build_basic()
    total_reward = 10.0
    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag")
    assert sum(credit) <= total_reward + 1e-9, credit
    print("test_conservation OK: sum(credit)=", sum(credit))


def test_postpone_excluded():
    """A postpone action must receive zero credit but still occupy a slot."""
    ct = CausalTraces()
    root = tok(0)
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root], is_action=False, reward=0.0, time=0)

    # postpone re-IDs root -> tP (postpone is an action transition)
    tP = tok("P")
    pp = tr("postpone_abc")
    ct.register_token(tP, pp, [root], time=1)
    ct.register_transition(pp, [root], [tP], is_action=True, reward=0.0, time=1)

    # real action X consumes tP, produces tA
    tA = tok("A")
    actX = tr("X")
    ct.register_token(tA, actX, [tP], time=2)
    ct.register_transition(actX, [tP], [tA], is_action=True, reward=0.0, time=2)

    # evolution consumes tA, reward 10
    tB = tok("B")
    evE = tr("E")
    ct.register_token(tB, evE, [tA], time=3)
    ct.register_transition(evE, [tA], [tB], is_action=False, reward=10.0, time=3)

    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag", include_postpone=False)
    # action_transitions order: [postpone, X]
    assert len(credit) == 2, credit
    assert abs(credit[0] - 0.0) < 1e-9, ("postpone got credit", credit)
    assert abs(credit[1] - 10.0) < 1e-9, ("X should get all", credit)
    print("test_postpone_excluded OK:", credit)


def test_determinism():
    ct = build_basic()
    a = ct.redistribute_rewards(gamma=0.9, scheme="flow_dag")
    b = ct.redistribute_rewards(gamma=0.9, scheme="flow_dag")
    assert a == b, (a, b)
    print("test_determinism OK:", a)


def test_two_actions_split():
    """Reward consuming tokens from two different actions splits credit."""
    ct = CausalTraces()
    r1, r2 = tok("r1"), tok("r2")
    ct.register_token(r1, tr("__initial__"), [], time=0)
    ct.register_token(r2, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [r1, r2], is_action=False, reward=0.0, time=0)

    # action X -> tX from r1 ; action Y -> tY from r2
    tX, tY = tok("X"), tok("Y")
    ct.register_token(tX, tr("X"), [r1], time=1)
    ct.register_transition(tr("X"), [r1], [tX], is_action=True, reward=0.0, time=1)
    ct.register_token(tY, tr("Y"), [r2], time=1)
    ct.register_transition(tr("Y"), [r2], [tY], is_action=True, reward=0.0, time=1)

    # join transition consumes tX and tY, reward 10
    tZ = tok("Z")
    ct.register_token(tZ, tr("J"), [tX, tY], time=2)
    ct.register_transition(tr("J"), [tX, tY], [tZ], is_action=False, reward=10.0, time=2)

    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag")
    # inputs tX, tY -> averaged: 10 * (1/2)*onehot(X) + 10 * (1/2)*onehot(Y)
    assert abs(credit[0] - 5.0) < 1e-9 and abs(credit[1] - 5.0) < 1e-9, credit
    print("test_two_actions_split OK:", credit)


def build_action_with_own_reward():
    """root -> [action X consumes root, produces tA, reward=10 on X itself]."""
    ct = CausalTraces()
    root = tok(0)
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root], is_action=False, reward=0.0, time=0)

    tA = tok("A")
    actX = tr("X")
    ct.register_token(tA, actX, [root], time=1)
    # the action transition itself carries the reward
    ct.register_transition(actX, [root], [tA], is_action=True, reward=10.0, time=1)
    return ct


def test_self_credit_full():
    """self_credit=1.0: the acting transition keeps all of its own reward."""
    ct = build_action_with_own_reward()
    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag", self_credit=1.0)
    assert len(credit) == 1
    assert abs(credit[0] - 10.0) < 1e-9, credit
    print("test_self_credit_full OK:", credit)


def test_self_credit_zero():
    """self_credit=0.0: legacy behaviour — reward flows to ancestors only.
    Here the only ancestor is the root (uncontrollable), so X gets nothing."""
    ct = build_action_with_own_reward()
    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag", self_credit=0.0)
    assert abs(credit[0] - 0.0) < 1e-9, credit
    print("test_self_credit_zero OK:", credit)


def test_self_credit_split():
    """self_credit=0.25 with an enabling ancestor action.

    action W -> tW ; action X consumes tW, produces tA, reward=10 on X.
    With self_credit=0.25, X gets 2.5 directly; the remaining 7.5 flows to the
    enabling action W (tW is W's sink output)."""
    ct = CausalTraces()
    root = tok(0)
    ct.register_token(root, tr("__initial__"), [], time=0)
    ct.register_transition(tr("__initial__"), [], [root], is_action=False, reward=0.0, time=0)

    tW = tok("W")
    ct.register_token(tW, tr("W"), [root], time=1)
    ct.register_transition(tr("W"), [root], [tW], is_action=True, reward=0.0, time=1)

    tA = tok("A")
    ct.register_token(tA, tr("X"), [tW], time=2)
    ct.register_transition(tr("X"), [tW], [tA], is_action=True, reward=10.0, time=2)

    credit = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag", self_credit=0.25)
    # action order: [W, X]
    assert abs(credit[0] - 7.5) < 1e-9 and abs(credit[1] - 2.5) < 1e-9, credit
    print("test_self_credit_split OK:", credit)


def test_self_credit_ignores_evolution_reward():
    """Rewards on evolution transitions are unaffected by self_credit."""
    ct = build_basic()  # reward is on evolution E
    a = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag", self_credit=1.0)
    b = ct.redistribute_rewards(gamma=1.0, scheme="flow_dag", self_credit=0.0)
    assert a == b == [10.0], (a, b)
    print("test_self_credit_ignores_evolution_reward OK:", a)


if __name__ == "__main__":
    test_basic_attribution()
    test_gamma_leak()
    test_conservation()
    test_postpone_excluded()
    test_determinism()
    test_two_actions_split()
    test_self_credit_full()
    test_self_credit_zero()
    test_self_credit_split()
    test_self_credit_ignores_evolution_reward()
    print("\nAll flow_dag tests passed.")