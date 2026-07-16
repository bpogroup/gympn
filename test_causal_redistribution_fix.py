"""Test causal RL redistribution fix."""
import uuid, types
from gympn.causal_traces import CausalTraces
def mt(tid=None):
    t = types.SimpleNamespace(); t._id = tid or str(uuid.uuid4()); return t
def test_postpone_excluded():
    ct = CausalTraces()
    w0, e0 = mt("w0"), mt("e0")
    pp = mt("postpone_" + str(uuid.uuid4()))
    w1, e1 = mt("w1"), mt("e1")
    ct.register_transition(pp, [w0, e0], [w1, e1], is_action=True, reward=0.0, time=0.0)
    ct.register_token(w1, pp, [w0, e0], time=0.0)
    ct.register_token(e1, pp, [w0, e0], time=0.0)
    st = mt("start")
    busy = mt("busy")
    ct.register_transition(st, [w1, e1], [busy], is_action=True, reward=0.0, time=1.0)
    ct.register_token(busy, st, [w1, e1], time=1.0)
    co = mt("complete")
    er = mt("emp_ret")
    ct.register_transition(co, [busy], [er], is_action=False, reward=1.0, time=1.5)
    ct.register_token(er, co, [busy], time=1.5)
    credits = ct.redistribute_rewards(gamma=1.0, scheme="depth")
    print("Credits:", credits)
    assert len(credits) == 2
    # New expected behavior: postpone actions receive (some) credit.
    assert credits[0] > 0.0, "Postpone should receive positive credit, got " + str(credits[0])
    # Both actions together should account for the full reward
    assert abs(sum(credits) - 1.0) < 1e-6, "Credits do not sum to 1.0: " + str(credits)
    print("PASS: postpone excluded")
def test_input_tracing():
    ct = CausalTraces()
    w, e = mt("w"), mt("e")
    st = mt("start"); busy = mt("busy")
    ct.register_transition(st, [w, e], [busy], is_action=True, reward=0.0, time=0.0)
    ct.register_token(busy, st, [w, e], time=0.0)
    co = mt("complete"); er = mt("er")
    ct.register_transition(co, [busy], [er], is_action=False, reward=1.0, time=0.5)
    ct.register_token(er, co, [busy], time=0.5)
    credits = ct.redistribute_rewards(gamma=1.0, scheme="depth")
    assert abs(credits[0] - 1.0) < 1e-6
    print("PASS: input tracing")
def test_depth_weighting():
    ct = CausalTraces()
    t = [mt("t" + str(i)) for i in range(4)]
    a1 = mt("a1"); a2 = mt("a2")
    ct.register_transition(a1, [t[0]], [t[1]], is_action=True, reward=0, time=0)
    ct.register_token(t[1], a1, [t[0]], time=0)
    ct.register_transition(a2, [t[1]], [t[2]], is_action=True, reward=0, time=1)
    ct.register_token(t[2], a2, [t[1]], time=1)
    ev = mt("ev")
    ct.register_transition(ev, [t[2]], [t[3]], is_action=False, reward=1.0, time=2)
    ct.register_token(t[3], ev, [t[2]], time=2)
    credits = ct.redistribute_rewards(gamma=1.0, scheme="depth")
    assert credits[1] > credits[0], "Closer action should get more"
    assert abs(sum(credits) - 1.0) < 1e-6
    print("PASS: depth weighting a1=" + str(round(credits[0],3)) + " a2=" + str(round(credits[1],3)))
if __name__ == "__main__":
    test_postpone_excluded()
    test_input_tracing()
    test_depth_weighting()
    print("\nALL TESTS PASSED")
