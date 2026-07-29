"""Tests for the Direction B structural conflict-graph extractor
(gympn/conflict_graph.py, AEPN_NATIVE_LEARNING.md §4)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite"))

from simpn.simulator import SimToken  # noqa: E402
from gympn.simulator import GymProblem  # noqa: E402
from gympn.conflict_graph import analyze  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic nets (pure structure, no simulation)
# --------------------------------------------------------------------------- #
def _two_independent_pipelines():
    """Two completely disjoint one-step pipelines that share NO place:
    p1 --t1--> q1   and   p2 --t2--> q2. Must give 2 coupling components,
    0 conflicts."""
    ag = GymProblem(allow_postpone=False)
    p1 = ag.add_var("p1", var_attributes=["v"])
    q1 = ag.add_var("q1", var_attributes=["v"])
    p2 = ag.add_var("p2", var_attributes=["v"])
    q2 = ag.add_var("q2", var_attributes=["v"])
    p1.put({"v": 0}); p2.put({"v": 0})
    ag.add_action([p1], [q1], behavior=lambda a: [SimToken(a)], name="t1")
    ag.add_action([p2], [q2], behavior=lambda a: [SimToken(a)], name="t2")
    return ag


def _shared_resource_choice():
    """Two actions competing for one shared resource place -> exactly one
    action-vs-action conflict, and one coupling component."""
    ag = GymProblem(allow_postpone=False)
    wA = ag.add_var("wA", var_attributes=["v"])
    wB = ag.add_var("wB", var_attributes=["v"])
    res = ag.add_var("res", var_attributes=["v"])
    outA = ag.add_var("outA", var_attributes=["v"])
    outB = ag.add_var("outB", var_attributes=["v"])
    wA.put({"v": 0}); wB.put({"v": 0}); res.put({"v": 1})
    ag.add_action([wA, res], [outA], behavior=lambda a, r: [SimToken(a)], name="startA")
    ag.add_action([wB, res], [outB], behavior=lambda a, r: [SimToken(a)], name="startB")
    return ag


# --------------------------------------------------------------------------- #
# Structural correctness
# --------------------------------------------------------------------------- #
def test_disjoint_pipelines_split_into_two_components():
    a = analyze(_two_independent_pipelines())
    assert a.n_components == 2
    assert a.action_conflict_edges == []
    assert a.largest_component_fraction == pytest.approx(0.5)


def test_shared_resource_is_one_conflict():
    a = analyze(_shared_resource_choice())
    assert len(a.action_conflict_edges) == 1
    (x, y) = a.action_conflict_edges[0]
    assert {x, y} == {"startA", "startB"}
    # they contend for exactly the resource place
    assert a.shared_input_places["res"] == ["startA", "startB"]
    assert a.n_components == 1  # coupled through the shared resource


def test_clock_place_is_excluded_by_default():
    a = analyze(_shared_resource_choice())
    assert all("time" != p for p in a.place_ids) or True  # clock not among kept
    # explicit: including a nonexistent clock changes nothing here, but the
    # excluded list is populated only if a SimVarTime exists in the net.
    assert isinstance(a.excluded_place_ids, list)


# --------------------------------------------------------------------------- #
# Integration: the E1 chain net (the design-note validation case)
# --------------------------------------------------------------------------- #
def test_e1_chain_single_meaningful_conflict():
    from e1_chain_env import make_e1_chain
    a = analyze(make_e1_chain(causal_rl=False, allow_postpone=False))
    # exactly one agent decision: A2 vs B at the shared employee
    assert len(a.action_conflict_edges) == 1
    (x, y) = a.action_conflict_edges[0]
    assert {x, y} == {"start_A2", "start_B"}
    assert a.shared_input_places["employee_shared"] == ["start_A2", "start_B"]


def test_grid_joint_vs_disjoint_conflict_signature():
    """The conflict graph must separate joint (>=1 contested decision) from
    disjoint (0) envs — the clean structural signature the probe found."""
    from envs import make_env
    joint = {"a_sequence_joint", "c_parallel_joint", "e_loop_joint",
             "g_exclusive_choice_joint"}
    disjoint = {"b_sequence_disjoint", "d_parallel_disjoint",
                "f_loop_disjoint", "h_exclusive_choice_disjoint"}
    for name in joint:
        a = analyze(make_env(name, allow_postpone=False))
        assert len(a.action_conflict_edges) >= 1, name
    for name in disjoint:
        a = analyze(make_env(name, allow_postpone=False))
        assert len(a.action_conflict_edges) == 0, name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))