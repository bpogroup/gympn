"""Environment factories for the causal-RL stability suite.

Each factory reproduces exactly one of the paper_examples (a..h) and returns a
fresh `GymProblem`, parameterized by `causal_rl` and `allow_postpone` so the same
topology can be trained under either method and evaluated with or without the
postpone action.

The first 8 environments form a 2x4 grid:
    {sequence, parallel, loop, exclusive-choice} x {joint, disjoint}
"joint" = the two stages/queues share a single resource pool; "disjoint" = each
stage has its own pool. A 9th, `i_mixed_credit`, sits outside the grid: it is
the one topology whose reward-types split non-degenerately into LS-HCA's PURE
and CONTESTED sets (see its own comment block). All use the same
`perfect_heuristic` (assign task-type t to the employee that is fastest for it:
type 0 -> employee 0, type 1 -> employee 1).
"""
from simpn.simulator import SimToken
from gympn.simulator import GymProblem


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def _start(c, r):
    """Service-time model shared by every environment.

    Employee 0 is fast (delay 1) for type-0 tasks, employee 1 is fast for type-1;
    the cross assignment is delay 2; employee 2 is slow (delay 3) for anything.
    """
    if (c['task_type'] == 0 and r['code_employee'] == 0) or \
       (c['task_type'] == 1 and r['code_employee'] == 1):
        return [SimToken((c, r), delay=1)]
    elif (c['task_type'] == 0 and r['code_employee'] == 1) or \
         (c['task_type'] == 1 and r['code_employee'] == 0):
        return [SimToken((c, r), delay=2)]
    else:
        return [SimToken((c, r), delay=3)]


def perfect_heuristic(observable_net, tokens_comb):
    """Optimal reference policy: always assign a task to its fastest employee.

    Robust to a 'postpone' pseudo-binding (skips it) and to non-matching
    binding shapes, so it can run on either env variant without crashing.
    """
    for k, bindings in tokens_comb.items():
        for binding in bindings:
            try:
                if binding and binding[0] and binding[0][0] == 'postpone':
                    continue
                task = binding[0][1].value
                resource = binding[1][1].value
            except (IndexError, TypeError, AttributeError):
                continue
            if (task['task_type'] == 0 and resource['code_employee'] == 0) or \
               (task['task_type'] == 1 and resource['code_employee'] == 1):
                return {k: binding}
    return None


def _pool(agency, name):
    """Add a 3-employee resource pool (codes 0, 1, 2)."""
    pool = agency.add_var(name, var_attributes=['code_employee'])
    for eid in range(3):
        pool.put({'code_employee': eid})
    return pool


# ---------------------------------------------------------------------------
# a) sequence, joint  — one task flows stage1 -> stage2, single shared pool
# ---------------------------------------------------------------------------
def make_a_sequence_joint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    employee1 = _pool(ag, "employee1")

    ag.add_event([arrival], [arrival, waiting1], lambda a: [SimToken(a, delay=1), SimToken(a)],
                 name='arrive')
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, waiting2],
                 lambda b: [SimToken(b[-1]), SimToken({'task_type': 1})], name='done1')
    ag.add_action([waiting2, employee1], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee1], lambda b: [SimToken(b[-1])],
                 name='done2', reward_function=lambda x: 1)
    return ag


# ---------------------------------------------------------------------------
# b) sequence, disjoint — stage1 and stage2 each have their own pool
# ---------------------------------------------------------------------------
def make_b_sequence_disjoint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    employee1 = _pool(ag, "employee1")
    employee2 = _pool(ag, "employee2")

    ag.add_event([arrival], [arrival, waiting1], lambda a: [SimToken(a, delay=1), SimToken(a)],
                 name='arrive')
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, waiting2],
                 lambda b: [SimToken(b[-1]), SimToken({'task_type': 1})], name='done1')
    ag.add_action([waiting2, employee2], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee2], lambda b: [SimToken(b[-1])],
                 name='done2', reward_function=lambda x: 1)
    return ag


# ---------------------------------------------------------------------------
# c) parallel, joint — type0->stage1, type1->stage2 in parallel, shared pool,
#    reward only when BOTH sub-tasks of a case complete
# ---------------------------------------------------------------------------
def make_c_parallel_joint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type', 'case_id'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type', 'case_id'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee', 'case_id'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type', 'case_id'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee', 'case_id'])
    completed1 = ag.add_var("completed1", var_attributes=['task_type', 'case_id'])
    completed2 = ag.add_var("completed2", var_attributes=['task_type', 'case_id'])
    arrival.put({'task_type': 0, 'case_id': 0})
    arrival.put({'task_type': 1, 'case_id': 0})
    employee1 = _pool(ag, "employee1")

    def arrive(a):
        a['case_id'] += 1
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        return [SimToken(a, delay=1), None, SimToken(a)]

    ag.add_event([arrival], [arrival, waiting1, waiting2], arrive)
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, completed1],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done1')
    ag.add_action([waiting2, employee1], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee1, completed2],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done2')
    ag.add_event([completed1, completed2], [], behavior=lambda x, y: [], name='doneFinal',
                 reward_function=lambda x, y: 1, guard=lambda e1, e2: e1['case_id'] == e2['case_id'])
    return ag


# ---------------------------------------------------------------------------
# d) parallel, disjoint — like c but each queue has its own pool
# ---------------------------------------------------------------------------
def make_d_parallel_disjoint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type', 'case_id'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type', 'case_id'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee', 'case_id'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type', 'case_id'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee', 'case_id'])
    completed1 = ag.add_var("completed1", var_attributes=['task_type', 'case_id'])
    completed2 = ag.add_var("completed2", var_attributes=['task_type', 'case_id'])
    arrival.put({'task_type': 0, 'case_id': 0})
    arrival.put({'task_type': 1, 'case_id': 0})
    employee1 = _pool(ag, "employee1")
    employee2 = _pool(ag, "employee2")

    def arrive(a):
        a['case_id'] += 1
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        return [SimToken(a, delay=1), None, SimToken(a)]

    ag.add_event([arrival], [arrival, waiting1, waiting2], arrive)
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, completed1],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done1')
    ag.add_action([waiting2, employee2], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee2, completed2],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done2')
    ag.add_event([completed1, completed2], [], behavior=lambda x, y: [], name='doneFinal',
                 reward_function=lambda x, y: 1, guard=lambda e1, e2: e1['case_id'] == e2['case_id'])
    return ag


# ---------------------------------------------------------------------------
# e) loop, joint — a wrong stage-2 assignment sends the task back to stage 1
#    (rework loop), single shared pool
# ---------------------------------------------------------------------------
def make_e_loop_joint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    completed = ag.add_var("completed", var_attributes=['task_type'])
    arrival.put({'task_type': 0})
    employee1 = _pool(ag, "employee1")

    ag.add_event([arrival], [arrival, waiting1], lambda a: [SimToken(a, delay=1), SimToken(a)],
                 name='arrive')
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, waiting2],
                 lambda b: [SimToken(b[-1]), SimToken({'task_type': 1})], name='done1')
    ag.add_action([waiting2, employee1], [busy2], behavior=_start, name="start2")

    def complete2(b):
        # Correct (fast) assignment -> case completes; otherwise rework: back to waiting1.
        if (b[0]['task_type'] == 0 and b[1]['code_employee'] == 0) or \
           (b[0]['task_type'] == 1 and b[1]['code_employee'] == 1):
            return [SimToken(b[-1]), None, SimToken(b[0])]
        return [SimToken(b[-1]), SimToken(b[0]), None]

    ag.add_event([busy2], [employee1, waiting1, completed], complete2, name='done2')
    ag.add_event([completed], [], behavior=lambda x: [], name='complete_case',
                 reward_function=lambda x: 1)
    return ag


# ---------------------------------------------------------------------------
# f) loop, disjoint — rework loop, each stage its own pool
# ---------------------------------------------------------------------------
def make_f_loop_disjoint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    completed = ag.add_var("completed", var_attributes=['task_type'])
    arrival.put({'task_type': 0})
    employee1 = _pool(ag, "employee1")
    employee2 = _pool(ag, "employee2")

    ag.add_event([arrival], [arrival, waiting1], lambda a: [SimToken(a, delay=1), SimToken(a)],
                 name='arrive')
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, waiting2],
                 lambda b: [SimToken(b[-1]), SimToken({'task_type': 1})], name='done1')
    ag.add_action([waiting2, employee2], [busy2], behavior=_start, name="start2")

    def complete2(b):
        if (b[0]['task_type'] == 0 and b[1]['code_employee'] == 0) or \
           (b[0]['task_type'] == 1 and b[1]['code_employee'] == 1):
            return [SimToken(b[-1]), None, SimToken(b[0])]
        return [SimToken(b[-1]), SimToken({'task_type': 0}), None]

    ag.add_event([busy2], [employee2, waiting1, completed], complete2, name='done2')
    ag.add_event([completed], [], behavior=lambda x: [], name='complete_case',
                 reward_function=lambda x: 1)
    return ag


# ---------------------------------------------------------------------------
# g) exclusive choice, joint — one pool must choose which queue to serve;
#    each completed task rewards independently
# ---------------------------------------------------------------------------
def make_g_exclusive_choice_joint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 1})
    employee = _pool(ag, "employee")

    def arrive(a):
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        return [SimToken(a, delay=1), None, SimToken(a)]

    ag.add_event([arrival], [arrival, waiting1, waiting2], arrive)
    ag.add_action([waiting1, employee], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee], lambda b: [SimToken(b[-1])],
                 name='done1', reward_function=lambda x: 1)
    ag.add_action([waiting2, employee], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee], lambda b: [SimToken(b[-1])],
                 name='done2', reward_function=lambda x: 1)
    return ag


# ---------------------------------------------------------------------------
# h) exclusive choice, disjoint — two queues, each its own pool
# ---------------------------------------------------------------------------
def make_h_exclusive_choice_disjoint(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 1})
    employee1 = _pool(ag, "employee1")
    employee2 = _pool(ag, "employee2")

    def arrive(a):
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        return [SimToken(a, delay=1), None, SimToken(a)]

    ag.add_event([arrival], [arrival, waiting1, waiting2], arrive)
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1], lambda b: [SimToken(b[-1])],
                 name='done1', reward_function=lambda x: 1)
    ag.add_action([waiting2, employee2], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee2], lambda b: [SimToken(b[-1])],
                 name='done2', reward_function=lambda x: 1)
    return ag


# ---------------------------------------------------------------------------
# i) mixed credit — the only env in the suite where LS-HCA's PURE/CONTESTED
#    split is non-degenerate.
#
# Motivation (`_diag_pure_scan.py`): every other env sits at a trivial corner
# of `_pure_contested_reward_types`. Nine are 100% CONTESTED (PURE=set(), so
# the exact lineage term is identically zero and LS-HCA reduces to plain HCA);
# `h_exclusive_choice_disjoint` and `s2_stoch_scaled` are 100% PURE (nothing
# left for the hindsight correction, so LS-HCA reduces to exact lineage credit,
# which lrq2 already computes). LS-HCA's actual premise -- lineage credits part
# of the return exactly, hindsight only patches the shared remainder -- was
# therefore never exercised by any env in the suite.
#
# This is `d_parallel_disjoint` with one change: `done1`/`done2` now carry
# their own reward. That is enough, because reachability decides the split:
#   reaches(start1) = {done1, doneFinal},  reaches(start2) = {done2, doneFinal}
#   => PURE(start1) = {done1},  CONTESTED(start1) = {doneFinal}
# The private rewards are unreachable from the sibling decision (disjoint
# pools, so the reachability walk cannot cross over via a shared resource
# hand-back), while `doneFinal` needs a completed token from BOTH stages of
# the same case and so is genuinely shared. Per case the mass splits 2:1
# pure:contested (1 + 1 private, 1 shared), all rewards 1 as everywhere else
# in the grid.
#
# There is deliberately no "joint" counterpart: sharing one pool puts every
# reward-type in reach of both decisions (exactly why `c_parallel_joint` is
# all-CONTESTED), so disjoint pools are a precondition for a nonempty PURE
# set, not a free variation.
# ---------------------------------------------------------------------------
def make_i_mixed_credit(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type', 'case_id'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type', 'case_id'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee', 'case_id'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type', 'case_id'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee', 'case_id'])
    completed1 = ag.add_var("completed1", var_attributes=['task_type', 'case_id'])
    completed2 = ag.add_var("completed2", var_attributes=['task_type', 'case_id'])
    arrival.put({'task_type': 0, 'case_id': 0})
    arrival.put({'task_type': 1, 'case_id': 0})
    employee1 = _pool(ag, "employee1")
    employee2 = _pool(ag, "employee2")

    def arrive(a):
        a['case_id'] += 1
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        return [SimToken(a, delay=1), None, SimToken(a)]

    ag.add_event([arrival], [arrival, waiting1, waiting2], arrive)
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, completed1],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done1',
                 reward_function=lambda x: 1)
    ag.add_action([waiting2, employee2], [busy2], behavior=_start, name="start2")
    ag.add_event([busy2], [employee2, completed2],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done2',
                 reward_function=lambda x: 1)
    ag.add_event([completed1, completed2], [], behavior=lambda x, y: [], name='doneFinal',
                 reward_function=lambda x, y: 1, guard=lambda e1, e2: e1['case_id'] == e2['case_id'])
    return ag


# ---------------------------------------------------------------------------
# j) mixed credit + stage-2 rework — PURE floor AND a contested reward whose
#    lineage the ACTION actually moves. The env LS-HCA was designed for.
#
# Motivation (`_diag_ls_hca_step0.py` across four envs). Two things must hold
# at once for LS-HCA to be more than either of its degenerate limits, and no
# existing env had both:
#   - a nonempty PURE set, so the exact lineage term carries real mass;
#     `i_mixed_credit` has this (21%), the rest of the suite does not.
#   - conditional signal I(A;Z|X) in the CONTESTED part, so the hindsight
#     correction has something to say; the rework loops `e_loop_joint` /
#     `f_loop_disjoint` have this (0.019/0.020 nats/record, 27-29x
#     `i_mixed_credit`'s 0.00069), and they have PURE=set().
# The two were mutually exclusive for a structural reason: a rework loop makes
# every reward mutually reachable, so reachability-based PURE is impossible in
# a loop.
#
# The way out is to CONFINE the loop to one stage. This is `i_mixed_credit`
# with stage-2 rework that returns the case to `waiting2` -- its own queue --
# rather than to `waiting1`. Then start2's reachable set stays
# {done2, doneFinal}: the loop cycles through waiting2/employee2/busy2, none of
# which lead to done1. So both private rewards remain PURE exactly as in
# `i_mixed_credit`, while whether a given start2 decision ends up in
# `doneFinal`'s lineage now depends on whether that attempt succeeded -- which
# is a function of the ACTION, not just of which case was touched. (That
# case-identity determinism is precisely why `i_mixed_credit`'s contested
# signal is ~0: `doneFinal` is a join on case_id, so a decision's lineage
# membership there is fixed before the action is chosen.)
#
# Stage 2 is the only stage with rework, mirroring `e`/`f`. `done2` rewards
# only a SUCCESSFUL completion, so a rework firing hands the resource back and
# returns the case without paying out; it remains a reward-type either way,
# which is what keeps it in the PURE classification. The reworked token keeps
# its case_id so `doneFinal`'s join still pairs the two streams correctly.
# `perfect_heuristic` never triggers rework (it assigns type-1 tasks to
# employee 1), so the optimum is clean while a random policy reworks often --
# which should also widen the headroom this env inherits from
# `d_parallel_disjoint`.
#
# Whether the contested signal actually materializes here is an empirical
# question, not a claim: measure it with `_diag_ls_hca_step0.py j_mixed_rework`.
# ---------------------------------------------------------------------------
def _stage2_ok(b):
    """Stage-2 assignment is correct iff the task went to its fast employee."""
    return ((b[0]['task_type'] == 0 and b[1]['code_employee'] == 0) or
            (b[0]['task_type'] == 1 and b[1]['code_employee'] == 1))


def make_j_mixed_rework(causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type', 'case_id'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type', 'case_id'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee', 'case_id'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type', 'case_id'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee', 'case_id'])
    completed1 = ag.add_var("completed1", var_attributes=['task_type', 'case_id'])
    completed2 = ag.add_var("completed2", var_attributes=['task_type', 'case_id'])
    arrival.put({'task_type': 0, 'case_id': 0})
    arrival.put({'task_type': 1, 'case_id': 0})
    employee1 = _pool(ag, "employee1")
    employee2 = _pool(ag, "employee2")

    def arrive(a):
        a['case_id'] += 1
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        return [SimToken(a, delay=1), None, SimToken(a)]

    ag.add_event([arrival], [arrival, waiting1, waiting2], arrive)

    # stage 1: no rework, private reward
    ag.add_action([waiting1, employee1], [busy1], behavior=_start, name="start1")
    ag.add_event([busy1], [employee1, completed1],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done1',
                 reward_function=lambda x: 1)

    # stage 2: rework back into its OWN queue, private reward only on success
    ag.add_action([waiting2, employee2], [busy2], behavior=_start, name="start2")

    def complete2(b):
        if _stage2_ok(b):
            return [SimToken(b[-1]), None, SimToken(b[0])]   # resource, -, completed2
        return [SimToken(b[-1]), SimToken(b[0]), None]       # resource, rework, -

    ag.add_event([busy2], [employee2, waiting2, completed2], complete2, name='done2',
                 reward_function=lambda b: 1 if _stage2_ok(b) else 0)

    ag.add_event([completed1, completed2], [], behavior=lambda x, y: [], name='doneFinal',
                 reward_function=lambda x, y: 1, guard=lambda e1, e2: e1['case_id'] == e2['case_id'])
    return ag


ENV_BUILDERS = {
    "a_sequence_joint": make_a_sequence_joint,
    "b_sequence_disjoint": make_b_sequence_disjoint,
    "c_parallel_joint": make_c_parallel_joint,
    "d_parallel_disjoint": make_d_parallel_disjoint,
    "e_loop_joint": make_e_loop_joint,
    "f_loop_disjoint": make_f_loop_disjoint,
    "g_exclusive_choice_joint": make_g_exclusive_choice_joint,
    "h_exclusive_choice_disjoint": make_h_exclusive_choice_disjoint,
    "i_mixed_credit": make_i_mixed_credit,
    "j_mixed_rework": make_j_mixed_rework,
}


# Stochastic / scaled tier (E5) — registered alongside the a-h grid so
# make_env/run_suite work uniformly.
from stoch_envs import STOCH_BUILDERS, STOCH_HEURISTICS  # noqa: E402

ENV_BUILDERS.update(STOCH_BUILDERS)

# Per-env heuristic anchors: the a-h grid shares perfect_heuristic (a true
# optimum); the stochastic tier uses strong myopic rules (normalized > 1.0
# is possible there and means the policy beat the myopic anchor).
HEURISTICS = {name: perfect_heuristic for name in ENV_BUILDERS
              if name not in STOCH_BUILDERS}
HEURISTICS.update(STOCH_HEURISTICS)


def make_env(name, causal_rl=False, allow_postpone=True, causal_postpone_tokenflow=False):
    if name not in ENV_BUILDERS:
        raise KeyError(f"Unknown env '{name}'. Known: {list(ENV_BUILDERS)}")
    return ENV_BUILDERS[name](causal_rl=causal_rl, allow_postpone=allow_postpone,
                              causal_postpone_tokenflow=causal_postpone_tokenflow)