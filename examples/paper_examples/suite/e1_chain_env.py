"""E1: the chain-vs-shortcut counterexample environment (PAPER_PLAN_LRQ.md).

Purpose: turn the two-decision REC counterexample into a live A-E PN so the
chain-dilution bias can be shown IN-PIPELINE: redistribution schemes that
split a reward over its lineage (rec: 12/2 = 6 per stage) rank the exclusive
shortcut (reward 7) above completing the chain (reward 12) at the shared
resource, and TD converges to the shortcut. LRQ sees the full 12 and prefers
the chain. Requires r_chain/2 < r_short < r_chain: 6 < 7 < 12.

Topology:
  - Stage A1: waiting_A1 --[start_A1, DEDICATED employee]--> busy_A1
              --done_A1--> waiting_A2 (+ employee back).  No reward.
  - Stage A2: waiting_A2 --[start_A2, SHARED employee]--> busy_A2
              --done_A2--> reward R_CHAIN (12).
  - Shortcut: waiting_B --[start_B, SHARED employee]--> busy_B
              --done_B--> reward R_SHORT (7).
  The dedicated employee only serves A1, so stage-1 work is FREE with respect
  to the contested resource; the only economically meaningful choice is
  A2-vs-B at the shared employee. True per-slot value: 12 vs 7 -> serve A2.
  Diluted per-slot credit (rec): 6 vs 7 -> serve B (the trap).

Deterministic (fixed initial marking, unit service times), so greedy eval is
exact and the optimum is computable in closed form:
  horizon 10 => ~10 shared slots; first A2 is ready at t=1 =>
  optimum = 1*R_SHORT + 9*R_CHAIN = 115 ; all-B trap = 10*R_SHORT = 70.
"""
from simpn.simulator import SimToken

from gympn.simulator import GymProblem

R_CHAIN = 12.0
R_SHORT = 7.0


def make_e1_chain(causal_rl=False, allow_postpone=True,
                  causal_postpone_tokenflow=False, n_a=10, n_b=12):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    waiting_A1 = ag.add_var("waiting_A1", var_attributes=['task_type'])
    busy_A1 = ag.add_var("busy_A1", var_attributes=['task_type', 'code_employee'])
    waiting_A2 = ag.add_var("waiting_A2", var_attributes=['task_type'])
    busy_A2 = ag.add_var("busy_A2", var_attributes=['task_type', 'code_employee'])
    waiting_B = ag.add_var("waiting_B", var_attributes=['task_type'])
    busy_B = ag.add_var("busy_B", var_attributes=['task_type', 'code_employee'])

    employee_ded = ag.add_var("employee_ded", var_attributes=['code_employee'])
    employee_ded.put({'code_employee': 0})
    employee_shared = ag.add_var("employee_shared", var_attributes=['code_employee'])
    employee_shared.put({'code_employee': 1})

    for _ in range(n_a):
        waiting_A1.put({'task_type': 0})
    for _ in range(n_b):
        waiting_B.put({'task_type': 1})

    def start(c, r):
        return [SimToken((c, r), delay=1)]

    ag.add_action([waiting_A1, employee_ded], [busy_A1], behavior=start, name="start_A1")
    ag.add_event([busy_A1], [employee_ded, waiting_A2],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done_A1')

    ag.add_action([waiting_A2, employee_shared], [busy_A2], behavior=start, name="start_A2")
    ag.add_event([busy_A2], [employee_shared], lambda b: [SimToken(b[-1])],
                 name='done_A2', reward_function=lambda x: R_CHAIN)

    ag.add_action([waiting_B, employee_shared], [busy_B], behavior=start, name="start_B")
    ag.add_event([busy_B], [employee_shared], lambda b: [SimToken(b[-1])],
                 name='done_B', reward_function=lambda x: R_SHORT)
    return ag


def _priority_heuristic(order):
    """Pick the first available REAL binding whose transition matches the
    priority order; never postpones."""
    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None
        def tr_id(b):
            try:
                return getattr(b[2], '_id', None) or getattr(b[2], 'name', None)
            except Exception:
                return None
        for want in order:
            for b in bindings:
                tid = tr_id(b)
                if tid is not None and str(tid).startswith(want):
                    return b
        # fall back to any real binding
        for b in bindings:
            if tr_id(b) is not None:
                return b
        return None
    return heuristic


# Optimal: keep the A pipeline running and always prefer A2 at the shared
# employee (B only fills otherwise-idle shared slots, e.g. t=0).
chain_heuristic = _priority_heuristic(["start_A2", "start_A1", "start_B"])
# The trap policy that diluted credit predicts: prefer the shortcut.
shortcut_heuristic = _priority_heuristic(["start_B", "start_A1", "start_A2"])
# Stage-generic variants for the one-shot envs (any chain stage over B / vice
# versa; at most one stage is ever available at a given decision point).
chain_heuristic_any = _priority_heuristic(["start_A", "start_B"])
shortcut_heuristic_any = _priority_heuristic(["start_B", "start_A"])


def make_e1b_oneshot(causal_rl=False, allow_postpone=True,
                     causal_postpone_tokenflow=False, stages=2):
    """E1b: ONE-SHOT FORECLOSURE variant — the decision the E1 experiment
    showed TD can heal is made terminal so the bootstrap cannot rescue a
    backward split.

    Why: in E1 the trap decision repeats and the foregone A2 token stays in
    the queue, so V(s') carries the mass a redistribution sent backward and
    the per-step bias cancels (rec reached the optimum 5/5 there). Here the
    shared employee only becomes available at t=1 and the horizon is 2: one
    real choice — complete the chain (A2, reward 12, lineage {A1, A2} =>
    diluted share 6) or serve the shortcut (B, reward 7, exclusive) — then the
    episode ends. V(s') ~ 0, so a diluted per-step credit compares 6 < 7 and
    prefers the shortcut on EVERY trajectory, while the true values are
    12 > 7.

    Anchors: optimum 12 (chain), trap 7 (B), postpone-everything 0.

    ``stages`` = total chain length k. Stages 1..k-1 run on the dedicated
    employee before t=1 (service time 0.5/(k-1) each); the final stage
    contests the shared employee against B. A per-lineage equal split gives
    the deciding stage only R_CHAIN/k — deeper chains widen the trap's pull
    (k=2: 6 vs 7; k=3: 4 vs 7) while the true comparison stays 12 vs 7.
    """
    assert stages >= 2
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)

    waitings, busies = [], []
    for i in range(1, stages + 1):
        waitings.append(ag.add_var(f"waiting_A{i}", var_attributes=['task_type']))
        busies.append(ag.add_var(f"busy_A{i}", var_attributes=['task_type', 'code_employee']))
    waiting_B = ag.add_var("waiting_B", var_attributes=['task_type'])
    busy_B = ag.add_var("busy_B", var_attributes=['task_type', 'code_employee'])

    employee_ded = ag.add_var("employee_ded", var_attributes=['code_employee'])
    employee_ded.put({'code_employee': 0})
    # The shared employee ARRIVES at t=1 (via a delayed evolution), so the
    # final-stage-vs-B choice happens exactly once, at t=1, horizon 2.
    starter = ag.add_var("starter", var_attributes=['task_type'])
    starter.put({'task_type': 0})
    employee_shared = ag.add_var("employee_shared", var_attributes=['code_employee'])
    ag.add_event([starter], [employee_shared],
                 lambda a: [SimToken({'code_employee': 1}, delay=1)],
                 name='shared_arrive')

    waitings[0].put({'task_type': 0})
    waiting_B.put({'task_type': 1})

    def start(c, r):
        return [SimToken((c, r), delay=1)]

    # Dedicated stages must ALL finish before t=1 (the A-phase at a new clock
    # precedes the E-phase, so a stage completing exactly at 1 would make the
    # final stage invisible at the choice). The simulator requires behaviors
    # to take exactly one parameter per input variable, hence the closure.
    d = 0.5 / (stages - 1)

    def _make_start(delay):
        def start_fn(c, r):
            return [SimToken((c, r), delay=delay)]
        return start_fn

    start_ded = _make_start(d)

    for i in range(stages - 1):
        ag.add_action([waitings[i], employee_ded], [busies[i]],
                      behavior=start_ded, name=f"start_A{i + 1}")
        ag.add_event([busies[i]], [employee_ded, waitings[i + 1]],
                     lambda b: [SimToken(b[-1]), SimToken(b[0])],
                     name=f'done_A{i + 1}')

    ag.add_action([waitings[-1], employee_shared], [busies[-1]],
                  behavior=start, name=f"start_A{stages}")
    ag.add_event([busies[-1]], [employee_shared], lambda b: [SimToken(b[-1])],
                 name=f'done_A{stages}', reward_function=lambda x: R_CHAIN)

    ag.add_action([waiting_B, employee_shared], [busy_B], behavior=start, name="start_B")
    ag.add_event([busy_B], [employee_shared], lambda b: [SimToken(b[-1])],
                 name='done_B', reward_function=lambda x: R_SHORT)
    return ag


if __name__ == "__main__":
    # Smoke: verify the closed-form baseline returns.
    from gympn.solvers import HeuristicSolver, RandomSolver
    import numpy as np
    import random

    def run(solver_factory, seeds=5):
        out = []
        for s in range(seeds):
            random.seed(s); np.random.seed(s)
            env = make_e1_chain(causal_rl=False, allow_postpone=False)
            out.append(float(env.testing_run(solver=solver_factory(), length=10)))
        return out

    opt = run(lambda: HeuristicSolver(chain_heuristic), seeds=1)
    trap = run(lambda: HeuristicSolver(shortcut_heuristic), seeds=1)
    rnd = run(lambda: RandomSolver(), seeds=10)
    print(f"chain (optimal) heuristic : {opt}   (expected ~115)")
    print(f"shortcut (trap) heuristic : {trap}   (expected ~70)")
    print(f"random                    : mean={np.mean(rnd):.1f} ± {np.std(rnd):.1f}  {rnd}")
    assert opt[0] > trap[0], "chain policy must dominate the shortcut policy"

    for k in (2, 3):
        def run_b(solver_factory, seeds=1, _k=k):
            out = []
            for s in range(seeds):
                random.seed(s); np.random.seed(s)
                env = make_e1b_oneshot(causal_rl=False, allow_postpone=False, stages=_k)
                out.append(float(env.testing_run(solver=solver_factory(), length=2)))
            return out

        opt_b = run_b(lambda: HeuristicSolver(chain_heuristic_any))
        trap_b = run_b(lambda: HeuristicSolver(shortcut_heuristic_any))
        rnd_b = run_b(lambda: RandomSolver(), seeds=10)
        print(f"\nE1b one-shot (stages={k}): chain={opt_b} (expected 12)  "
              f"shortcut={trap_b} (expected 7)  random mean={np.mean(rnd_b):.1f}")
        assert opt_b[0] == 12.0 and trap_b[0] == 7.0, (k, opt_b, trap_b)
