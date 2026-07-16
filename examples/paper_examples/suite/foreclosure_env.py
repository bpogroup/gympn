"""Foreclosure-stressing benchmark env (the Gap-D discriminator).

Motivation (see CAUSAL_GAP_D_FORECLOSURE.md): the a–h suite cannot test Gap D —
its envs are decoupled or mild enough that the value baseline absorbs any
contention. To actually discriminate foreclosure-aware credit (Route B/C) from
plain lineage credit (+ value baseline, Route A) we need a problem whose OPTIMUM
*requires declining a locally-good binding to avoid foreclosing a globally-better
one*. This is the defining feature of combinatorial optimization (foreclosure =
the coupling that makes a problem combinatorial).

Design — "scarce specialist, one shot" (a unit-weight knapsack / online selection
in disguise):

  * One SPECIALIST resource, consumed after a single use (one shot per episode).
  * `n_gen` GENERALIST resources, renewable (return after each job).
  * `n_low` LOW tasks present at t=0; ONE HIGH task arrives at t=`h_arrival`.
  * Reward depends on (task, resource):
        specialist + HIGH = `spec_high` (10)   <- the big prize
        specialist + LOW  = `spec_low`  (3)    <- the TRAP (locally tempting)
        generalist + any  = `gen_reward`(1)
  * With `n_gen < n_low`, lows queue on the single generalist, so at most epochs
    a low waits while only the specialist is free — the agent is *tempted* to spend
    its one specialist shot on a low (+3 now, lineage-credited positively) instead
    of HOLDING it (postpone, 0 lineage credit) for the high task (+10 later).

Why this isolates Gap D:
  - Forward lineage credits `specialist->low` a positive +3 (looks great, beats the
    generalist's +1), and credits postpone/declining ~0. A lineage-only signal
    therefore *reinforces the trap*.
  - The optimal play is to DECLINE the +3 to preserve the +10 — i.e. price the
    foreclosed alternative. Only the value baseline (Route A) or a
    foreclosure-aware credit (Route B/C) can represent that.

Optimum (intended reference policy): serve lows with the generalist, hold the
specialist (postpone when only specialist+low is available) until the high task
arrives, then specialist->high; after the high is captured, the specialist is free
to take a low (+3 > +1). The `perfect_heuristic` below encodes exactly this.

This module is standalone (its own heuristic + baselines), because the suite's
shared `perfect_heuristic` is assignment-specific. Run `python foreclosure_env.py`
for a self-test that prints Random / greedy-trap / optimal returns and checks the
trap spread.
"""
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import HeuristicSolver


# --- default problem parameters (tune difficulty here) ---
DEFAULTS = dict(
    n_low=3,          # low tasks present at t=0
    n_gen=1,          # generalist resources (renewable). n_gen < n_low => postpone needed
    h_arrival=3.0,    # clock time the high task becomes available
    spec_high=10.0,   # specialist serves the high task
    spec_low=3.0,     # specialist serves a low task  (the locally-tempting trap)
    gen_reward=1.0,   # generalist serves anything
    gen_time=1.0,     # generalist service duration
    spec_time=1.0,    # specialist service duration (it is consumed regardless)
)


def make_foreclosure(causal_rl=False, allow_postpone=True, **overrides):
    """Build the foreclosure benchmark GymProblem. See module docstring."""
    p = {**DEFAULTS, **overrides}
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl)

    arrival = ag.add_var("arrival", var_attributes=['task_type', 'case_id'])
    waiting = ag.add_var("waiting", var_attributes=['task_type', 'case_id'])
    busy_s = ag.add_var("busy_s", var_attributes=['task_type', 'case_id', 'code_employee'])
    busy_g = ag.add_var("busy_g", var_attributes=['task_type', 'case_id', 'code_employee'])
    specialist = ag.add_var("specialist", var_attributes=['code_employee'])
    generalist = ag.add_var("generalist", var_attributes=['code_employee'])
    done = ag.add_var("done", var_attributes=['task_type', 'case_id'])

    # initial marking: low tasks ready now; the high task scheduled via `arrival`.
    for i in range(p["n_low"]):
        waiting.put({'task_type': 0, 'case_id': i + 1})
    arrival.put({'task_type': 1, 'case_id': 0})           # the high task (one shot)
    specialist.put({'code_employee': 0})                  # single, consumed on use
    for r in range(p["n_gen"]):
        generalist.put({'code_employee': 1 + r})

    # the high task becomes available at t = h_arrival (one-shot arrival).
    ag.add_event([arrival], [waiting],
                 lambda a: [SimToken(a, delay=p["h_arrival"])], name='arriveH')

    # actions: choose specialist OR generalist for a waiting task.
    ag.add_action([waiting, specialist], [busy_s],
                  behavior=lambda c, r: [SimToken((c, r), delay=p["spec_time"])],
                  name="start_spec")
    ag.add_action([waiting, generalist], [busy_g],
                  behavior=lambda c, r: [SimToken((c, r), delay=p["gen_time"])],
                  name="start_gen")

    # specialist is CONSUMED (busy_s -> done only; the specialist token is not returned).
    ag.add_event([busy_s], [done],
                 lambda b: [SimToken(b[0])],
                 name='complete_spec',
                 reward_function=lambda b: (p["spec_high"] if b[0]['task_type'] == 1
                                            else p["spec_low"]))
    # generalist is RENEWED (returns to its pool).
    ag.add_event([busy_g], [generalist, done],
                 lambda b: [SimToken(b[1]), SimToken(b[0])],
                 name='complete_gen',
                 reward_function=lambda b: p["gen_reward"])
    return ag


# --------------------------------------------------------------------------- #
# Heuristics                                                                   #
# --------------------------------------------------------------------------- #
def _classify(tokens_comb):
    """Bucket the enabled bindings by (task is high?, resource is specialist?)."""
    buckets = dict(spec_high=None, spec_low=None, gen_high=None, gen_low=None)
    for k, blist in tokens_comb.items():
        for b in blist:
            task = b[0][1].value
            is_spec = (b[1][0]._id == 'specialist')
            high = (task.get('task_type') == 1)
            key = ('spec' if is_spec else 'gen') + ('_high' if high else '_low')
            if buckets[key] is None:
                buckets[key] = {k: b}
    return buckets


def _high_still_obtainable(net):
    """Is a high task anywhere upstream (not yet completed)? arrival/waiting/busy."""
    for pid in ('arrival', 'waiting', 'busy_s', 'busy_g'):
        for tv in HeuristicSolver.get_place_tokens(pid, net):
            t = tv[0] if isinstance(tv, tuple) else tv     # busy_* hold (task, res)
            if isinstance(t, dict) and t.get('task_type') == 1:
                return True
    return False


def perfect_heuristic(net, tokens_comb, bindings=None):
    """Foreclosure-aware optimum: hold the one-shot specialist for the high task.

    Priority while a high task is still obtainable:
        high->specialist  >  low->generalist  >  high->generalist  >  POSTPONE
    (never low->specialist; postpone rather than burn the specialist on a low).
    Once no high remains, the specialist is free to take a low (+3 > +1):
        low->specialist   >  low->generalist
    """
    b = _classify(tokens_comb)
    if _high_still_obtainable(net):
        for key in ('spec_high', 'gen_low', 'gen_high'):
            if b[key] is not None:
                return b[key]
        return 'postpone'                      # only specialist+low left -> hold it
    # high already captured/gone: specialist is now best spent on a low.
    for key in ('spec_high', 'spec_low', 'gen_low', 'gen_high'):
        if b[key] is not None:
            return b[key]
    return 'postpone'


def greedy_heuristic(net, tokens_comb, bindings=None):
    """Foreclosure-BLIND trap policy: always take the highest immediate reward.

    spec_high(10) > spec_low(3) > gen(1). Never postpones, so it burns the one-shot
    specialist on the first low it sees -> foreclosing the +10. This is what a
    lineage-credited policy is pulled toward; included as a contrast baseline.
    """
    b = _classify(tokens_comb)
    for key in ('spec_high', 'spec_low', 'gen_low', 'gen_high'):
        if b[key] is not None:
            return b[key]
    return 'postpone'


# --------------------------------------------------------------------------- #
# Self-test / baselines                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os, random
    import numpy as np
    from gympn.solvers import RandomSolver

    LENGTH = 8

    def avg(make_solver, seed0=0, n=20, **ov):
        rs = []
        for i in range(n):
            os.environ["PYTHONHASHSEED"] = str(seed0 + i)
            random.seed(seed0 + i); np.random.seed(seed0 + i)
            env = make_foreclosure(causal_rl=False, allow_postpone=True, **ov)
            rs.append(float(env.testing_run(solver=make_solver(), length=LENGTH)))
        return float(np.mean(rs)), float(np.std(rs))

    print(f"foreclosure benchmark self-test (length={LENGTH}, defaults={DEFAULTS})\n")
    rnd = avg(lambda: RandomSolver(), seed0=10_000)
    greedy = avg(lambda: HeuristicSolver(greedy_heuristic), seed0=20_000)
    perfect = avg(lambda: HeuristicSolver(perfect_heuristic), seed0=30_000)
    print(f"  Random        : {rnd[0]:6.2f} ± {rnd[1]:.2f}")
    print(f"  greedy (trap) : {greedy[0]:6.2f} ± {greedy[1]:.2f}   <- foreclosure-blind")
    print(f"  perfect (opt) : {perfect[0]:6.2f} ± {perfect[1]:.2f}   <- foreclosure-aware")
    print(f"\n  headroom (perfect - random) = {perfect[0] - rnd[0]:.2f}")
    print(f"  trap gap (perfect - greedy) = {perfect[0] - greedy[0]:.2f}")
    assert perfect[0] > greedy[0] + 1e-6, "FAIL: optimum should beat the greedy trap"
    assert perfect[0] > rnd[0] + 1e-6, "FAIL: optimum should beat random"
    print("\n  OK: the foreclosure trap is present (optimum > greedy > ~random).")