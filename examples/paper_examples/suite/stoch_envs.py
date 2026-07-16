"""E5: the stochastic / scaled environment tier (PAPER_PLAN_LRQ.md).

Why this tier exists: the deterministic a-h suite is SATURATED for LRQ
(1.00 +/- 0.00 across 80 cells) — it can no longer discriminate, and its fixed
arrival streams understate both the credit-assignment problem and LRQ's
variance-reduction claim. These envs add (i) stochastic arrivals, task types
and service times, (ii) more entities and longer horizons, (iii) a realistic
embedding of the chain-vs-shortcut motif under load.

Conventions shared with envs.py so the observation/heuristic machinery
carries over: case tokens carry 'task_type', resources 'code_employee';
rewards fire on completion events. Behaviors draw randomness at firing time
(the simulator calls each behavior exactly once per firing), seeded per cell
by run_suite._set_seed, so runs are reproducible per seed and genuinely
stochastic across episodes.

The heuristic anchors here are STRONG MYOPIC references, not optima —
normalized scores above 1.0 are possible and meaningful (the policy beat the
myopic rule). Random remains the floor.

Suggested horizons (threaded via SuiteConfig.env_length):
  s1_stoch_sequence 20 · s2_stoch_scaled 30 · s3_stoch_mixed 25
"""
import random

from simpn.simulator import SimToken

from gympn.simulator import GymProblem


# --------------------------------------------------------------------------
# s1) stochastic two-stage sequence, joint pool
# --------------------------------------------------------------------------
def make_s1_stoch_sequence(causal_rl=False, allow_postpone=True,
                           causal_postpone_tokenflow=False):
    """Two-stage flow, one shared 3-employee pool, random task types,
    U{1,2} interarrivals, service = base(match 1 / cross 2 / generalist 3)
    + U{0,1} noise. Reward 1 per completed case."""
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting1 = ag.add_var("waiting1", var_attributes=['task_type'])
    busy1 = ag.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = ag.add_var("waiting2", var_attributes=['task_type'])
    busy2 = ag.add_var("busy2", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    pool = ag.add_var("employee", var_attributes=['code_employee'])
    for eid in range(3):
        pool.put({'code_employee': eid})
    # warm start: one case of each type already queued
    waiting1.put({'task_type': 0})
    waiting1.put({'task_type': 1})

    def arrive(a):
        # One case per time unit (keeps the pool saturated so assignment
        # quality matters); the type and service durations stay stochastic.
        return [SimToken({'task_type': random.randint(0, 1)}, delay=1),
                SimToken({'task_type': random.randint(0, 1)})]

    ag.add_event([arrival], [arrival, waiting1], arrive, name='arrive')

    def start(c, r):
        if c['task_type'] == r['code_employee']:
            base = 1
        elif r['code_employee'] < 2:
            base = 2
        else:
            base = 3  # generalist
        return [SimToken((c, r), delay=base + random.randint(0, 1))]

    ag.add_action([waiting1, pool], [busy1], behavior=start, name="start1")
    ag.add_event([busy1], [pool, waiting2],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done1')
    ag.add_action([waiting2, pool], [busy2], behavior=start, name="start2")
    ag.add_event([busy2], [pool], lambda b: [SimToken(b[-1])], name='done2',
                 reward_function=lambda x: 1)
    return ag


# --------------------------------------------------------------------------
# s2) scaled single-stage assignment: 3 task types, 6 heterogeneous employees
# --------------------------------------------------------------------------
def make_s2_stoch_scaled(causal_rl=False, allow_postpone=True,
                         causal_postpone_tokenflow=False):
    """One assignment decision at scale: 3 task types arriving U{1,2}, a
    6-employee pool where employee e is fast for type e%3 (1+U{0,1}) and slow
    otherwise (2+U{0,1}). Reward 1 per completion. Big enabled-binding sets
    stress the variable-action machinery."""
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waiting = ag.add_var("waiting", var_attributes=['task_type'])
    busy = ag.add_var("busy", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    pool = ag.add_var("employee", var_attributes=['code_employee'])
    for eid in range(6):
        pool.put({'code_employee': eid})
    for t in range(3):
        waiting.put({'task_type': t})

    # Three cases per time unit: demand ~= matched capacity (6 employees at
    # matched service ~1.5 => ~4/time; mismatched ~2.5 => ~2.4/time), so the
    # backlog — and the return — hinges on assignment quality.
    q2 = ag.add_var("spawn2", var_attributes=['task_type'])
    q3 = ag.add_var("spawn3", var_attributes=['task_type'])

    def arrive(a):
        return [SimToken({'task_type': random.randint(0, 2)}, delay=1),
                SimToken({'task_type': random.randint(0, 2)}),
                SimToken({'task_type': random.randint(0, 2)}),
                SimToken({'task_type': random.randint(0, 2)})]

    ag.add_event([arrival], [arrival, waiting, q2, q3], arrive, name='arrive')
    ag.add_event([q2], [waiting], lambda c: [SimToken(c)], name='spawn2_move')
    ag.add_event([q3], [waiting], lambda c: [SimToken(c)], name='spawn3_move')

    def start(c, r):
        base = 1 if (r['code_employee'] % 3) == c['task_type'] else 2
        return [SimToken((c, r), delay=base + random.randint(0, 1))]

    ag.add_action([waiting, pool], [busy], behavior=start, name="assign")
    ag.add_event([busy], [pool], lambda b: [SimToken(b[-1])], name='complete',
                 reward_function=lambda x: 1)
    return ag


# --------------------------------------------------------------------------
# s3) mixed chain lengths under load: the E1 motif, stochastic and realistic
# --------------------------------------------------------------------------
R_LONG, R_SHORT = 8.0, 2.0


def make_s3_stoch_mixed(causal_rl=False, allow_postpone=True,
                        causal_postpone_tokenflow=False):
    """Long cases (2 stages, reward 8) and short cases (1 stage, reward 2)
    arrive mixed (50/50, one per time unit) and compete for one homogeneous
    2-employee pool; service 1+U{0,1}, oversaturated. Per-slot economics:
    completing a long chain (8 over ~3 slots ~ 2.7/slot) clearly beats a diet
    of shorts (2 over ~1.5 slots ~ 1.3/slot) — the chain-vs-shortcut pressure
    of E1 embedded in a stochastic queueing system."""
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    arrival = ag.add_var("arrival", var_attributes=['task_type'])
    waitL1 = ag.add_var("waitL1", var_attributes=['task_type'])
    busyL1 = ag.add_var("busyL1", var_attributes=['task_type', 'code_employee'])
    waitL2 = ag.add_var("waitL2", var_attributes=['task_type'])
    busyL2 = ag.add_var("busyL2", var_attributes=['task_type', 'code_employee'])
    waitS = ag.add_var("waitS", var_attributes=['task_type'])
    busyS = ag.add_var("busyS", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    pool = ag.add_var("employee", var_attributes=['code_employee'])
    for eid in range(2):
        pool.put({'code_employee': eid})
    waitL1.put({'task_type': 0})
    waitS.put({'task_type': 1})

    # Arrivals land in 'incoming'; a routing event splits them by class
    # (task_type 0 = long -> waitL1, 1 = short -> waitS).
    incoming = ag.add_var("incoming", var_attributes=['task_type'])

    def arrive(a):
        # One mixed case per time unit: with 2 employees the system runs
        # oversaturated (~37 service-slots of demand vs ~33 capacity at
        # horizon 25), so prioritization — not raw capacity — sets the return.
        return [SimToken({'task_type': random.randint(0, 1)}, delay=1),
                SimToken({'task_type': random.randint(0, 1)})]

    ag.add_event([arrival], [arrival, incoming], arrive, name='arrive')

    def route(c):
        if c['task_type'] == 0:
            return [SimToken(c), None]
        return [None, SimToken(c)]

    ag.add_event([incoming], [waitL1, waitS], route, name='route')

    def start(c, r):
        return [SimToken((c, r), delay=1 + random.randint(0, 1))]

    ag.add_action([waitL1, pool], [busyL1], behavior=start, name="start_L1")
    ag.add_event([busyL1], [pool, waitL2],
                 lambda b: [SimToken(b[-1]), SimToken(b[0])], name='done_L1')
    ag.add_action([waitL2, pool], [busyL2], behavior=start, name="start_L2")
    ag.add_event([busyL2], [pool], lambda b: [SimToken(b[-1])], name='done_L2',
                 reward_function=lambda x: R_LONG)
    ag.add_action([waitS, pool], [busyS], behavior=start, name="start_S")
    ag.add_event([busyS], [pool], lambda b: [SimToken(b[-1])], name='done_S',
                 reward_function=lambda x: R_SHORT)
    return ag


# --------------------------------------------------------------------------
# Tier heuristics: strong myopic anchors (NOT optima; normalized > 1 is
# possible and means "the policy beat the myopic rule").
# --------------------------------------------------------------------------
def _tier_heuristic(stage_order, n_types):
    """Priority over transition names (finish work-in-progress first), and
    within a transition prefer a type-matched employee (code % n_types ==
    task_type). Never postpones."""
    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None

        def parts(b):
            try:
                tr = getattr(b[2], '_id', None) or getattr(b[2], 'name', None)
                case = b[0][0][1].value
                res = b[0][1][1].value if len(b[0]) > 1 else None
                return str(tr), case, res
            except Exception:
                return None, None, None

        for want in stage_order:
            best = None
            for b in bindings:
                tr, case, res = parts(b)
                if tr is None or not tr.startswith(want):
                    continue
                matched = (res is not None and case is not None and
                           res.get('code_employee', -1) % n_types ==
                           case.get('task_type', -2))
                if matched:
                    return b
                if best is None:
                    best = b
            if best is not None:
                return best
        for b in bindings:
            if parts(b)[0] is not None:
                return b
        return None
    return heuristic


s1_heuristic = _tier_heuristic(["start2", "start1"], n_types=2)
s2_heuristic = _tier_heuristic(["assign"], n_types=3)
# s3: employees homogeneous; the priority IS the policy. Chains dominate
# per-slot (8/~3 vs 2/~1.5), so invest in chain work FIRST and let shorts
# fill leftover capacity — an L2>S>L1 order would never start new chains
# under saturation (the short queue is never empty) and degenerates into an
# all-shorts policy.
s3_heuristic = _tier_heuristic(["start_L2", "start_L1", "start_S"], n_types=1)


STOCH_BUILDERS = {
    "s1_stoch_sequence": make_s1_stoch_sequence,
    "s2_stoch_scaled": make_s2_stoch_scaled,
    "s3_stoch_mixed": make_s3_stoch_mixed,
}

STOCH_HEURISTICS = {
    "s1_stoch_sequence": s1_heuristic,
    "s2_stoch_scaled": s2_heuristic,
    "s3_stoch_mixed": s3_heuristic,
}
