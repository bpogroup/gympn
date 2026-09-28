"""BPM decision-type environments for the conference version of the paper.

Three decision types a process manager faces, each as an A-E PN with N
causally independent copies (own arrival stream, own queue, own resource
pool per copy, nothing shared), so the causal fan-out K tracks N exactly as
in `ncopies_env.py`. N=1 is the K=1 negative control; N=8 is the concurrent
setting the paper's claim is about.

  1. resource assignment      -> `multisite_env.make_multisite` (already run)
  2. next-activity selection  -> `make_next_activity(n)`   (this module)
  3. rework / quality gate    -> `make_rework(n)`          (this module)

Conventions shared with the rest of the suite:
  * outcomes are DETERMINISTIC in the case attributes. The simulator calls
    `behavior` and `reward_function` in separate invocations, so a coin flip
    inside either would be drawn twice and the two could disagree. All
    stochasticity comes from arrivals, attribute draws and service times.
  * one case per time unit per copy; service = base + U{0,1}.
  * reward 1 on a successful completion; the objective is throughput over
    a fixed horizon, so a slow activity costs resource time, not reward.
  * a warm start puts one case of each attribute value in every queue, so
    there is a genuine choice at t=0.

Next-activity selection (`make_next_activity`)
----------------------------------------------
A case with attribute risk in {0 (low), 1 (high)} waits. Two action
transitions (the busy token records which one fired as activity 0/1) compete for the same case token and the copy's single resource:
  approve_i     : base delay 1; pays 1 iff risk == 0 (approving a high-risk
                  case is a wrong outcome, reward 0)
  investigate_i : base delay 3; always pays 1
Per unit of resource time the right activity is approve for low risk (1 per
~1.5 time units) and investigate for high risk (1 per ~3.5 vs 0 per ~1.5).
The queue may also hold several cases, so the policy also picks WHICH case,
as in the assignment environments.

Rework / quality gate (`make_rework`)
-------------------------------------
A case with attribute risk in {0, 1} and flag reworked in {0, 1} waits. Two
action transitions compete for it and the copy's resource:
  ship_i  : base delay 1. If risk == 1 and reworked == 0 the defect surfaces
            and the case RETURNS to the queue as (risk 1, reworked 1) after a
            fix delay of 2, paying 0; otherwise it completes and pays 1.
  check_i : base delay 2; always completes and pays 1 (the check catches the
            defect and fixes it in place).
So skipping the check on a high-risk case costs ~1 + 2 + ~1 time units and
one extra decision before it pays, against ~2.5 for checking it directly.
The consequence of a bad decision arrives LATER, through the loop, which is
where temporal credit and provenance credit differ most.
"""
import random

from simpn.simulator import SimToken
from gympn.simulator import GymProblem


def _service(base):
    return base + random.randint(0, 1)


# --------------------------------------------------------------------------
# 2) next-activity selection
# --------------------------------------------------------------------------
def _arrive_risk(a):
    return [SimToken({'risk': random.randint(0, 1)}, delay=1),
            SimToken({'risk': random.randint(0, 1)})]


def _approve(c, r):
    return [SimToken((c, r, {'activity': 0}), delay=_service(1))]


def _investigate(c, r):
    return [SimToken((c, r, {'activity': 1}), delay=_service(3))]


def _na_ok(b):
    case, _res, act = b[0], b[1], b[2]['activity']
    return act == 1 or case['risk'] == 0


def _na_done(b):
    return [SimToken(b[1])]


def make_next_activity(n=4, causal_rl=False, allow_postpone=True,
                       causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    for i in range(n):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['risk'])
        waiting = ag.add_var(f"waiting_{i}", var_attributes=['risk'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['risk', 'code_employee', 'activity'])
        employee = ag.add_var(f"employee_{i}", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        arrival.put({'risk': 0})
        waiting.put({'risk': 0})
        waiting.put({'risk': 1})
        ag.add_event([arrival], [arrival, waiting], _arrive_risk, name=f'arrive_{i}')
        ag.add_action([waiting, employee], [busy], behavior=_approve, name=f"approve_{i}")
        ag.add_action([waiting, employee], [busy], behavior=_investigate, name=f"investigate_{i}")
        ag.add_event([busy], [employee], _na_done, name=f'done_{i}',
                     reward_function=lambda b: 1 if _na_ok(b) else 0)
    return ag


def next_activity_heuristic(observable_net, tokens_comb, bindings=None):
    """Attribute-matched activity: approve low-risk cases, investigate
    high-risk ones; prefer approving a low-risk case when both are queued
    (higher reward per resource time). Never postpones."""
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ['postpone']]
    if not real:
        return bindings[0]

    def parts(b):
        tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
        risk = None
        for (_place, tok) in b[0]:
            v = getattr(tok, 'value', tok)
            if isinstance(v, dict) and 'risk' in v:
                risk = v['risk']
        return tr, risk

    for want_tr, want_risk in (('approve', 0), ('investigate', 1)):
        for b in real:
            tr, risk = parts(b)
            if tr.startswith(want_tr) and risk == want_risk:
                return b
    # only mismatched bindings enabled (e.g. one high-risk case, nothing
    # else): investigate it rather than approve it wrongly
    for b in real:
        if parts(b)[0].startswith('investigate'):
            return b
    return real[0]


# --------------------------------------------------------------------------
# 3) rework / quality gate
# --------------------------------------------------------------------------
def _arrive_rework(a):
    return [SimToken({'risk': random.randint(0, 1), 'reworked': 0}, delay=1),
            SimToken({'risk': random.randint(0, 1), 'reworked': 0})]


def _ship(c, r):
    return [SimToken((c, r, {'activity': 0}), delay=_service(1))]


def _check(c, r):
    return [SimToken((c, r, {'activity': 1}), delay=_service(2))]


def _rw_defect(b):
    case, _res, act = b[0], b[1], b[2]['activity']
    return act == 0 and case['risk'] == 1 and case['reworked'] == 0


def _rw_done(b):
    case, res, _act = b[0], b[1], b[2]
    if _rw_defect(b):
        fixed = dict(case); fixed['reworked'] = 1
        return [SimToken(res), SimToken(fixed, delay=2)]   # resource back, case re-queued
    return [SimToken(res), None]


def make_rework(n=4, causal_rl=False, allow_postpone=True,
                causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    for i in range(n):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['risk', 'reworked'])
        waiting = ag.add_var(f"waiting_{i}", var_attributes=['risk', 'reworked'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['risk', 'reworked', 'code_employee', 'activity'])
        employee = ag.add_var(f"employee_{i}", var_attributes=['code_employee'])
        employee.put({'code_employee': 0})
        arrival.put({'risk': 0, 'reworked': 0})
        waiting.put({'risk': 0, 'reworked': 0})
        waiting.put({'risk': 1, 'reworked': 0})
        ag.add_event([arrival], [arrival, waiting], _arrive_rework, name=f'arrive_{i}')
        ag.add_action([waiting, employee], [busy], behavior=_ship, name=f"ship_{i}")
        ag.add_action([waiting, employee], [busy], behavior=_check, name=f"check_{i}")
        ag.add_event([busy], [employee, waiting], _rw_done, name=f'done_{i}',
                     reward_function=lambda b: 0 if _rw_defect(b) else 1)
    return ag


def rework_heuristic(observable_net, tokens_comb, bindings=None):
    """Check high-risk unreworked cases, ship everything else; prefer a ship
    of a safe case when both are queued. Never postpones."""
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ['postpone']]
    if not real:
        return bindings[0]

    def parts(b):
        tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
        risky = None
        for (_place, tok) in b[0]:
            v = getattr(tok, 'value', tok)
            if isinstance(v, dict) and 'risk' in v:
                risky = (v['risk'] == 1 and v.get('reworked', 0) == 0)
        return tr, risky

    for want_tr, want_risky in (('ship', False), ('check', True)):
        for b in real:
            tr, risky = parts(b)
            if tr.startswith(want_tr) and risky == want_risky:
                return b
    for b in real:
        if parts(b)[0].startswith('check'):
            return b
    return real[0]


BPM_BUILDERS = {
    "next_activity": make_next_activity,
    "rework": make_rework,
}
BPM_HEURISTICS = {
    "next_activity": next_activity_heuristic,
    "rework": rework_heuristic,
}


if __name__ == "__main__":
    # smoke: anchors (random vs heuristic) and lineage independence per N
    import os, sys, types, uuid
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    import numpy as np
    from gympn.environment import AEPN_Env
    from gympn.solvers import RandomSolver, HeuristicSolver

    LENGTH = 20

    def anchors(name, n, episodes=20):
        rnd, heu = [], []
        for s in range(episodes):
            random.seed(1000 + s); np.random.seed(1000 + s)
            env = BPM_BUILDERS[name](n, causal_rl=False, allow_postpone=False)
            rnd.append(float(env.testing_run(solver=RandomSolver(), length=LENGTH)))
            random.seed(1000 + s); np.random.seed(1000 + s)
            env = BPM_BUILDERS[name](n, causal_rl=False, allow_postpone=False)
            heu.append(float(env.testing_run(solver=HeuristicSolver(BPM_HEURISTICS[name]),
                                             length=LENGTH)))
        return np.mean(rnd), np.std(rnd), np.mean(heu), np.std(heu)

    def trace_check(name, n):
        random.seed(0); np.random.seed(0)
        pn = BPM_BUILDERS[name](n, causal_rl=True, allow_postpone=False)
        pn.length = LENGTH
        for p in pn.places:
            for t in p.marking:
                setattr(t, '_id', str(uuid.uuid4()))
        pn.causal_trace.flush()
        sen = types.SimpleNamespace(_id="__initial__")
        toks = [t for p in pn.places for t in p.marking]
        for t in toks:
            pn.causal_trace.register_token(t, sen, [], time=0)
        pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
        env = AEPN_Env(pn); env.reset()
        for _ in range(60):
            k = len(env.pn.pn_actions)
            if k == 0:
                break
            _, _, d, _, _ = env.step(np.random.randint(k))
            if d:
                break
        ct = env.pn.causal_trace
        ccf = np.array(ct.redistribute_rewards(scheme='ccf', beta=0.5))
        mcq = np.array(ct.redistribute_rewards(scheme='mc_q', beta=0.5))
        sc = max(1e-9, float(np.mean(np.abs(mcq))))
        return np.mean(np.abs(ccf - mcq)) / sc

    for name in BPM_BUILDERS:
        print(f"== {name}")
        for n in (1, 4, 8):
            r, rs, h, hs = anchors(name, n)
            gap = h - r
            print(f"  N={n}: random={r:6.2f} +-{rs:4.2f}  heuristic={h:6.2f} +-{hs:4.2f}  "
                  f"headroom={gap:5.2f} ({gap/max(rs,1e-9):.1f}x sigma_rnd)  "
                  f"factoring |ccf-mcq|/scale={trace_check(name, n):.3f}")
