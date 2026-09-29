"""BPM decision-type environments for the conference version of the paper
(HARD variants, 2026-09-28).

Three decision types a process manager faces, each as an A-E PN with N
causally independent copies (own arrival stream, own queue, own resource
pool per copy, nothing shared), so the causal fan-out K tracks N exactly as
in `ncopies_env.py`. N=1 is the K=1 negative control; N=8 is the concurrent
setting the paper's claim is about.

  1. resource assignment      -> `multisite_env.make_multisite` (already run)
  2. next-activity selection  -> `make_next_activity(n)`   (this module)
  3. rework / quality gate    -> `make_rework(n)`          (this module)

WHY "HARD". The first design (one resource, two risk levels, deterministic
outcomes) was degenerate at N=1: every seed of every arm converged to the
same greedy lookup table within nine epochs and the 20 seeds gave identical
finals (PPO 0.944 x20, cgae-cf 0.944 x20, mc-q 0.813 x20), so the K=1
control had zero variance. This version, modelled on `make_n_copies_hard`:

  * THREE risk levels, risk in {0, 1, 2}, with a HIDDEN per-case outcome
    `bad` drawn at arrival with P(bad | risk) = 0.1 / 0.5 / 0.9. The policy
    sees `risk`, never `bad`; `bad` is stripped from observations with
    `set_unobservable`, and it decides the outcome deterministically once
    drawn, so `behavior` and `reward_function` (which the simulator calls
    in separate invocations) always agree.
  * TWO heterogeneous resources per copy, skill in {0 (normal), 1 (expert)}.
    The slow, safe activity is much faster for the expert.
  * The right activity therefore depends on BOTH the case's risk and the
    resource's skill (a 3 x 2 table with four distinct rules), and the policy
    also chooses WHICH queued case and WHICH resource, as in the assignment
    environments.

Rates (expected reward per expected resource time; service = base + U{0,1}):
                     risk 0   risk 1   risk 2
  fast activity      0.60     0.33     0.07     (base 1, pays iff not bad)
  safe, expert       0.40     0.40     0.40     (base 2, always pays)
  safe, normal       0.22     0.22     0.22     (base 4, always pays)
so the expert does the fast activity only on risk-0 cases and the safe one
otherwise, while the normal resource does the fast activity on risk-0 AND
risk-1 cases and the safe one only on risk-2 cases.

Next-activity selection (`make_next_activity`): the fast activity is
`approve` and the safe one `investigate`; a wrong approval simply pays 0
(the case leaves with a bad outcome).

Rework / quality gate (`make_rework`): the fast activity is `ship` and the
safe one `check`; a shipped bad case pays 0 and RETURNS to the queue as
(risk, reworked=1, fixed) after a fix delay of 2, so the cost of the wrong
decision arrives later, through the loop, and consumes future resource time
and a further decision. Same decision table, different way the penalty
arrives -- which is exactly what separates temporal from provenance credit.

Shared conventions: one case per time unit per copy, reward 1 per good
completion (throughput of good outcomes over a fixed horizon), warm start
with one case of each risk level in every queue.
"""
import random

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

P_BAD = {0: 0.1, 1: 0.5, 2: 0.9}
FAST_BASE = 1
SAFE_BASE = {0: 4, 1: 2}      # by resource skill
FIX_DELAY = 2                 # rework loop (quality-gate env only)


def _service(base):
    return base + random.randint(0, 1)


def _draw_case(reworked=None):
    risk = random.randint(0, 2)
    bad = 1 if random.random() < P_BAD[risk] else 0
    c = {'risk': risk, 'bad': bad}
    if reworked is not None:
        c['reworked'] = reworked
    return c


def _visible(c):
    return {k: v for k, v in c.items() if k != 'bad'}


def _rate(activity, risk, skill):
    """Myopic expected reward per expected resource time; the heuristic's
    decision statistic. Knows the risk model, not the hidden draw."""
    if activity == 0:   # fast
        return (1.0 - P_BAD[risk]) / (FAST_BASE + 0.5)
    return 1.0 / (SAFE_BASE[skill] + 0.5)


# --------------------------------------------------------------------------
# 2) next-activity selection
# --------------------------------------------------------------------------
def _arrive_na(a):
    return [SimToken(_draw_case(), delay=1), SimToken(_draw_case())]


def _approve(c, r):
    ok = 0 if c['bad'] else 1
    return [SimToken((_visible(c), r, {'activity': 0, 'ok': ok}), delay=_service(FAST_BASE))]


def _investigate(c, r):
    return [SimToken((_visible(c), r, {'activity': 1, 'ok': 1}),
                     delay=_service(SAFE_BASE[r['skill']]))]


def _na_done(b):
    return [SimToken(b[1])]


def make_next_activity(n=4, causal_rl=False, allow_postpone=True,
                       causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    hidden = {}
    for i in range(n):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['risk', 'bad'])
        waiting = ag.add_var(f"waiting_{i}", var_attributes=['risk', 'bad'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['risk', 'skill', 'activity', 'ok'])
        employee = ag.add_var(f"employee_{i}", var_attributes=['skill'])
        employee.put({'skill': 0})
        employee.put({'skill': 1})
        arrival.put(_draw_case())
        for risk in (0, 1, 2):
            waiting.put({'risk': risk, 'bad': 1 if random.random() < P_BAD[risk] else 0})
        hidden[f"arrival_{i}"] = ['bad']
        hidden[f"waiting_{i}"] = ['bad']
        ag.add_event([arrival], [arrival, waiting], _arrive_na, name=f'arrive_{i}')
        ag.add_action([waiting, employee], [busy], behavior=_approve, name=f"approve_{i}")
        ag.add_action([waiting, employee], [busy], behavior=_investigate, name=f"investigate_{i}")
        ag.add_event([busy], [employee], _na_done, name=f'done_{i}',
                     reward_function=lambda b: b[2]['ok'])
    ag.set_unobservable(token_attrs=hidden)
    return ag


# --------------------------------------------------------------------------
# 3) rework / quality gate
# --------------------------------------------------------------------------
def _arrive_rw(a):
    return [SimToken(_draw_case(reworked=0), delay=1), SimToken(_draw_case(reworked=0))]


def _ship(c, r):
    ok = 0 if (c['bad'] and not c['reworked']) else 1
    return [SimToken((_visible(c), r, {'activity': 0, 'ok': ok}), delay=_service(FAST_BASE))]


def _check(c, r):
    return [SimToken((_visible(c), r, {'activity': 1, 'ok': 1}),
                     delay=_service(SAFE_BASE[r['skill']]))]


def _rw_done(b):
    case, res, tag = b[0], b[1], b[2]
    if tag['ok']:
        return [SimToken(res), None]
    fixed = dict(case); fixed['reworked'] = 1; fixed['bad'] = 0
    return [SimToken(res), SimToken(fixed, delay=FIX_DELAY)]   # resource back, case re-queued


def make_rework(n=4, causal_rl=False, allow_postpone=True,
                causal_postpone_tokenflow=False):
    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)
    hidden = {}
    for i in range(n):
        arrival = ag.add_var(f"arrival_{i}", var_attributes=['risk', 'bad', 'reworked'])
        waiting = ag.add_var(f"waiting_{i}", var_attributes=['risk', 'bad', 'reworked'])
        busy = ag.add_var(f"busy_{i}", var_attributes=['risk', 'reworked', 'skill', 'activity', 'ok'])
        employee = ag.add_var(f"employee_{i}", var_attributes=['skill'])
        employee.put({'skill': 0})
        employee.put({'skill': 1})
        arrival.put(_draw_case(reworked=0))
        for risk in (0, 1, 2):
            waiting.put({'risk': risk, 'bad': 1 if random.random() < P_BAD[risk] else 0,
                         'reworked': 0})
        hidden[f"arrival_{i}"] = ['bad']
        hidden[f"waiting_{i}"] = ['bad']
        ag.add_event([arrival], [arrival, waiting], _arrive_rw, name=f'arrive_{i}')
        ag.add_action([waiting, employee], [busy], behavior=_ship, name=f"ship_{i}")
        ag.add_action([waiting, employee], [busy], behavior=_check, name=f"check_{i}")
        ag.add_event([busy], [employee, waiting], _rw_done, name=f'done_{i}',
                     reward_function=lambda b: b[2]['ok'])
    ag.set_unobservable(token_attrs=hidden)
    return ag


# --------------------------------------------------------------------------
# heuristic anchor: myopic rate rule over the enabled bindings
# --------------------------------------------------------------------------
def _make_rate_heuristic(fast_prefix):
    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None
        real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
                and b[0] != ['postpone']]
        if not real:
            return bindings[0]
        best, best_rate = None, -1.0
        for b in real:
            tr = str(getattr(b[2], '_id', None) or getattr(b[2], 'name', ''))
            risk = skill = None; reworked = 0
            for (_place, tok) in b[0]:
                v = getattr(tok, 'value', tok)
                if isinstance(v, dict):
                    if 'risk' in v:
                        risk = v['risk']; reworked = v.get('reworked', 0)
                    if 'skill' in v:
                        skill = v['skill']
            if risk is None or skill is None:
                continue
            act = 0 if tr.startswith(fast_prefix) else 1
            eff_risk = 0 if (act == 0 and reworked) else risk   # a reworked case is fixed
            rate = _rate(act, eff_risk, skill)
            if rate > best_rate:
                best, best_rate = b, rate
        return best if best is not None else real[0]
    return heuristic


next_activity_heuristic = _make_rate_heuristic('approve')
rework_heuristic = _make_rate_heuristic('ship')

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
        obs_widths = None
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

    def hidden_check(name):
        """The hidden attribute must not reach the observation graph."""
        pn = BPM_BUILDERS[name](1, causal_rl=False, allow_postpone=False)
        pn.length = LENGTH
        env = AEPN_Env(pn); obs = env.reset()
        g = obs['graph'] if isinstance(obs, dict) and 'graph' in obs else obs
        w = {nt: int(g[nt].x.size(1)) for nt in g.node_types if hasattr(g[nt], 'x') and g[nt].x.dim() == 2}
        return {k: v for k, v in w.items() if k.startswith(('waiting', 'arrival', 'busy'))}

    for name in BPM_BUILDERS:
        print(f"== {name}   observed widths: {hidden_check(name)}")
        for n in (1, 4, 8):
            r, rs, h, hs = anchors(name, n)
            gap = h - r
            print(f"  N={n}: random={r:6.2f} +-{rs:4.2f}  heuristic={h:6.2f} +-{hs:4.2f}  "
                  f"headroom={gap:5.2f} ({gap/max(rs,1e-9):.1f}x sigma_rnd)  "
                  f"factoring |ccf-mcq|/scale={trace_check(name, n):.3f}")
