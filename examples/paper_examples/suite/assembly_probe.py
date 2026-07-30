r"""Estimator-level probe: does lrq's bias (vs ccf) ever flip the policy?

Minimal assembly/join net -- the ONE Petri-net motif where ccf and lrq can
diverge (a decision shares a component with a reward via a downstream join, yet
is not that reward's token-ancestor):

    part1 --[d1: route_asm]--> jw1 --.
                                     +--[join]--> reward r_join (+10)
    part2 --[split2]--> jw2 ---------'
                    \--> p2priv --[priv2]--> reward r2 (+3)   (private to stream 2)
    part1 --[d1: route_sa]--> sa1 --[sa_done]--> reward r1 (+1)   (d1's standalone)

d1 chooses route_asm (A) or route_sa (B). Under A the join fires (r_join), and
d1 shares a component with stream 2 via r_join -- so ccf ALSO credits d1 with
stream 2's private r2, which lrq does not (d1 is not r2's ancestor). r2 fires
regardless of d1's choice (action-independent), so this tests whether ccf's
action-dependent component membership miscalibrates d1's credit, and whether
lrq's omission does.

We compute the redistributed credit for the d1 decision under A vs B for each of
{mc_q, lrq, ccf}, and compare the credit difference (A-B) against the TRUE return
difference. A scheme whose (A-B) has the wrong sign would flip the policy.
"""
import sys, os, types, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import numpy as np
import gympn
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.environment import AEPN_Env


def make_assembly(causal_rl=True, r_join=10.0, r1=1.0, r2=3.0):
    ag = GymProblem(allow_postpone=False, causal_rl=causal_rl)
    part1 = ag.add_var("part1", var_attributes=['id'])
    part2 = ag.add_var("part2", var_attributes=['id'])
    jw1 = ag.add_var("jw1", var_attributes=['id'])
    jw2 = ag.add_var("jw2", var_attributes=['id'])
    p2priv = ag.add_var("p2priv", var_attributes=['id'])
    sa1 = ag.add_var("sa1", var_attributes=['id'])
    done = ag.add_var("done", var_attributes=['id'])
    part1.put({'id': 1})
    part2.put({'id': 2})

    # stream 2 has its OWN decision d2 (an action) that routes part2 to the join
    # AND spawns its private-reward branch -> r2 now has an action ancestor (d2),
    # so d1 and d2 can share a component via the join while r2 stays private to d2.
    ag.add_action([part2], [jw2, p2priv],
                  behavior=lambda p: [SimToken(p, delay=1), SimToken(p)], name='route2')
    # stream 2's PRIVATE reward (ancestor = d2; fires regardless of d1's choice)
    ag.add_event([p2priv], [done], lambda p: [SimToken(p)],
                 name='priv2', reward_function=lambda b: r2)

    # d1: route part1 to assembly (A) or to standalone (B)  -- the two actions
    ag.add_action([part1], [jw1], behavior=lambda c: [SimToken(c, delay=1)], name='route_asm')
    ag.add_action([part1], [sa1], behavior=lambda c: [SimToken(c, delay=1)], name='route_sa')

    # d1's standalone reward
    ag.add_event([sa1], [done], lambda p: [SimToken(p)],
                 name='sa_done', reward_function=lambda b: r1)
    # join: needs BOTH parts -> big shared reward
    ag.add_event([jw1, jw2], [done], lambda a, b: [SimToken(a)],
                 name='join', reward_function=lambda a, b: r_join)
    return ag


def make_shared_r(causal_rl=True, r_join=5.0, r1=5.0, r2=10.0):
    """Join coupled through a SHARED RESOURCE R, so stream 2's 'private' reward r2
    genuinely DEPENDS on d1 (its timing shifts with whether d1 grabbed R first).
    Tests whether lrq stays unbiased when the coupling is real (my claim: yes,
    because lrq follows the resource token; the probe decides)."""
    ag = GymProblem(allow_postpone=False, causal_rl=causal_rl)
    for nm in ("part1", "part2", "R", "busy1", "jw1", "jw2", "p2waitR", "busypriv", "sa1", "done"):
        ag.add_var(nm, var_attributes=['id'])
    P = {p._id: p for p in ag.places}
    P['part1'].put({'id': 1}); P['part2'].put({'id': 2}); P['R'].put({'id': 9})

    # d2: route part2 -> jw2 (to join) + p2waitR (private branch, needs R)
    ag.add_action([P['part2']], [P['jw2'], P['p2waitR']],
                  behavior=lambda p: [SimToken(p, delay=1), SimToken(p)], name='route2')
    # d1: use_R (consume R now, delaying stream 2's private branch) OR standalone
    ag.add_action([P['part1'], P['R']], [P['busy1']],
                  behavior=lambda c, r: [SimToken((c, r), delay=3)], name='use_R')
    ag.add_action([P['part1']], [P['sa1']],
                  behavior=lambda c: [SimToken(c, delay=1)], name='standalone')
    ag.add_event([P['busy1']], [P['jw1'], P['R']],
                 lambda b: [SimToken(b[0]), SimToken(b[1])], name='done1')  # frees R
    ag.add_event([P['sa1']], [P['done']], lambda p: [SimToken(p)],
                 name='sa_done', reward_function=lambda b: r1)
    ag.add_event([P['jw1'], P['jw2']], [P['done']], lambda a, b: [SimToken(a)],
                 name='join', reward_function=lambda a, b: r_join)
    # stream 2's private reward: needs R (auto-fires when R is free) -> timing depends on d1
    ag.add_event([P['p2waitR'], P['R']], [P['R'], P['done']],
                 lambda a, b: [SimToken(b), SimToken(a)],
                 name='priv', reward_function=lambda a, b: r2)
    return ag


def _static_component_reward_types(pn):
    """For each action-type, the set of reward-transition-types in its STATIC
    (action-invariant) component: actions are unioned when they can statically
    reach a common reward transition (topological reachability, not the realized
    trajectory). This is the fix candidate -- membership can't depend on the action."""
    trans = list(pn.actions) + list(pn.events)
    consumers = {}
    for t in trans:
        for p in t.incoming:
            consumers.setdefault(p._id, []).append(t)
    reward_types = set(pn.reward_functions.keys())

    def reaches(t):
        reached, seen_t, seen_p, stack = set(), set(), set(), list(t.outgoing)
        if t._id in reward_types:
            reached.add(t._id)
        while stack:
            p = stack.pop()
            if p._id in seen_p:
                continue
            seen_p.add(p._id)
            for ct in consumers.get(p._id, []):
                if ct._id in reward_types:
                    reached.add(ct._id)
                if ct._id not in seen_t:
                    seen_t.add(ct._id); stack.extend(ct.outgoing)
        return reached

    reach_by_action = {a._id: reaches(a) for a in pn.actions}

    # Group competing actions into ONE decision point (they share an input place),
    # so a decision's component is the union of ALL its options' reach -> the
    # component can't change with which action the decision picks (the crux).
    ap = {a._id: a._id for a in pn.actions}
    def af(x):
        while ap[x] != x:
            ap[x] = ap[ap[x]]; x = ap[x]
        return x
    place_to_actions = {}
    for a in pn.actions:
        for p in a.incoming:
            place_to_actions.setdefault(p._id, []).append(a._id)
    for aids in place_to_actions.values():
        for k in range(1, len(aids)):
            ap[af(aids[0])] = af(aids[k])
    dp_reach = {}
    for aid, rr in reach_by_action.items():
        dp_reach.setdefault(af(aid), set()).update(rr)

    # Union decision points that can reach a common reward -> static components.
    dps = list(dp_reach)
    dp = {d: d for d in dps}
    def df(x):
        while dp[x] != x:
            dp[x] = dp[dp[x]]; x = dp[x]
        return x
    for i in range(len(dps)):
        for j in range(i + 1, len(dps)):
            if dp_reach[dps[i]] & dp_reach[dps[j]]:
                dp[df(dps[i])] = df(dps[j])
    comp = {}
    for d in dps:
        comp.setdefault(df(d), set()).update(dp_reach[d])
    return {aid: comp[df(af(aid))] for aid in reach_by_action}


def _structural_credit(pn, beta):
    """s_ccf: credit each decision with the realized rewards of its STATIC
    component, at/after its decision time (discounted). Aligned with the
    action-decision firing order."""
    import math
    comp_rw = _static_component_reward_types(pn)
    base = pn._get_string_before_last_dot
    th = pn.causal_trace.transition_history.transitions
    rewards = [(tr['reward'], tr['time'], base(tr['transition']._id))
               for tr in th if tr.get('reward', 0.0)]
    out = []
    for tr in th:
        if not tr['is_action']:
            continue
        a_type = base(tr['transition']._id)
        u = tr['time']
        allowed = comp_rw.get(a_type, set())
        c = 0.0
        for (rv, rt, rtype) in rewards:
            if rtype in allowed and rt is not None and u is not None and rt >= u:
                c += rv * math.exp(-beta * max(0.0, rt - u))
        out.append(c)
    return np.array(out)


def make_independent(causal_rl=True, ra=5.0, rb=7.0):
    """Two structurally INDEPENDENT decision-streams (no join, no shared resource).
    s_ccf must FACTOR here (credit each decision only its own reward), like lrq/ccf,
    i.e. strictly less than mc_q's full return -> variance reduction is preserved."""
    ag = GymProblem(allow_postpone=False, causal_rl=causal_rl)
    for nm in ("a_in", "a_out", "b_in", "b_out", "done"):
        ag.add_var(nm, var_attributes=['id'])
    P = {p._id: p for p in ag.places}
    P['a_in'].put({'id': 1}); P['b_in'].put({'id': 2})
    ag.add_action([P['a_in']], [P['a_out']], behavior=lambda c: [SimToken(c, delay=1)], name='da')
    ag.add_action([P['b_in']], [P['b_out']], behavior=lambda c: [SimToken(c, delay=1)], name='db')
    ag.add_event([P['a_out']], [P['done']], lambda p: [SimToken(p)], name='ea', reward_function=lambda b: ra)
    ag.add_event([P['b_out']], [P['done']], lambda p: [SimToken(p)], name='eb', reward_function=lambda b: rb)
    return ag


def run_forced(choice_name, make_fn=make_assembly, beta=0.0, length=12, **rw):
    """Run one trajectory forcing d1 = choice_name; return (return, credits, forced)."""
    gympn.seed_everything(0)
    pn = make_fn(causal_rl=True, **rw)
    pn.length = length
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
    total = 0.0
    forced = False
    for _ in range(30):
        acts = env.pn.pn_actions
        if not acts:
            break
        idx = 0
        # pick the binding whose transition is choice_name, if present
        for i, a in enumerate(acts):
            tname = getattr(a[2], '_id', getattr(a[2], 'name', '')) if isinstance(a, tuple) and len(a) > 2 and a[2] is not None else ''
            if choice_name in str(tname):
                idx = i; forced = True; break
        _, r, done, _, _ = env.step(idx)
        total += float(r)
        if done:
            break
    ct = env.pn.causal_trace
    creds = {s: np.array(ct.redistribute_rewards(scheme=s, beta=beta)) for s in ('mc_q', 'lrq', 'ccf')}
    creds['s_ccf'] = _structural_credit(env.pn, beta)          # probe reference impl
    ct._pn = env.pn; ct._static_comp_cache = None              # library impl
    creds['lib_s_ccf'] = np.array(ct.redistribute_rewards(scheme='s_ccf', beta=beta))
    return total, creds, forced


def probe(name, make_fn, actionA, actionB, beta=0.0, **rw):
    """On a DETERMINISTIC net, per-trajectory credit == its expectation, and
    mc_q's d1 (A-B) difference is the UNBIASED action effect Q(s,A)-Q(s,B).
    A scheme is unbiased iff its d1 (A-B) equals mc_q's; a sign disagreement is a
    policy flip. d1 is the first decision (index 0)."""
    tA, cA, fA = run_forced(actionA, make_fn, beta, **rw)
    tB, cB, fB = run_forced(actionB, make_fn, beta, **rw)
    assert fA and fB, f"could not force actions in {name}"
    ref = float(cA['mc_q'][0] - cB['mc_q'][0])       # unbiased reference
    opt = 'A' if ref > 1e-9 else ('B' if ref < -1e-9 else '=')
    print(f"\n=== {name}   (beta={beta}, {rw}) ===")
    print(f"    mc_q d1(A-B) = {ref:+.3f}  [UNBIASED ref]  -> optimal action = {opt}")
    for s in ('lrq', 'ccf', 's_ccf', 'lib_s_ccf'):
        diff = float(cA[s][0] - cB[s][0])
        flip = abs(ref) > 1e-9 and (diff > 0) != (ref > 0)
        biased = abs(diff - ref) > 1e-6
        tag = "FLIP -> suboptimal" if flip else ("BIASED (magnitude)" if biased else "unbiased OK")
        print(f"    {s:>9}: d1(A-B) = {diff:+.3f}   [{tag}]")


if __name__ == "__main__":
    print("Assembly/join probes: is lrq unbiased where ccf is not? (mc_q = unbiased ref)")

    # Motif 1: bare join, private reward r2 action-INDEPENDENT of d1.
    probe("M1 join, A-optimal", make_assembly, 'route_asm', 'route_sa',
          beta=0.0, r_join=10, r1=1, r2=3)
    probe("M1 join, B-optimal (flip test)", make_assembly, 'route_asm', 'route_sa',
          beta=0.0, r_join=1, r1=5, r2=10)

    # Motif 2: join coupled via a SHARED RESOURCE, so r2 genuinely depends on d1.
    # The domination-breaker: does lrq stay unbiased when the coupling is real?
    for b in (0.0, 0.3):
        probe("M2 shared-R join", make_shared_r, 'use_R', 'standalone',
              beta=b, r_join=5, r1=5, r2=10)

    # Motif 3: structurally INDEPENDENT streams -> s_ccf must still FACTOR.
    print("\n=== M3 independent streams: does s_ccf still factor (credit vectors)? ===")
    _, c, _ = run_forced('da', make_independent, beta=0.0, ra=5, rb=7)
    for s in ('mc_q', 'lrq', 'ccf', 's_ccf'):
        print(f"    {s:>5}: {np.round(c[s], 2).tolist()}   "
              f"(mc_q gives each decision both rewards=12; factoring drops the other's)")