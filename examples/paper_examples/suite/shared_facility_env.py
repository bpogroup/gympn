r"""Shared facility: causal components that are NOT planted in the topology.

WHY THIS EXISTS. On `ncopies` the independence is structural -- N disjoint
copies, each with its own queue and server -- so a referee can fairly say the
factorization was designed in, and any method told the partition would exploit
it. This env removes that objection: ONE connected net, all k pools and all
routes present at every setting, and the causal factorization is a property of
the REALIZED trajectory rather than of the model.

THE DIAL. Every case is assigned a class at arrival, which fixes its route:

    with prob 1-p   a SINGLE-pool route  [i]              (confined)
    with prob p     a BRIDGE route       [i,j] or [i,j,l] (spans pools)

At p=0 no case ever touches two pools, so the realized provenance DAG splits
into k independent sub-systems -- even though the net is one connected
component and nothing in it declares a partition. At p=1 every case chains two
or three pools together and the whole episode is one component. In between,
bridge cases stitch pools together and the realized component count falls from
k to 1 (percolation over the pool graph, so expect a fairly sharp transition;
sample p finely near it).

The method is never told p, k, or the routes. It has to recover the structure
from token provenance at runtime -- which is the actual claim being tested,
and the thing factored-MDP methods cannot do because they are handed the DBN.

WHY COUPLING IS DRIVEN BY ARRIVALS, NOT BY THE AGENT. A case's class is drawn
on arrival and fixed; the agent only chooses WHICH waiting case a free server
starts next, never where a case goes. Component membership is therefore
F_d-measurable and assumption (A1) of EJOR_PROPOSITIONS.md holds. Letting the
agent route instead would reproduce the AND-join motif of `assembly_probe.py`,
where the realized partition becomes action-dependent, (A1) fails, and ccf
picks up a sign error -- interesting as a negative result, wrong for a
headline env.

WHY REUSE, NOT CONTENTION, IS THE THING BEING TUNED. A first design tried to
tune LOAD on one shared pool: low utilization -> cases rarely contend ->
components separate. That does not work. Two cases are unioned when they reuse
the same server TOKEN, whether or not either ever waited, so a single shared
pool yields one component at every load -- which is exactly why s1 measures as
one reward-bearing component holding 100% of reward mass despite not being
saturated. The dial has to control whether two cases ever touch the same pool
at all, which is what the route classes do.

HEADROOM. Service times vary by (class, pool) and a case pays for every stage
of its route, so a short single-pool case is cheap and a three-stage bridge
case is expensive. With a bounded horizon and reward 1 per COMPLETED case,
which case a free server picks up matters a lot; the anchor heuristic is
shortest-remaining-work-first, and a random policy loses badly to it.

Run `python shared_facility_env.py` for a structural smoke test (component
count vs p) -- no training.
"""
import random

from simpn.simulator import SimToken

from gympn.simulator import GymProblem


# --------------------------------------------------------------------------
# Route catalogue: deterministic in k, so `behavior` and `reward_function`
# always agree (the simulator invokes them separately -- see the s4 docstring
# in stoch_envs.py; a coin flip inside either would be drawn twice).
# --------------------------------------------------------------------------
def routes_for(k: int):
    """(singles, bridges) as tuples of pool indices. Route length 1, 2 or 3."""
    singles = [(i,) for i in range(k)]
    bridges = []
    for i in range(k):
        for j in range(i + 1, k):
            bridges.append((i, j))
    for i in range(k):
        for j in range(i + 1, k):
            for l in range(j + 1, k):
                bridges.append((i, j, l))
    return singles, bridges


_ROUTES = None          # catalogue of the most recently built env (heuristic uses it)


_SPREAD = 3             # set by make_shared_facility; width of the svc range
_SERVERS = 1            # servers per pool (heuristic needs it for the match rule)
_PENALTY = 3            # cross-assignment multiplier: the size of a mistake
_VALUE_SPREAD = 1       # max reward of a completed case (1 = all cases equal)


def _value(cls: int) -> int:
    """Reward paid for COMPLETING a case of this class.

    Headroom has to come from a dimension the agent does not route. Making
    SERVERS heterogeneous (a matching decision) works, but with >1 server per
    pool the agent then chooses which cases share a server token -- i.e. it
    controls the realized partition, component membership stops being
    F_d-measurable, and (A1) fails exactly as in assembly_probe.py. Making
    CASES heterogeneous instead keeps one server per pool, so the partition is
    forced and exogenous, while still giving the ordering decision teeth."""
    return 1 + (cls * 5 + 3) % max(1, _VALUE_SPREAD)


def _svc(cls: int, pool: int, code_server: int = 0, n_srv: int = 1) -> int:
    """Service time = ordering cost x MATCHING cost.

    The base term (class, pool) makes some cases cheaper than others, so
    ordering matters. The match term makes assigning the RIGHT server cheap
    and the wrong one expensive, so assignment quality matters -- that is what
    generates real headroom over a random policy. An ordering-only version of
    this env gave the heuristic just +4% over random even under load, because
    every job has to be served eventually and a short horizon leaves little
    for SPT to win; s1 and ncopies get their 40-50% gaps from matching, not
    from ordering."""
    base = 1 + ((cls * 7 + pool * 3) % _SPREAD)
    if n_srv > 1 and (cls % n_srv) != code_server:
        base *= _PENALTY                # cross-assigned: _PENALTY times slower
    return base


def make_shared_facility(k=4, p_bridge=0.5, servers_per_pool=1,
                         arrivals_per_tick=1, svc_spread=3, match_penalty=3,
                         value_spread=1, deadline=None,
                         causal_rl=False, allow_postpone=True,
                         causal_postpone_tokenflow=False):
    """k pools sharing one net; p_bridge tunes realized coupling (see module doc)."""
    global _SPREAD, _SERVERS, _PENALTY, _VALUE_SPREAD
    _SPREAD, _SERVERS, _PENALTY = svc_spread, servers_per_pool, match_penalty
    _VALUE_SPREAD = value_spread
    singles, bridges = routes_for(k)
    ROUTES = singles + bridges
    n_single = len(singles)

    ag = GymProblem(allow_postpone=allow_postpone, causal_rl=causal_rl,
                    causal_postpone_tokenflow=causal_postpone_tokenflow)

    completed = ag.add_var("completed", var_attributes=['cls', 'stage', 'cid'])
    waiting, busy, server, dl = [], [], [], []
    for i in range(k):
        waiting.append(ag.add_var(f"waiting_{i}", var_attributes=['cls', 'stage', 'cid']))
        # companion deadline token, available only at t+deadline: it is what
        # makes `abandon_i` become enabled while the case is still queued.
        dl.append(ag.add_var(f"dl_{i}", var_attributes=['cid']))
        busy.append(ag.add_var(f"busy_{i}", var_attributes=['cls', 'stage', 'cid', 'code_server']))
        s = ag.add_var(f"server_{i}", var_attributes=['code_server'])
        for e in range(servers_per_pool):
            s.put({'code_server': e})
        server.append(s)

    # One arrival STREAM per case-per-tick. A PN transition emits at most one
    # token per output place per firing, so a single `arrive` could not put two
    # cases into the same queue in one tick -- the earlier version silently
    # dropped the extras (`mine[0]`), which capped the real arrival rate and
    # starved the queue pressure that abandonment needs. Separate streams also
    # give every case its OWN cid, which the abandon guard matches on; sharing
    # one cid per tick let a deadline token expire the wrong case.
    arrivals = []
    for j in range(arrivals_per_tick):
        av = ag.add_var(f"arrival_{j}", var_attributes=['cid'])
        av.put({'cid': 0})
        arrivals.append(av)

    def _draw_cls():
        if random.random() < p_bridge and bridges:
            return n_single + random.randrange(len(bridges))
        return random.randrange(n_single)

    # warm start: one confined and one bridge case already queued, so t=0 is a
    # real decision rather than a forced move.
    if bridges:
        waiting[ROUTES[n_single][0]].put({'cls': n_single, 'stage': 0, 'cid': -1})
    waiting[0].put({'cls': 0, 'stage': 0, 'cid': -2})

    def _make_arrive(j):
        """One case per tick per stream. Class (hence route) is exogenous -- the
        agent cannot influence which pools a case will visit, which is what
        keeps component membership F_d-measurable."""
        def arrive(a):
            nxt_cid = a['cid'] + 1
            cid = nxt_cid * arrivals_per_tick + j      # globally unique per case
            cls = _draw_cls()
            first = ROUTES[cls][0]
            out = [SimToken({'cid': nxt_cid}, delay=1)]
            for i in range(k):
                out.append(SimToken({'cls': cls, 'stage': 0, 'cid': cid})
                           if i == first else None)
            for i in range(k):
                out.append(SimToken({'cid': cid}, delay=deadline)
                           if (deadline is not None and i == first) else None)
            return out
        return arrive

    for j in range(arrivals_per_tick):
        ag.add_event([arrivals[j]], [arrivals[j]] + waiting + dl, _make_arrive(j),
                     name=f'arrive_{j}')

    def _is_last(b):
        case = b[0]
        return case['stage'] + 1 >= len(ROUTES[case['cls']])

    def _make_start(_i):
        # closure, not a default arg: add_action checks the parameter COUNT
        # against the number of input places, so the behavior must take
        # exactly (case, server).
        def start(c, r):
            d = _svc(c['cls'], _i, r['code_server'], servers_per_pool)
            return [SimToken((c, r), delay=d + random.randint(0, 1))]
        return start

    def _make_done(_i):
        def done(b):
            """Release the server, then route: next stage's queue, or the sink.
            Outgoing order is [server_i, waiting_0..waiting_{k-1}, completed]."""
            case, res = b[0], b[-1]
            out = [SimToken(res)]
            nxt = None
            if _is_last(b):
                for _ in range(k):
                    out.append(None)
            else:
                nxt = ROUTES[case['cls']][case['stage'] + 1]
                moved = dict(case)
                moved['stage'] = case['stage'] + 1
                for j in range(k):
                    out.append(SimToken(moved) if j == nxt else None)
            for j in range(k):
                out.append(SimToken({'cid': case['cid']}, delay=deadline)
                           if (deadline is not None and j == nxt) else None)
            out.append(SimToken(dict(case)) if _is_last(b) else None)
            return out
        return done

    for i in range(k):
        ag.add_action([waiting[i], server[i]], [busy[i]], behavior=_make_start(i),
                      name=f"start_{i}")
        ag.add_event([busy[i]], [server[i]] + waiting + dl + [completed], _make_done(i),
                     name=f'done_{i}',
                     reward_function=lambda b: _value(b[0]['cls']) if _is_last(b) else 0)

    if deadline is not None:
        # A queued case is LOST if it is not started within `deadline`. This is
        # what makes a choice have lasting consequences in a SINGLE-STAGE
        # system: serve the wrong case and the other is gone for good. Without
        # it, difficulty could only be created by longer routes -- the same
        # knob that creates coupling -- so difficulty and coupling could not be
        # varied independently, which is what blocked every earlier version of
        # this env (ordering +4%, matching +7.6%, value +5.2% headroom at p=0).
        for i in range(k):
            ag.add_event([waiting[i], dl[i]], [], lambda w, d: [],
                         guard=lambda w, d: w['cid'] == d['cid'],
                         name=f'abandon_{i}')

    global _ROUTES
    _ROUTES = ROUTES             # for the heuristic / diagnostics
    ag._routes = ROUTES
    ag._k = k
    return ag


def _ratio(b, remaining_fn):
    """Value per unit of remaining work -- the myopic anchor."""
    try:
        for (_place, tok) in b[0]:
            v = getattr(tok, 'value', tok)
            if isinstance(v, dict) and 'cls' in v:
                return _value(v['cls']) / max(1e-9, remaining_fn(b))
    except Exception:
        pass
    return 0.0


def shared_facility_heuristic(observable_net, tokens_comb, bindings=None):
    """Shortest REMAINING work first: prefer the waiting case whose unfinished
    route costs least, so a bounded horizon buys the most completions. A single
    -pool case therefore outranks a three-stage bridge case, and among equals
    the cheaper class wins. Never postpones. A strong myopic anchor, not the
    optimum -- a normalized score above 1.0 means the policy beat it."""
    if not bindings:
        return None
    real = [b for b in bindings if isinstance(b, tuple) and b and isinstance(b[0], list)
            and b[0] != ['postpone']]
    if not real:
        return bindings[0]

    def remaining(b):
        """Cost of finishing this case from here, given the server on offer:
        matched-and-short first."""
        try:
            case = srv = None
            for (_place, tok) in b[0]:
                v = getattr(tok, 'value', tok)
                if isinstance(v, dict) and 'cls' in v and 'stage' in v:
                    case = v
                elif isinstance(v, dict) and 'code_server' in v:
                    srv = v
            if case is None:
                return 999
            route = _ROUTES[case['cls']]
            here = route[case['stage']]
            n_srv = _SERVERS
            first = _svc(case['cls'], here,
                         srv['code_server'] if srv else 0, n_srv)
            rest = sum(_svc(case['cls'], pool, case['cls'] % max(1, n_srv), n_srv)
                       for pool in route[case['stage'] + 1:])
            return first + rest
        except Exception:
            pass
        return 999

    return max(real, key=lambda b: _ratio(b, remaining))


if __name__ == "__main__":
    import os, sys, types, uuid
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gympn.environment import AEPN_Env

    K, EPISODES, LENGTH = 4, 5, 20

    def prep(p, seed):
        random.seed(seed)
        pn = make_shared_facility(K, p_bridge=p, causal_rl=True, allow_postpone=True,
                                  causal_postpone_tokenflow=True)
        pn.length = LENGTH
        for pl in pn.places:
            for t in pl.marking:
                setattr(t, '_id', str(uuid.uuid4()))
        pn.causal_trace._pn = pn
        pn.causal_trace.flush()
        sen = types.SimpleNamespace(_id="__initial__")
        toks = [t for pl in pn.places for t in pl.marking]
        for t in toks:
            pn.causal_trace.register_token(t, sen, [], time=0)
        pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
        return AEPN_Env(pn)

    def analyse(env):
        """ccf's realized union-find partition + cgae's fan-out, on the LIVE
        trace (env.pn, not the frozen net reset() copied from)."""
        ct = env.pn.causal_trace
        acts = ct.transition_history.get_action_transitions()
        n = len(acts)
        if n == 0:
            return None
        out_tok = {}
        for idx, a in enumerate(acts):
            for t in a.get("output_tokens", ()) or ():
                out_tok[t] = idx
        def par(tid):
            info = ct.token_history.get_token(tid)
            return info.get("parents", []) if info else []
        def lineage(ids):
            found, seen, stack = set(), set(), list(ids)
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                hit = out_tok.get(tid)
                if hit is not None:
                    found.add(hit)
                for q in par(tid):
                    if q not in seen:
                        stack.append(q)
            return found
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        # two passes: every union must be done BEFORE any root is used as a
        # dict key, or early entries end up filed under stale roots and the
        # component count inflates to roughly one-per-reward.
        rew, rew_n = [], 0
        for tr in ct.transition_history.transitions:
            rv = tr.get("reward", 0.0)
            if rv == 0.0:
                continue
            rew_n += 1
            decs = [d for d in lineage(tr.get("input_tokens", ())) if 0 <= d < n]
            for i in range(1, len(decs)):
                union(decs[0], decs[i])
            rew.append((rv, decs))
        mass = {}
        for rv, decs in rew:
            if decs:
                mass[find(decs[0])] = mass.get(find(decs[0]), 0.0) + rv
        # cgae fan-out
        succ = [set() for _ in range(n)]
        for idx, a in enumerate(acts):
            seen, stack = set(), list(a.get("input_tokens", ()) or ())
            while stack:
                tid = stack.pop()
                if tid in seen:
                    continue
                seen.add(tid)
                src = out_tok.get(tid)
                if src is not None and src != idx:
                    succ[src].add(idx); continue
                for q in par(tid):
                    if q not in seen:
                        stack.append(q)
        tot = sum(mass.values()) or 1.0
        return dict(n=n, rew=rew_n, K=len(mass),
                    largest=max(mass.values()) / tot if mass else float("nan"),
                    fan=sum(len(x) for x in succ) / max(1, n))

    print(f"shared_facility, k={K}, {EPISODES} episodes/point "
          f"({len(routes_for(K)[0])} single + {len(routes_for(K)[1])} bridge classes)")
    print(f"  {'p':<6}{'decisions':>10}{'rewards':>9}{'K_realized':>12}{'largest%':>10}{'fan-out':>9}")
    for p in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        rows = []
        for ep in range(EPISODES):
            env = prep(p, 400 + ep)
            env.reset()
            done, steps = False, 0
            while not done and steps < 500:
                m = len(env.pn.pn_actions)
                if m == 0:
                    break
                _, _, done, _, _ = env.step(random.randrange(m))
                steps += 1
            r = analyse(env)
            if r:
                rows.append(r)
        if not rows:
            print(f"  {p:<6} (no decisions)"); continue
        f = lambda key: sum(r[key] for r in rows) / len(rows)
        print(f"  {p:<6}{f('n'):>10.1f}{f('rew'):>9.1f}{f('K'):>12.2f}"
              f"{100*f('largest'):>9.1f}%{f('fan'):>9.2f}")
