r"""What exactly does alin fail to reach on s1?

After fixing the time-tie ordering (closure now exact, 0 fallbacks), s1 still
reaches only ~32% of decisions' full mc_q horizon, against the docstring's
"degenerates to mc_q EXACTLY" on a saturated pool. This classifies every
(decision, missed reward) pair to find out whether that is a defect or the
docstring overclaiming.

For a decision d and a reward j in d's mc_q horizon (t_j >= u_d) but NOT in
reach[d], exactly one of:

  IN-FLIGHT   every decision in j's realized lineage fired STRICTLY BEFORE d.
              The case was already under way; d did not start it and cannot
              have delayed it (it is being served, not queued). Contention
              edges run earlier->later, so d structurally cannot reach it --
              and SHOULD NOT. mc_q counts it only because it happens later in
              wall-clock time. Correct subtraction; the docstring is wrong.

  TRUNCATED   some decision in j's lineage fired at or after d, yet d does not
              reach it. That IS a real hole in the contention closure.

  ORPHAN      j has no lineage decisions at all.

Split by whether d is a postpone decision, since postpone sentinels carry no
`incoming` places and so never join a pool's contention chain.

Run: python _diag_alin_p3b.py
"""
import os, sys, random, types, uuid
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env

EPISODES = 6
LENGTH = 20
ENV = "s1_stoch_sequence"


def build():
    pn = make_env(ENV, causal_rl=True, allow_postpone=True,
                  causal_postpone_tokenflow=True)
    pn.length = LENGTH
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn
    pn.causal_trace._static_comp_cache = None
    pn.causal_trace.flush()
    sent = types.SimpleNamespace(_id="__initial__")
    for p in pn.places:
        for t in p.marking:
            pn.causal_trace.register_token(t, sent, parent_tokens=[], time=0)
    pn.causal_trace.register_transition(
        transition=sent, input_tokens=[],
        output_tokens=[t for p in pn.places for t in p.marking],
        is_action=False, reward=0.0, time=0)
    return AEPN_Env(pn)


cls = Counter()
by_kind = Counter()

for ep in range(EPISODES):
    random.seed(100 + ep)
    env = build(); env.reset(); done = False
    while not done:
        _, _, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions)))

    ct = env.pn.causal_trace
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    if n == 0:
        continue
    pn = ct._pn
    times = [a.get('time') for a in acts]
    is_pp = [isinstance(getattr(a.get('transition'), '_id', None), str)
             and str(getattr(a.get('transition'), '_id', '')).startswith('postpone_')
             for a in acts]

    rewards = []
    for tr in ct.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        rewards.append((rv, tr.get('time'), tr.get('input_tokens', []), None))

    token_to_action = {}
    for idx, a in enumerate(acts):
        for t in a.get('output_tokens', ()) or ():
            token_to_action[t] = idx

    def parents(tid):
        info = ct.token_history.get_token(tid)
        return info.get("parents", []) if info else []

    def lineage_decisions(in_ids, firing_idx=None):
        found = set()
        if firing_idx is not None:
            found.add(firing_idx)
        seen, stack = set(), list(in_ids)
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            hit = token_to_action.get(tid)
            if hit is not None:
                found.add(hit)
            for pp in parents(tid):
                if pp not in seen:
                    stack.append(pp)
        return found

    lin = [lineage_decisions(r[2]) for r in rewards]
    direct = [set() for _ in range(n)]
    for j, ds in enumerate(lin):
        for idx in ds:
            if 0 <= idx < n:
                direct[idx].add(j)

    # contention chain (same construction as alin)
    trans = list(pn.actions) + list(pn.events)
    consumers = {}
    for t in trans:
        for p in t.incoming:
            consumers.setdefault(p._id, []).append(t)

    def cyc(pid, seeds):
        seen_p, seen_t, stack = set(), set(), list(seeds)
        while stack:
            t = stack.pop()
            if t._id in seen_t:
                continue
            seen_t.add(t._id)
            for p in t.outgoing:
                if p._id == pid:
                    return True
                if p._id in seen_p:
                    continue
                seen_p.add(p._id)
                stack.extend(consumers.get(p._id, []))
        return False

    pools = {pid for pid in consumers if cyc(pid, consumers.get(pid, []))}
    by_pool = {}
    for idx, a in enumerate(acts):
        tobj = a.get('transition')
        if tobj is None:
            continue
        u = a.get('time')
        for p in getattr(tobj, 'incoming', ()):
            if p._id in pools:
                by_pool.setdefault(p._id, []).append((float(u) if u is not None else 0.0, idx))
    succ = [set() for _ in range(n)]
    for pid, entries in by_pool.items():
        entries.sort()
        for k in range(len(entries) - 1):
            succ[entries[k][1]].add(entries[k + 1][1])

    order = sorted(range(n), key=lambda i: (times[i] is None, times[i] or 0.0, i), reverse=True)
    reach = [None] * n
    for idx in order:
        acc = set(direct[idx])
        for s_ in succ[idx]:
            acc |= reach[s_] if reach[s_] is not None else direct[s_]
        reach[idx] = acc

    for d in range(n):
        u = times[d]
        for j, (rv, t_j, _in, _f) in enumerate(rewards):
            if not (t_j is None or u is None or u <= t_j):
                continue                     # outside the mc_q horizon
            if j in reach[d]:
                cls['reached'] += 1
                continue
            ds = [k for k in lin[j] if 0 <= k < n]
            if not ds:
                kind = 'ORPHAN'
            elif all((times[k] is not None and u is not None and times[k] < u) for k in ds):
                kind = 'IN-FLIGHT'
            else:
                kind = 'TRUNCATED'
                # of the lineage decisions at/after d, are they ALL postpones?
                at_after = [k for k in ds
                            if not (times[k] is not None and u is not None and times[k] < u)]
                if at_after and all(is_pp[k] for k in at_after):
                    cls['TRUNC_via_postpone_only'] += 1
                else:
                    cls['TRUNC_other'] += 1
            cls[kind] += 1
            by_kind[(kind, 'postpone' if is_pp[d] else 'real')] += 1

tot_missed = cls['IN-FLIGHT'] + cls['TRUNCATED'] + cls['ORPHAN']
print(f"{ENV}, {EPISODES} episodes  (post-fix closure)")
print(f"  (decision, reward) pairs in the mc_q horizon : {sum(cls.values())}")
print(f"  reached by alin                              : {cls['reached']}")
print(f"  missed                                       : {tot_missed}")
for k in ('IN-FLIGHT', 'TRUNCATED', 'ORPHAN'):
    pct = 100 * cls[k] / tot_missed if tot_missed else 0
    print(f"     {k:10s} {cls[k]:6d}  ({pct:5.1f}% of misses)")
print(f"     of the {cls['TRUNCATED']} TRUNCATED: "
      f"{cls['TRUNC_via_postpone_only']} reachable ONLY via a postpone lineage node, "
      f"{cls['TRUNC_other']} other")
print()
print("  missed pairs split by decision type:")
for k in ('IN-FLIGHT', 'TRUNCATED', 'ORPHAN'):
    print(f"     {k:10s} real={by_kind[(k,'real')]:6d}  postpone={by_kind[(k,'postpone')]:6d}")
