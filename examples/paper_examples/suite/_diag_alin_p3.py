r"""Why doesn't alin degenerate to mc_q on s1?

alin's docstring claims: "Where every decision draws on one saturated pool,
every decision reaches every reward and this degenerates to mc_q exactly (s1,
s3)." Measured by _test_alin.py: s1 gives alin=0.2878 of mc_q's mass, not
1.0000 -- it subtracts ~71% of the mass where it should subtract nothing.

This reproduces alin's internals on real s1 traces and reports, stage by
stage, where the reachability stops:

  1. pools detected (_is_cyclic_place) and whether the employee pool is one
  2. how many decisions acquire from a pool -> enter the contention chain
  3. contention edges built, and how many of them link SAME-TIME decisions
  4. cycle-guard fallbacks -- reach[s] not yet computed when needed
  5. final coverage: |reach[d]| / |rewards at or after u_d|

Hypothesis for (4): the closure walks decisions in reverse time order, and
contention edges point earlier->later. For DISTINCT times a successor is
always processed first, so the closure is exact. For EQUAL times, `order`
sorts by time only and Python's stable sort leaves equal keys in ascending
index order, so a decision can be processed BEFORE its same-time successor --
and the guard then falls back to direct[s] instead of reach[s], truncating the
closure. s1 fires several assignments at the same clock (3-employee pool), so
ties should be common.

Run: python _diag_alin_p3.py
"""
import os, sys, random, types, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env

EPISODES = 6
LENGTH = 20


def build():
    pn = make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
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


tot = dict(pools=0, dec=0, in_chain=0, edges=0, tie_edges=0,
           fallbacks=0, cover_num=0.0, cover_den=0, exact=0)
pool_names = set()

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

    # --- rewards, as alin collects them ---
    rewards = []
    for tr in ct.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        rewards.append((rv, tr.get('time'), tr.get('input_tokens', []),
                        (ct.transition_history.get_action_transitions().index(tr)
                         if tr.get('is_action') and tr in acts else None)))

    # direct[]: rewards in each decision's realized token lineage (as alin does)
    token_to_action = {}
    for idx, a in enumerate(acts):
        for t in a.get('output_tokens', ()) or ():
            token_to_action[t] = idx
    def get_parents(tid):
        info = ct.token_history.get_token(tid)
        return info.get("parents", []) if info else []
    def lineage_decisions(in_ids, firing_idx):
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
            for pp in get_parents(tid):
                if pp not in seen:
                    stack.append(pp)
        return found
    direct = [set() for _ in range(n)]
    for j, (_rv, _t, in_ids, firing_idx) in enumerate(rewards):
        for idx in lineage_decisions(in_ids, firing_idx):
            if 0 <= idx < n:
                direct[idx].add(j)

    # --- pools ---
    trans = list(pn.actions) + list(pn.events)
    consumers = {}
    for t in trans:
        for p in t.incoming:
            consumers.setdefault(p._id, []).append(t)

    def _is_cyclic_place(pid, seeds):
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

    pool_ids = {pid for pid in consumers if _is_cyclic_place(pid, consumers.get(pid, []))}
    pool_names |= pool_ids

    # --- who enters the chain ---
    by_pool = {}
    for idx, act in enumerate(acts):
        tobj = act.get('transition')
        if tobj is None:
            continue
        u = act.get('time')
        for p in getattr(tobj, 'incoming', ()):
            if p._id in pool_ids:
                by_pool.setdefault(p._id, []).append((float(u) if u is not None else 0.0, idx))

    succ = [set() for _ in range(n)]
    in_chain = set()
    edges = tie = 0
    for pid, entries in by_pool.items():
        entries.sort()
        for (t_a, a) in entries:
            in_chain.add(a)
        for k in range(len(entries) - 1):
            succ[entries[k][1]].add(entries[k + 1][1])
            edges += 1
            if entries[k][0] == entries[k + 1][0]:
                tie += 1

    # --- closure, counting cycle-guard fallbacks ---
    def closure(with_index_tiebreak):
        key = (lambda i: (acts[i].get('time') is None, acts[i].get('time') or 0.0, i))               if with_index_tiebreak else               (lambda i: (acts[i].get('time') is None, acts[i].get('time') or 0.0))
        order = sorted(range(n), key=key, reverse=True)
        reach_, fb = [None] * n, 0
        for idx in order:
            acc = set(direct[idx])
            for s_ in succ[idx]:
                if reach_[s_] is not None:
                    acc |= reach_[s_]
                else:
                    acc |= direct[s_]
                    fb += 1
            reach_[idx] = acc
        return reach_, fb

    reach_old, fb_old = closure(False)
    reach, fallbacks = closure(True)
    tot['fallbacks_old'] = tot.get('fallbacks_old', 0) + fb_old
    for idx, act in enumerate(acts):
        u = act.get('time')
        avail = sum(1 for (_rv, t_j, _i, _f) in rewards if t_j is None or u is None or u <= t_j)
        got_o = sum(1 for j in reach_old[idx]
                    if rewards[j][1] is None or u is None or u <= rewards[j][1])
        if avail:
            tot['cover_old'] = tot.get('cover_old', 0.0) + got_o / avail
            if got_o >= avail:
                tot['exact_old'] = tot.get('exact_old', 0) + 1

    # --- coverage vs the mc_q horizon ---
    for idx, act in enumerate(acts):
        u = act.get('time')
        avail = sum(1 for (_rv, t_j, _i, _f) in rewards
                    if t_j is None or u is None or u <= t_j)
        got = sum(1 for j in reach[idx]
                  if rewards[j][1] is None or u is None or u <= rewards[j][1])
        tot['cover_den'] += 1
        if avail:
            tot['cover_num'] += got / avail
            if got >= avail:
                tot['exact'] += 1

    tot['pools'] += len(pool_ids); tot['dec'] += n
    tot['in_chain'] += len(in_chain); tot['edges'] += edges
    tot['tie_edges'] += tie; tot['fallbacks'] += fallbacks

print(f"s1, {EPISODES} episodes")
print(f"  pools detected           : {sorted(pool_names)}")
print(f"  decisions                : {tot['dec']}")
print(f"  decisions in the chain   : {tot['in_chain']} "
      f"({100*tot['in_chain']/max(1,tot['dec']):.1f}%)  <- rest get NO contention successor")
print(f"  contention edges         : {tot['edges']}")
print(f"    of which SAME-TIME     : {tot['tie_edges']} "
      f"({100*tot['tie_edges']/max(1,tot['edges']):.1f}%)")
print(f"  cycle-guard fallbacks    : time-only sort {tot['fallbacks_old']} "
      f"-> (time,index) sort {tot['fallbacks']}")
print(f"  mean reachable fraction  : time-only {tot['cover_old']/max(1,tot['cover_den']):.3f} "
      f"-> fixed {tot['cover_num']/max(1,tot['cover_den']):.3f}  (of the mc_q horizon)")
print(f"  decisions reaching ALL   : time-only {tot['exact_old']}/{tot['cover_den']} "
      f"({100*tot['exact_old']/max(1,tot['cover_den']):.1f}%) -> fixed {tot['exact']}/{tot['cover_den']} "
      f"({100*tot['exact']/max(1,tot['cover_den']):.1f}%)  <- P3 wants ~100%")
