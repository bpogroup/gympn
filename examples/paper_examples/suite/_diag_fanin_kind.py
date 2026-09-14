r"""Is s1's fan-in genuine RE-CONVERGENCE, or merely joint causation?

The distinction decides whether the dilution is fixable.

  * JOINT CAUSATION -- s consumes a task token from d and an employee token
    from an unrelated e. From d there is exactly ONE path to s, so d's weight
    on s's downstream reward should be 1. The inflow normalization sets it to
    0.5. That is pure loss, and it is fixable.

  * RE-CONVERGENCE (a true diamond) -- d -> s1 -> j and d -> s2 -> j. From d
    there are TWO paths to j, so a plain weight-1 sum really would count j's
    reward twice. Here some correction IS needed; the inflow normalization is
    a crude one.

Measured per decision d: the number of distinct causal paths from d to each
reward-owning descendant j. Path count 1 for (almost) every (d, j) pair means
the DAG is effectively a forest from any given root, the diamonds are rare,
and weight-1 reachability with closure dedup is exactly right.

Run: python _diag_fanin_kind.py
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
EPISODES = 5
CAPTURE = {}
_orig = CausalTraces._redistribute_cgae


def _spy(self, action_transitions, token_to_action, record_to_action,
         redistribution, beta, get_parents, values=None, lam=1.0,
         flow=False, convex=False):
    n = len(action_transitions)
    out_tok = {}
    for idx, act in enumerate(action_transitions):
        for t in act.get('output_tokens', ()) or ():
            out_tok[t] = idx

    def producers(start, self_idx):
        found, seen, stack = set(), set(), [start]
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            src = out_tok.get(tid)
            if src is not None and src != self_idx:
                found.add(src)
                continue
            for p in get_parents(tid):
                if p not in seen:
                    stack.append(p)
        return found

    w_edge, succ = {}, {}
    for idx, act in enumerate(action_transitions):
        contrib, k = {}, 0
        for tid in list(act.get('input_tokens', ()) or ()):
            prods = producers(tid, idx)
            if not prods:
                continue
            k += 1
            for d in prods:
                contrib[d] = contrib.get(d, 0.0) + 1.0 / len(prods)
        if k:
            for d, c in contrib.items():
                w_edge[(d, idx)] = c / k
                succ.setdefault(d, set()).add(idx)

    times = [a.get('time') for a in action_transitions]

    def lineage(input_ids, firing_idx):
        found = set()
        if firing_idx is not None:
            found.add(firing_idx)
        seen, stack = set(), list(input_ids)
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            hit = token_to_action.get(tid)
            if hit is not None:
                found.add(hit[0])
            for p in get_parents(tid):
                if p not in seen:
                    stack.append(p)
        return found

    owned = [0.0] * n
    for tr in self.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        se = record_to_action.get(id(tr))
        decs = lineage(tr.get('input_tokens', []), se[0] if se else None)
        decs = [d for d in decs if 0 <= d < n]
        if not decs:
            continue
        owner = max(decs, key=lambda d: (times[d] is not None, times[d] or 0.0))
        owned[owner] += rv
    CAPTURE.update(succ=succ, w_edge=w_edge, owned=owned, n=n, times=times)
    return _orig(self, action_transitions, token_to_action, record_to_action,
                 redistribution, beta, get_parents, values, lam, flow, convex)


CausalTraces._redistribute_cgae = _spy


def build(builder):
    pn = builder()
    pn.length = LENGTH
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace._pn = pn
    pn.causal_trace._static_comp_cache = None
    pn.causal_trace.postpone_tokenflow = True
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


def rollout(builder, seed):
    random.seed(seed)
    env = build(builder)
    env.reset()
    done = False
    while not done:
        _, _, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions)))
    return env


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    pathcounts, indeg_kind = [], []
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        ct.redistribute_rewards(scheme='cgae_flow', beta=0.0,
                                values=[0.0] * n_dec, lam=1.0)
        succ, owned, n = CAPTURE['succ'], CAPTURE['owned'], CAPTURE['n']
        order = sorted(range(n), reverse=True)

        # in-degree composition: how many predecessors each node has
        pred = {}
        for d, kids in succ.items():
            for s in kids:
                pred.setdefault(s, set()).add(d)
        for s, ps in pred.items():
            indeg_kind.append(len(ps))

        # number of distinct causal paths d -> j, for reward-owning j
        for d in range(n):
            npaths = {d: 1}
            for x in sorted(npaths):
                pass
            # forward DP in topological (index) order
            npaths = [0] * n
            npaths[d] = 1
            for x in range(d, n):
                if npaths[x] == 0:
                    continue
                for s in succ.get(x, ()):
                    if s > x:
                        npaths[s] += npaths[x]
            for j in range(n):
                if j != d and npaths[j] > 0 and abs(owned[j]) > 1e-12:
                    pathcounts.append(npaths[j])

    pc = np.asarray(pathcounts)
    ik = np.asarray(indeg_kind)
    print("=" * 70)
    print(label)
    print("  in-degree of causal successors: mean %.2f | ==1: %.1f%% | >=2: %.1f%%"
          % (ik.mean(), 100.0 * np.mean(ik == 1), 100.0 * np.mean(ik >= 2)))
    if len(pc):
        print("  distinct causal paths d->j (reward-owning j), n=%d pairs" % len(pc))
        print("      ==1 path : %.1f%%   >=2 paths : %.1f%%   max %d   mean %.2f"
              % (100.0 * np.mean(pc == 1), 100.0 * np.mean(pc >= 2), pc.max(), pc.mean()))
print("=" * 70)
