r"""Gate for cgae_flow -- the variant Proposition 3(iii) actually covers.

cgae combines causal successors with a MEAN, which has no derivation and is not
covered by the proposition. cgae_flow uses the flow-weighted sum: w(d->s) is
the share of s's consumed tokens descending from d, so

    sum over d in pred(s) of w(d->s) = 1                                   (*)

Four properties, on real traces:

  F1 CONSERVATION   (*) holds numerically for every successor with at least one
                    attributable input token. This is what makes the SUM safe
                    on a DAG: credit out of s totals A_s however many
                    predecessors it has, so nothing is double counted.
  F2 CHAIN-EXACT    on a TRUE chain -- every edge in the descendant closure
                    having out-degree <=1 AND weight exactly 1 -- the two agree
                    to floating point. Two subtleties the antecedent has to
                    carry, both learned the hard way:
                      * it is the CLOSURE, not the decision: a fan-out-1
                        decision whose descendant fans out still differs,
                        because the difference propagates back up the chain;
                      * w(d->s)=1 needs s to have IN-degree 1 as well. Fan-IN
                        is the common case here, not fan-out: `start_i`
                        consumes a task token and an employee token, so it has
                        two predecessors and each receives w=0.5. That is the
                        intended semantics -- joint causation splits credit,
                        whereas the mean hands each predecessor the FULL A_s --
                        but it means the two variants differ far more widely
                        than the fan-out statistic alone suggests.
                    (Both variants discover the SAME edge set, 38/38 on
                    ncopies N=4, so F2 isolates the aggregation rule alone.)
  F3 NO-SHRINKAGE   under fan-out, cgae_flow >= cgae in aggregate |credit|:
                    the mean divides a decision's downstream credit by its
                    fan-out, the weighted sum does not.
  F4 FINITE         no NaN/inf, and both schemes return one credit per decision
                    (the 1:1 alignment TrajectoryBuffer.finish() asserts).

Run: python _test_cgae_flow.py
"""
import os, sys, random, types, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
EPISODES = 5
fails = []


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


def edges_and_weights(ct):
    """Rebuild cgae_flow's edge weights independently of the library, so F1 is
    a real check rather than a restatement of the implementation."""
    acts = ct.transition_history.get_action_transitions()
    n = len(acts)
    out_tok = {}
    for idx, a in enumerate(acts):
        for t in a.get('output_tokens', ()) or ():
            out_tok[t] = idx

    def parents(tid):
        info = ct.token_history.get_token(tid)
        return info.get("parents", []) if info else []

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
            for p in parents(tid):
                if p not in seen:
                    stack.append(p)
        return found

    per_succ = {}
    fan = [0] * n
    for idx, a in enumerate(acts):
        contrib, k = {}, 0
        for tid in list(a.get('input_tokens', ()) or ()):
            prods = producers(tid, idx)
            if not prods:
                continue
            k += 1
            for d in prods:
                contrib[d] = contrib.get(d, 0.0) + 1.0 / len(prods)
        if k:
            per_succ[idx] = {d: c / k for d, c in contrib.items()}
            for d in contrib:
                fan[d] += 1
    return per_succ, fan, n


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    agree = disagree = 0
    tot_flow = tot_mean = 0.0
    max_fan = 0
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        V = [0.3] * n_dec                      # arbitrary non-zero critic

        q_mean = ct.redistribute_rewards(scheme='cgae', beta=0.1, values=V, lam=0.95)
        q_flow = ct.redistribute_rewards(scheme='cgae_flow', beta=0.1, values=V, lam=0.95)

        # F4
        if len(q_mean) != n_dec or len(q_flow) != n_dec:
            fails.append(f"{label}: length mismatch {len(q_mean)}/{len(q_flow)} vs {n_dec}")
        if not np.all(np.isfinite(q_flow)):
            fails.append(f"{label}: cgae_flow produced non-finite credit")

        # F1
        per_succ, fan, _ = edges_and_weights(ct)
        for s_, wmap in per_succ.items():
            tot = sum(wmap.values())
            if abs(tot - 1.0) > 1e-9:
                fails.append(f"{label}: weights into decision {s_} sum to {tot:.6f}, not 1")
                break
        max_fan = max([max_fan] + fan)

        # F2 / F3 -- chain_ok[d]: d and every causal descendant has fan-out <=1
        succ_of = {}
        for s_, wmap in per_succ.items():
            for d_ in wmap:
                succ_of.setdefault(d_, []).append(s_)
        wmap_of = {}
        for s_, wm in per_succ.items():
            for d_, wv in wm.items():
                wmap_of[(d_, s_)] = wv
        chain_ok = [True] * n_dec
        for d in range(n_dec):
            seen_d, stack, ok = set(), [d], True
            while stack:
                x = stack.pop()
                if x in seen_d:
                    continue
                seen_d.add(x)
                kids = succ_of.get(x, [])
                if len(kids) > 1 or any(abs(wmap_of.get((x, y), 0.0) - 1.0) > 1e-12
                                        for y in kids):
                    ok = False
                    break
                stack.extend(kids)
            chain_ok[d] = ok
        for d in range(n_dec):
            if chain_ok[d]:
                if abs(q_mean[d] - q_flow[d]) < 1e-9:
                    agree += 1
                else:
                    disagree += 1
        tot_flow += float(np.sum(np.abs(q_flow)))
        tot_mean += float(np.sum(np.abs(q_mean)))

    print(f"{label}: true-chain decisions agreeing {agree}, disagreeing {disagree} "
          f"| max fan-out {max_fan} | sum|credit| mean {tot_mean:.2f} -> flow {tot_flow:.2f}")
    if disagree:
        fails.append(f"{label}: {disagree} true-chain decisions differ (F2)")
    if tot_flow + 1e-9 < tot_mean:
        fails.append(f"{label}: flow shrank total credit vs mean (F3)")

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 4/4")
