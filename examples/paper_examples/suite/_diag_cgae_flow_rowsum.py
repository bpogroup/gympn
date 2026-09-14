r"""Diagnostic: is cgae_flow's bootstrap coefficient a convex combination?

Proposition 3(iii) normalizes flow weights over PREDECESSORS:

    sum over d in pred(s) of w(d->s) = 1                                (*)

but `_redistribute_cgae`'s recursion sums over SUCCESSORS:

    A[d] = owned[d] + sum_{s in succ(d)} w(d->s) rho_s V[s] - V[d]
                    + lam sum_{s in succ(d)} w(d->s) rho_s A[s]

The coefficient multiplying the next-value term is therefore

    R(d) = sum_{s in succ(d)} w(d->s)                                   (row sum)

which (*) does NOT constrain. A valid TD backup needs R(d) <= 1 (convex
combination); R(d) > 1 inflates the bootstrap, and because the recursion is
applied backwards along the DAG the inflation compounds multiplicatively with
causal depth.

This measures, on real traces, for both envs:
  * the distribution of R(d),
  * the fraction of decisions with R(d) > 1,
  * the resulting credit scale vs the mean variant and vs the true return,
  * how the blow-up tracks causal depth.

Run: python _diag_cgae_flow_rowsum.py
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
EPISODES = 5


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
    total_r = 0.0
    while not done:
        _, r, done, _, _ = env.step(random.randrange(len(env.pn.pn_actions)))
        total_r += r
    return env, total_r


def edges_and_weights(ct):
    """Rebuild cgae_flow's edge weights (same construction as the library)."""
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

    w_edge = {}
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
            for d, c in contrib.items():
                w_edge[(d, idx)] = c / k
    return w_edge, n


def causal_depth(w_edge, n):
    """Longest causal path length from each decision (0 = sink)."""
    succ = {}
    for (d, s) in w_edge:
        succ.setdefault(d, []).append(s)
    depth = [0] * n
    for d in range(n - 1, -1, -1):
        kids = succ.get(d, [])
        if kids:
            depth[d] = 1 + max(depth[s] for s in kids)
    return depth


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    rows_all, depths_all = [], []
    max_flow = max_mean = 0.0
    ret_scale = []
    depth_vs_credit = []
    for ep in range(EPISODES):
        env, total_r = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        acts = ct.transition_history.get_action_transitions()
        n_dec = len(acts)
        if n_dec == 0:
            continue
        V = [0.3] * n_dec
        q_mean = np.asarray(ct.redistribute_rewards(scheme='cgae', beta=0.1,
                                                    values=V, lam=0.95), dtype=float)
        q_flow = np.asarray(ct.redistribute_rewards(scheme='cgae_flow', beta=0.1,
                                                    values=V, lam=0.95), dtype=float)

        w_edge, n = edges_and_weights(ct)
        succ = {}
        for (d, s) in w_edge:
            succ.setdefault(d, []).append(s)
        rows = [sum(w_edge[(d, s)] for s in succ[d]) for d in sorted(succ)]
        rows_all.extend(rows)
        dep = causal_depth(w_edge, n)
        depths_all.extend(dep)
        max_flow = max(max_flow, float(np.max(np.abs(q_flow))))
        max_mean = max(max_mean, float(np.max(np.abs(q_mean))))
        ret_scale.append(total_r)
        for d in sorted(succ):
            if d < len(q_flow):
                depth_vs_credit.append((dep[d], abs(float(q_flow[d]))))

    rows_all = np.asarray(rows_all)
    print("=" * 66)
    print(label)
    print("  decisions with >=1 causal successor : %d" % len(rows_all))
    print("  row sum R(d)=sum_succ w  mean %.3f  max %.3f" % (rows_all.mean(), rows_all.max()))
    print("  R(d) > 1 + 1e-9                     : %d / %d  (%.1f%%)"
          % ((rows_all > 1 + 1e-9).sum(), len(rows_all),
             100.0 * (rows_all > 1 + 1e-9).mean()))
    print("  R(d) > 1.5                          : %.1f%%" % (100.0 * (rows_all > 1.5).mean()))
    print("  max |credit|   mean-variant %.2f   flow-variant %.2f" % (max_mean, max_flow))
    print("  episode return scale (mean)         : %.2f" % (float(np.mean(ret_scale))))
    print("  max causal depth                    : %d" % (max(depths_all) if depths_all else 0))
    if depth_vs_credit:
        by_d = {}
        for d, c in depth_vs_credit:
            by_d.setdefault(d, []).append(c)
        print("  mean |flow credit| by causal depth:")
        for d in sorted(by_d)[:12]:
            print("      depth %2d  n=%4d  mean|q| = %8.2f" % (d, len(by_d[d]), float(np.mean(by_d[d]))))
print("=" * 66)
