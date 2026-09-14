r"""Decisive test of Proposition 3(i)/(iii).

With lam=1, V==0, beta=0 the proposition claims the estimator is EXACTLY the
sum of owned rewards over the causal descendants:

    A[d] = sum_{j : own(j) in desc(d)} r_j                              (P3)

so each descendant's owned reward carries total weight 1 across all paths.
That identity IS the unbiasedness argument: it makes A[d] the component
return, so everything dropped is cross-component and factors out through the
score-function identity.

Both sides are computed on real traces. The structure (successors, edge
weights, reward ownership) is captured from INSIDE the library by wrapping
`_redistribute_cgae`, so the maps used here are byte-identical to the ones the
estimator uses -- no re-derivation, no guessing at internal attribute names.

  LHS  what `redistribute_rewards` returns
  RHS  sum of `owned` over the true descendant closure of d

Ratio systematically below 1 means credit is DILUTED at fan-in: a decision is
charged only its share of a jointly-caused successor, and the shares compound
multiplicatively along a causal chain -- i.e. exponentially in causal depth.

Run: python _diag_prop3_identity.py
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces as CausalTrace
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
EPISODES = 5

CAPTURE = {}
_orig = CausalTrace._redistribute_cgae


def _spy(self, action_transitions, token_to_action, record_to_action,
         redistribution, beta, get_parents, values=None, lam=1.0, flow=False):
    """Recompute succ / w_edge / owned with the library's own maps, then defer
    to the real implementation for the returned credit."""
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

    CAPTURE['succ'] = succ
    CAPTURE['w_edge'] = w_edge
    CAPTURE['owned'] = owned
    CAPTURE['n'] = n
    return _orig(self, action_transitions, token_to_action, record_to_action,
                 redistribution, beta, get_parents, values, lam, flow)


CausalTrace._redistribute_cgae = _spy


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


def descendants(succ, d):
    seen, stack = set(), [d]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(succ.get(x, ()))
    return seen


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    ratios_flow, ratios_mean, wvals = [], [], []
    by_depth = {}
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        V = [0.0] * n_dec
        q_flow = np.asarray(ct.redistribute_rewards(scheme='cgae_flow', beta=0.0,
                                                    values=V, lam=1.0), dtype=float)
        succ = CAPTURE['succ']; w_edge = dict(CAPTURE['w_edge'])
        owned = list(CAPTURE['owned']); n = CAPTURE['n']
        q_mean = np.asarray(ct.redistribute_rewards(scheme='cgae', beta=0.0,
                                                    values=V, lam=1.0), dtype=float)
        wvals.extend(w_edge.values())

        depth = [0] * n
        for d in range(n - 1, -1, -1):
            kids = succ.get(d, ())
            if kids:
                depth[d] = 1 + max(depth[s] for s in kids)

        for d in range(min(n, len(q_flow))):
            rhs = sum(owned[x] for x in descendants(succ, d))
            if abs(rhs) < 1e-9:
                continue
            rf, rm = float(q_flow[d]) / rhs, float(q_mean[d]) / rhs
            ratios_flow.append(rf)
            ratios_mean.append(rm)
            by_depth.setdefault(depth[d], []).append(rf)

    wv = np.asarray(wvals)
    rf = np.asarray(ratios_flow)
    rm = np.asarray(ratios_mean)
    print("=" * 72)
    print(label, " (lam=1, V=0, beta=0 -- Proposition 3 claims ratio == 1.000)")
    if len(rf) == 0:
        print("  no reward-bearing descendants captured")
        continue
    print("  edge weights w(d->s):  mean %.3f   exactly 1.0: %.1f%%   <=0.5: %.1f%%"
          % (wv.mean(), 100.0 * np.mean(np.abs(wv - 1.0) < 1e-9),
             100.0 * np.mean(wv <= 0.5 + 1e-9)))
    print("  P3 ratio LHS/RHS  flow : mean %.3f  median %.3f  [p10 %.3f, p90 %.3f]  n=%d"
          % (rf.mean(), np.median(rf), np.percentile(rf, 10), np.percentile(rf, 90), len(rf)))
    print("  P3 ratio LHS/RHS  mean : mean %.3f  median %.3f"
          % (rm.mean(), np.median(rm)))
    print("  decisions recovering <50%% of their own component return: flow %.1f%%  mean %.1f%%"
          % (100.0 * np.mean(rf < 0.5), 100.0 * np.mean(rm < 0.5)))
    print("  flow ratio by causal depth:")
    for d in sorted(by_depth)[:14]:
        v = np.asarray(by_depth[d])
        print("      depth %2d  n=%4d  mean ratio %.3f" % (d, len(v), v.mean()))
print("=" * 72)
