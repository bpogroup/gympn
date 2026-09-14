r"""Offline prototype of the cgae_flow fix, before touching the library.

Two defects are separated here.

D1  UNANCHORED BOOTSTRAP (flow only -- this is what distinguishes flow from
    mean). The recursion multiplies the successor-value term by

        R(d) = sum_{s in succ(d)} w(d->s)

    which Proposition 3's constraint (sum over PREDECESSORS = 1) does not pin
    down. A TD backup needs that coefficient to be a convex combination. The
    mean variant has it equal to 1 by construction; flow lets it wander.

D2  REWARD DILUTION AT FAN-IN (both variants). w(d->s) = 0.5 when s is jointly
    caused, so the lam=1,V=0 estimator recovers only a fraction of the
    component return, decaying multiplicatively with causal depth.

The proposed fix addresses D1 by normalizing the weights over SUCCESSORS for
the recursion, keeping the inflow shares only as relative weights:

    what(d->s) = w(d->s) / R(d)          sum over s of what = 1
    A[d] = owned[d] - V[d] + sum_s what(d->s) rho_s ( V[s] + lam A[s] )

On a chain what == 1 (unchanged). Under uniform inflow shares what == 1/k,
i.e. exactly the mean variant -- so the mean stops being an underived
shrinkage and becomes the equal-share special case.

Reports, per env: spread of R(d); and the P3 ratio under current-flow,
mean, and fixed-flow.

Run: python _diag_cgae_fix_proto.py
"""
import os, sys, random, types, uuid, math
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
         redistribution, beta, get_parents, values=None, lam=1.0, flow=False):
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
                 redistribution, beta, get_parents, values, lam, flow)


CausalTraces._redistribute_cgae = _spy


def recur(succ, w_edge, owned, times, n, V, beta, lam, mode):
    """Reference recursions. mode in {'flow','mean','fixed'}."""
    def disc(dt):
        if beta == 0.0 or dt is None:
            return 1.0
        return math.exp(-beta * max(0.0, float(dt)))

    order = sorted(range(n), key=lambda i: (times[i] is not None, times[i] or 0.0),
                   reverse=True)
    A = [0.0] * n
    for d in order:
        nxt = [s for s in succ.get(d, ()) if s != d]
        if not nxt:
            A[d] = owned[d] - V[d]
            continue
        dts = {s: ((times[s] - times[d]) if (times[s] is not None and times[d] is not None)
                   else 0.0) for s in nxt}
        if mode == 'flow':
            v_next = sum(w_edge.get((d, s), 0.0) * disc(dts[s]) * V[s] for s in nxt)
            a_next = sum(w_edge.get((d, s), 0.0) * disc(dts[s]) * A[s] for s in nxt)
            A[d] = owned[d] + v_next - V[d] + lam * a_next
        elif mode == 'mean':
            g = sum(disc(dt) for dt in dts.values()) / len(nxt)
            A[d] = (owned[d] + g * sum(V[s] for s in nxt) / len(nxt) - V[d]
                    + g * lam * sum(A[s] for s in nxt) / len(nxt))
        else:  # fixed: outflow-normalized flow weights
            R = sum(w_edge.get((d, s), 0.0) for s in nxt)
            if R <= 0:
                A[d] = owned[d] - V[d]
                continue
            v_next = sum(w_edge.get((d, s), 0.0) / R * disc(dts[s]) * V[s] for s in nxt)
            a_next = sum(w_edge.get((d, s), 0.0) / R * disc(dts[s]) * A[s] for s in nxt)
            A[d] = owned[d] + v_next - V[d] + lam * a_next
    return A


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


def desc_of(succ, d):
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
    Rs = []
    rat = {'flow': [], 'mean': [], 'fixed': []}
    # realistic training regime: nonzero critic, lam<1, SMDP discount on
    scale = {'ncopies N=4': 1.5, 's1': 0.5}[label]
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        ct.redistribute_rewards(scheme='cgae_flow', beta=0.0,
                                values=[0.0] * n_dec, lam=1.0)
        succ, w_edge = CAPTURE['succ'], CAPTURE['w_edge']
        owned, n, times = CAPTURE['owned'], CAPTURE['n'], CAPTURE['times']
        for d in succ:
            Rs.append(sum(w_edge.get((d, s), 0.0) for s in succ[d]))
        V = [0.0] * n
        res = {m: recur(succ, w_edge, owned, times, n, V, 0.0, 1.0, m)
               for m in ('flow', 'mean', 'fixed')}
        for d in range(n):
            rhs = sum(owned[x] for x in desc_of(succ, d))
            if abs(rhs) < 1e-9:
                continue
            for m in rat:
                rat[m].append(res[m][d] / rhs)

    R = np.asarray(Rs)
    print("=" * 74)
    print(label)
    print("  bootstrap coefficient R(d) = sum_succ w   (a TD backup needs ~1)")
    print("      mean %.3f   SD %.3f   min %.3f   max %.3f   |R-1|>0.25: %.1f%%"
          % (R.mean(), R.std(), R.min(), R.max(), 100.0 * np.mean(np.abs(R - 1) > 0.25)))
    print("  P3 ratio (lam=1, V=0)   ideal 1.000")
    for m in ('flow', 'mean', 'fixed'):
        v = np.asarray(rat[m])
        print("      %-6s mean %.3f  median %.3f  SD %.3f" % (m, v.mean(), np.median(v), v.std()))
print("=" * 74)
