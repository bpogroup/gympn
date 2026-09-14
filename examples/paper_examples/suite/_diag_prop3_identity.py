r"""Decisive test of Proposition 3(i)/(iii): how close is each cgae variant to
the closure estimand?

With lam=1, V==0, beta=0 the proposition claims

    A*[d] = sum_{j : own(j) in desc*(d)} owned[j]                        (P3)

so each descendant's owned reward carries total weight 1 across all paths.

  LHS  what `redistribute_rewards(scheme=...)` returns
  RHS  sum of `owned` over the descendant closure of d (including d)

REWRITTEN 2026-09-14. The previous version had been dead for some time: its
spy wrapper's signature stopped at `flow=`, so once `_redistribute_cgae` grew
`convex=`, `skip_postpone=` and `cap=` the dispatcher's keyword call raised
TypeError on EVERY cgae scheme. It could not have produced the figures quoted
from it (flow 0.577 / mean 0.602 / cflow 0.626 / dag 1.000). Three further
defects are fixed here:

  1. it re-derived succ/w_edge/owned in the diagnostic instead of using the
     library's, despite a docstring claiming the opposite. We now call the
     library's own `_cgae_structure`, and CHECK it against the successor set
     the mean branch builds by its separate pooled walk;
  2. it scored the mean variant's LHS against the flow variant's closure;
  3. it reported a mean of per-decision ratios, which mixes leaf decisions
     (ratio identically 1 -- no successors, so LHS == RHS == owned[d]) with
     interior ones, so the headline number tracks the leaf fraction as much as
     any dilution. We now report leaves and interior nodes separately, plus an
     aggregate mass ratio sum(LHS)/sum(RHS) that no single tiny denominator can
     swing.

NOTE ON WHAT THIS CAN AND CANNOT SHOW. `cgae_dag` computes the closure sum by
construction (see `_redistribute_cgae_dag`: "At lam=1 the bootstrap vanishes
and Q is precisely (P3)"), so its ratio is 1.000 as an identity, not as a
measurement. This diagnostic therefore measures AGREEMENT WITH `cgae_dag`, not
an independent notion of causal fidelity. `cgae_cflow2` is excluded: it changes
the DAG itself, so its estimand is a different object.

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
SCHEMES = ("cgae_flow", "cgae", "cgae_cflow", "cgae_cap", "cgae_dag")

CAPTURE = {}
_orig_cgae = CausalTrace._redistribute_cgae
_orig_dag = CausalTrace._redistribute_cgae_dag


def _grab(self, action_transitions, token_to_action, record_to_action, get_parents):
    """Structure from the LIBRARY's own shared constructor -- not re-derived."""
    succ, w_edge, owned, times = self._cgae_structure(
        action_transitions, token_to_action, record_to_action, get_parents)
    CAPTURE['succ'] = succ
    CAPTURE['w_edge'] = dict(w_edge)
    CAPTURE['owned'] = list(owned)
    CAPTURE['n'] = len(action_transitions)
    # Cross-check: the mean branch builds succ by a POOLED backward walk with a
    # shared `seen`, not the per-token walk `_cgae_structure` uses. If those
    # disagree, one closure cannot serve both variants.
    out_tok = {}
    for idx, act in enumerate(action_transitions):
        for t in act.get('output_tokens', ()) or ():
            out_tok[t] = idx
    pooled = {}
    for idx, act in enumerate(action_transitions):
        seen, stack = set(), list(act.get('input_tokens', ()) or ())
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            src = out_tok.get(tid)
            if src is not None and src != idx:
                pooled.setdefault(src, set()).add(idx)
                continue
            for p in get_parents(tid):
                if p not in seen:
                    stack.append(p)
    CAPTURE['pooled_matches'] = ({k: set(v) for k, v in pooled.items() if v}
                                 == {k: set(v) for k, v in succ.items() if v})


def _spy_cgae(self, action_transitions, token_to_action, record_to_action,
              redistribution, beta, get_parents, *a, **kw):
    _grab(self, action_transitions, token_to_action, record_to_action, get_parents)
    return _orig_cgae(self, action_transitions, token_to_action, record_to_action,
                      redistribution, beta, get_parents, *a, **kw)


def _spy_dag(self, action_transitions, token_to_action, record_to_action,
             redistribution, beta, get_parents, *a, **kw):
    _grab(self, action_transitions, token_to_action, record_to_action, get_parents)
    return _orig_dag(self, action_transitions, token_to_action, record_to_action,
                     redistribution, beta, get_parents, *a, **kw)


CausalTrace._redistribute_cgae = _spy_cgae
CausalTrace._redistribute_cgae_dag = _spy_dag


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
    acc = {s: {'leaf': [], 'int': [], 'lhs': 0.0, 'rhs': 0.0} for s in SCHEMES}
    pooled_ok, neg_rhs, tiny_rhs, total = True, 0, 0, 0
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        V = [0.0] * n_dec
        q = {}
        for s in SCHEMES:
            q[s] = np.asarray(ct.redistribute_rewards(scheme=s, beta=0.0,
                                                      values=V, lam=1.0), dtype=float)
        succ, owned, n = CAPTURE['succ'], CAPTURE['owned'], CAPTURE['n']
        pooled_ok = pooled_ok and CAPTURE['pooled_matches']
        for d in range(min(n, min(len(v) for v in q.values()))):
            rhs = sum(owned[x] for x in descendants(succ, d))
            total += 1
            if rhs < 0:
                neg_rhs += 1
            if abs(rhs) < 1e-9:
                tiny_rhs += 1
                continue
            is_leaf = not [x for x in succ.get(d, ()) if x != d]
            for s in SCHEMES:
                acc[s]['leaf' if is_leaf else 'int'].append(float(q[s][d]) / rhs)
                acc[s]['lhs'] += float(q[s][d])
                acc[s]['rhs'] += rhs

    print("=" * 78)
    print(label, "  (lam=1, V=0, beta=0; closure estimand == what cgae_dag computes)")
    print("  pooled-walk succ == _cgae_structure succ : %s" % pooled_ok)
    print("  decisions %d | zero-RHS skipped %d | NEGATIVE RHS %d" % (total, tiny_rhs, neg_rhs))
    print("  %-11s %9s %9s | %9s %7s | %9s" %
          ("scheme", "int mean", "int med", "leaf mean", "n_leaf", "mass"))
    for s in SCHEMES:
        iv = np.asarray(acc[s]['int'])
        lv = np.asarray(acc[s]['leaf'])
        mass = acc[s]['lhs'] / acc[s]['rhs'] if acc[s]['rhs'] else float('nan')
        print("  %-11s %9.3f %9.3f | %9.3f %7d | %9.3f" %
              (s, iv.mean() if len(iv) else float('nan'),
               np.median(iv) if len(iv) else float('nan'),
               lv.mean() if len(lv) else float('nan'), len(lv), mass))
    ni, nl = len(acc['cgae_flow']['int']), len(acc['cgae_flow']['leaf'])
    print("  interior decisions: %d of %d (%.1f%%)"
          % (ni, ni + nl, 100.0 * ni / max(1, ni + nl)))
print("=" * 78)
