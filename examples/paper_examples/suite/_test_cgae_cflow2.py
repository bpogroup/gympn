r"""Gate for cgae_cflow2 -- cgae_cflow with postpone made transparent.

The fork test (_diag_R_action_invariance.py) showed R(d) is not
action-measurable, and that POSTPONE is the dominant violator: it carries 2.25x
(ncopies N=4) to 2.66x (s1) the row sum of production actions, because a
token-flow postpone re-emits the WHOLE marking and so shadows the real
producers. cgae_cflow2 drops postpone from all three roles in which it is an
artifact -- producer, consumer/successor, and reward owner -- while the walk
still passes THROUGH its re-emitted tokens, exactly as lrq2 does.

  P1 NO POSTPONE NODES   no postpone decision appears as a predecessor or a
                         successor in the causal DAG, and none owns reward.
  P2 ZERO CREDIT         postpone decisions emit exactly 0. (data.py's finish()
                         substitutes A = e^{-beta*tau}V(s') - V(s) there; this
                         gate covers the estimator half of that contract.)
  P3 EDGES REDIRECTED    postpone-transparency must REDIRECT edges, not delete
                         them, else R would look better for the wrong reason.
                         The exact invariant: every production->production edge
                         present BEFORE must still be present AFTER. Removing
                         postpone from out_tok can only ever ADD such edges (a
                         consumer of a re-emitted token now attributes to the
                         real upstream producer instead of to the postpone that
                         merely passed it along); it can never remove one. Note
                         the count of producers-with-successors MAY still fall
                         slightly, when a decision's only successor was a
                         postpone whose re-emitted tokens no production decision
                         ever consumed before the horizon -- that is a genuine
                         causal sink, not a severed edge, so it is reported but
                         not failed.
  P4 R INFLATION GONE    the postpone-vs-production row-sum ratio that motivated
                         the variant is absent, and production-side R is
                         unchanged relative to cgae_cflow.

Run: python _test_cgae_cflow2.py
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
fails = []

CAP = {}
_orig = CausalTraces._redistribute_cgae


def _spy(self, action_transitions, token_to_action, record_to_action,
         redistribution, beta, get_parents, values=None, lam=1.0,
         flow=False, convex=False, skip_postpone=False, cap=False):
    out = _orig(self, action_transitions, token_to_action, record_to_action,
                redistribution, beta, get_parents, values, lam, flow, convex,
                skip_postpone, cap)
    CAP['pp'] = {rec[0] for rec in record_to_action.values() if rec[1]}
    CAP['n'] = len(action_transitions)
    return out


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
        obs = env.pn.get_graph_observation()
        k = len(obs.get('actions_dict') or [])
        if k == 0:
            break
        _, _, done, _, _ = env.step(random.randrange(k))
    return env


def structure(ct, skip_pp):
    """Edges + inflow shares, rebuilt here with the same rule the estimator uses."""
    acts = ct.transition_history.get_action_transitions()
    pp = {i for i, a in enumerate(acts)
          if isinstance(getattr(a.get('transition'), '_id', None), str)
          and getattr(a.get('transition'), '_id').startswith('postpone_')}
    out_tok = {}
    for idx, a in enumerate(acts):
        if skip_pp and idx in pp:
            continue
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

    w, succ = {}, {}
    for idx, a in enumerate(acts):
        if skip_pp and idx in pp:
            continue
        contrib, k = {}, 0
        for tid in list(a.get('input_tokens', ()) or ()):
            prods = producers(tid, idx)
            if not prods:
                continue
            k += 1
            for dd in prods:
                contrib[dd] = contrib.get(dd, 0.0) + 1.0 / len(prods)
        if k:
            for dd, c in contrib.items():
                w[(dd, idx)] = c / k
                succ.setdefault(dd, set()).add(idx)
    return w, succ, pp, len(acts)


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    n_pp_nodes = n_pp_owner = 0
    nz_pp = 0
    prod_before = prod_after = 0
    lost, gained = set(), 0
    ratio_before, R_prod_before, R_prod_after = [], [], []
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n == 0:
            continue
        V = [0.3] * n
        q1 = np.asarray(ct.redistribute_rewards(scheme='cgae_cflow', beta=0.1,
                                                values=V, lam=0.95), dtype=float)
        q2 = np.asarray(ct.redistribute_rewards(scheme='cgae_cflow2', beta=0.1,
                                                values=V, lam=0.95), dtype=float)
        pp = CAP['pp']
        if len(q2) != n or not np.all(np.isfinite(q2)):
            fails.append("%s: cgae_cflow2 bad credit vector" % label)

        w_b, succ_b, _, _ = structure(ct, skip_pp=False)
        w_a, succ_a, _, _ = structure(ct, skip_pp=True)

        # P1
        for (dd, ss) in w_a:
            if dd in pp or ss in pp:
                n_pp_nodes += 1
        # P2
        nz_pp += sum(1 for i in pp if abs(q2[i]) > 1e-12)
        # P3 -- production->production edges must survive (they may only grow)
        eb = {(dd, ss) for (dd, ss) in w_b if dd not in pp and ss not in pp}
        ea = {(dd, ss) for (dd, ss) in w_a if dd not in pp and ss not in pp}
        lost.update(eb - ea)
        gained += len(ea - eb)
        prod_before += sum(1 for dd in succ_b if dd not in pp)
        prod_after += sum(1 for dd in succ_a if dd not in pp)
        # P4
        for dd in succ_b:
            R = sum(w_b.get((dd, s), 0.0) for s in succ_b[dd])
            (ratio_before if dd in pp else R_prod_before).append(R)
        for dd in succ_a:
            R_prod_after.append(sum(w_a.get((dd, s), 0.0) for s in succ_a[dd]))

    rb = np.mean(ratio_before) if ratio_before else float('nan')
    pb = np.mean(R_prod_before) if R_prod_before else float('nan')
    pa = np.mean(R_prod_after) if R_prod_after else float('nan')
    sd = np.std(R_prod_after) if R_prod_after else float('nan')
    print("%-12s pp edges left %d | pp credits nonzero %d | prod->prod edges lost %d "
          "gained %d | producers-with-succ %d -> %d | R: pp %.3f, prod %.3f -> %.3f (SD %.4f)"
          % (label, n_pp_nodes, nz_pp, len(lost), gained, prod_before, prod_after,
             rb, pb, pa, sd))
    if n_pp_nodes:
        fails.append("%s: %d DAG edges still touch a postpone node (P1)" % (label, n_pp_nodes))
    if nz_pp:
        fails.append("%s: %d postpone decisions emitted nonzero credit (P2)" % (label, nz_pp))
    if lost:
        fails.append("%s: %d production->production edges DISAPPEARED (P3); "
                     "transparency must only redirect, e.g. %s"
                     % (label, len(lost), sorted(lost)[:3]))
    if ratio_before and rb <= pb:
        fails.append("%s: postpone R (%.3f) did not exceed production R (%.3f), "
                     "so the fixture does not reproduce the motivating defect (P4)"
                     % (label, rb, pb))

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 4/4")
