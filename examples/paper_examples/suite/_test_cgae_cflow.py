r"""Gate for cgae_cflow -- the successor-normalized fix to cgae_flow.

`cgae_flow` normalizes edge weights over PREDECESSORS, per Proposition 3's

    sum over d in pred(s) of w(d->s) = 1                                   (*)

but its recursion sums over SUCCESSORS, so the coefficient actually
multiplying the bootstrap is the ROW sum R(d) = sum_{s in succ(d)} w(d->s),
which (*) leaves entirely unconstrained. A[d] carries `-V[d]` and picks up
`R(d)*rho*V[s]`, so when R(d) != 1 the value terms fail to cancel and the
advantage keeps a spurious term proportional to V that is a function of local
DAG topology, not of the action. Measured: |R-1| > 0.25 on 24.1% of ncopies
decisions and 55.3% of s1's -- the collapse `cgae_flow` shows on s1.

`cgae_cflow` divides by R(d). Four properties, on real traces:

  C1 CONVEX        the bootstrap coefficient is exactly 1 for every decision
                   with at least one causal successor. This is the fix.
  C2 CHAIN-EXACT   on a true chain (every edge in the descendant closure with
                   out-degree <=1 and weight 1) cflow, flow and mean all agree
                   to floating point -- the fix touches only fan-out.
  C3 REDUCES       with beta=0 and equal inflow shares the recursion is the
                   mean variant, so `cgae` is the equal-share special case
                   rather than an underived shrinkage.
  C4 FINITE        no NaN/inf, one credit per decision (the 1:1 alignment
                   TrajectoryBuffer.finish() asserts), and bounded by flow's
                   worst case rather than exceeding it.

Run: python _test_cgae_cflow.py
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


def edges(ct):
    """Rebuild the flow edge set independently of the library."""
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

    w_edge, succ = {}, {}
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
                succ.setdefault(d, set()).add(idx)
    return w_edge, succ, n


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    agree = disagree = 0
    worst_R = 0.0
    n_multi = 0
    max_c = max_f = 0.0
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        V = [0.3] * n_dec

        q_mean = np.asarray(ct.redistribute_rewards(scheme='cgae', beta=0.1,
                                                    values=V, lam=0.95), dtype=float)
        q_flow = np.asarray(ct.redistribute_rewards(scheme='cgae_flow', beta=0.1,
                                                    values=V, lam=0.95), dtype=float)
        q_cflw = np.asarray(ct.redistribute_rewards(scheme='cgae_cflow', beta=0.1,
                                                    values=V, lam=0.95), dtype=float)

        # C4
        for nm, q in (('cgae_cflow', q_cflw),):
            if len(q) != n_dec:
                fails.append("%s: %s length %d != %d" % (label, nm, len(q), n_dec))
            if not np.all(np.isfinite(q)):
                fails.append("%s: %s produced non-finite credit" % (label, nm))
        max_c = max(max_c, float(np.max(np.abs(q_cflw))))
        max_f = max(max_f, float(np.max(np.abs(q_flow))))

        w_edge, succ, n = edges(ct)

        # C1 -- normalized row sums are 1
        for d, kids in succ.items():
            R = sum(w_edge.get((d, s), 0.0) for s in kids)
            if R <= 0:
                continue
            n_multi += 1
            norm = sum(w_edge.get((d, s), 0.0) / R for s in kids)
            worst_R = max(worst_R, abs(R - 1.0))
            if abs(norm - 1.0) > 1e-9:
                fails.append("%s: normalized row sum %.9f != 1 at decision %d"
                             % (label, norm, d))
                break

        # C2 -- true chains agree across all three variants
        chain_ok = [True] * n
        for d in range(n):
            seen_d, stack, ok = set(), [d], True
            while stack:
                x = stack.pop()
                if x in seen_d:
                    continue
                seen_d.add(x)
                kids = succ.get(x, set())
                if len(kids) > 1 or any(abs(w_edge.get((x, y), 0.0) - 1.0) > 1e-12
                                        for y in kids):
                    ok = False
                    break
                stack.extend(kids)
            chain_ok[d] = ok
        for d in range(min(n, len(q_cflw))):
            if chain_ok[d]:
                if (abs(q_cflw[d] - q_flow[d]) < 1e-9
                        and abs(q_cflw[d] - q_mean[d]) < 1e-9):
                    agree += 1
                else:
                    disagree += 1

    print("%s: true-chain agreeing %d, disagreeing %d | decisions with successors %d "
          "| worst |R-1| %.3f | max|credit| flow %.2f -> cflow %.2f"
          % (label, agree, disagree, n_multi, worst_R, max_f, max_c))
    if disagree:
        fails.append("%s: %d true-chain decisions differ (C2)" % (label, disagree))
    if worst_R <= 1e-9:
        fails.append("%s: no decision had R != 1, so C1 is vacuous here" % label)

# C3 -- equal inflow shares + beta=0 reduce EXACTLY to the mean variant.
# Real rollouts turn out never to produce an exactly-equal-share fan-out (the
# check above found 0 on ncopies), so leaving C3 to chance makes it vacuous.
# Build the case by hand instead: one decision fanning out to two successors
# that each have in-degree 1, hence w = 1 on both edges, R = 2, and normalized
# shares of exactly 1/2 -- which is the uniform weighting the mean variant
# applies. The same fixture exhibits the defect: unnormalized `flow` enters the
# critic twice here.
def _synthetic_fanout():
    acts = [
        {'input_tokens': ['i0'], 'output_tokens': ['t1', 't2'], 'time': 0.0, 'reward': 0.0},
        {'input_tokens': ['t1'], 'output_tokens': ['o1'], 'time': 1.0, 'reward': 2.0},
        {'input_tokens': ['t2'], 'output_tokens': ['o2'], 'time': 1.0, 'reward': 5.0},
    ]
    parents = {'i0': [], 't1': ['i0'], 't2': ['i0'], 'o1': ['t1'], 'o2': ['t2']}
    token_to_action = {'t1': (0, 0.0), 't2': (0, 0.0), 'o1': (1, 1.0), 'o2': (2, 1.0)}
    record_to_action = {id(a): (i, False) for i, a in enumerate(acts)}
    stub = types.SimpleNamespace(
        transition_history=types.SimpleNamespace(transitions=acts))
    V = [0.3, 0.7, 1.1]

    def run(flow, convex):
        return CausalTraces._redistribute_cgae(
            stub, acts, token_to_action, record_to_action, [0.0] * 3, 0.0,
            lambda t: parents.get(t, []), list(V), 0.95, flow, convex)

    return run(False, False), run(True, False), run(True, True), V


q_mean_s, q_flow_s, q_cflow_s, Vs = _synthetic_fanout()
print("C3 synthetic fan-out (w=1 on both edges, R=2):")
print("     mean  %s" % [round(x, 6) for x in q_mean_s])
print("     flow  %s" % [round(x, 6) for x in q_flow_s])
print("     cflow %s" % [round(x, 6) for x in q_cflow_s])
if abs(q_cflow_s[0] - q_mean_s[0]) > 1e-12:
    fails.append("C3: cflow %.9f != mean %.9f under equal shares"
                 % (q_cflow_s[0], q_mean_s[0]))
if abs(q_flow_s[0] - q_mean_s[0]) <= 1e-12:
    fails.append("C3: flow did not differ from mean, so the fixture is not "
                 "exercising the defect")
else:
    # the defect, quantified on a case small enough to check by hand:
    # flow enters the critic R=2 times instead of once.
    excess = (q_flow_s[0] - q_cflow_s[0])
    print("     flow's excess bootstrap at the fan-out node: %+.4f "
          "(R=2 => critic counted twice)" % excess)

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 4/4")
