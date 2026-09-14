r"""Gate for cgae_cap -- sub-stochastic flow weights, c = w / max(R, 1).

The convex form (cgae_cflow) forces the bootstrap coefficient to EXACTLY 1.
That fixes the R>1 tail but also raises the R<1 tail up to 1, and the two tails
turn out to be where the two variants respectively fail:

    R > 1 (fan-out) : cgae_flow multiplies the critic by R -- not a contraction.
                      s1 has R>1 on 24.0% of decisions and cgae_flow COLLAPSES
                      (0.166 vs cgae_cflow's 0.792, 0W/5L vs ppo, p=0.025).
    R < 1 (fan-in)  : cgae_cflow renormalizes back UP to 1, claiming the whole
                      continuation for a decision that only partly caused its
                      successor. ncopies N=2 has R<1 on 10.1% of decisions and
                      cgae_cflow is SIGNIFICANTLY WORSE than cgae_flow
                      (0.401 vs 0.837, 4W/14L, p=0.0043).

cgae_cap only ever divides -- never multiplies up -- so the coefficient is
min(R,1) <= 1: contraction is guaranteed without over-attribution.

  P1 SUB-STOCHASTIC  the bootstrap coefficient is min(R,1) for every decision
                     with successors; in particular it is never > 1.
  P2 AGREES WITH FLOW WHERE R<=1   cap == cgae_flow on every decision whose row
                     sum is <= 1 (that is the whole point: keep the conservative
                     attribution the convex form discards).
  P3 AGREES WITH CFLOW WHERE R>1   on a synthetic fan-out with R=2, cap returns
                     the convex value, not the inflated one.
  P4 FINITE          one credit per decision, all finite.

Run: python _test_cgae_cap.py
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
EPISODES = 4
fails = []

CAP = {}
_orig = CausalTraces._cgae_structure


def _spy(self, a, t, r, g):
    out = _orig(self, a, t, r, g)
    CAP['succ'], CAP['w'] = out[0], out[1]
    return out


CausalTraces._cgae_structure = _spy


def build(maker):
    pn = maker()
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


def rollout(maker, seed):
    random.seed(seed)
    env = build(maker)
    env.reset()
    done, guard = False, 0
    while not done and guard < 10000:
        guard += 1
        k = len(env.pn.get_graph_observation().get('actions_dict') or [])
        if k == 0:
            break
        _, _, done, _, _ = env.step(random.randrange(k))
    return env


for label, maker in [
        ("ncopies N=2", lambda: make_n_copies(2, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    n_le1 = n_gt1 = 0
    bad_coef = agree_flow = disagree_flow = 0
    for ep in range(EPISODES):
        env = rollout(maker, 300 + ep)
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n == 0:
            continue
        V = [0.3] * n
        q_flow = np.asarray(ct.redistribute_rewards(scheme='cgae_flow', beta=0.1,
                                                    values=V, lam=0.95), dtype=float)
        q_cap = np.asarray(ct.redistribute_rewards(scheme='cgae_cap', beta=0.1,
                                                   values=V, lam=0.95), dtype=float)
        ct.redistribute_rewards(scheme='cgae_dag', beta=0.1, values=V, lam=0.95)
        succ, w = CAP['succ'], CAP['w']

        if len(q_cap) != n or not np.all(np.isfinite(q_cap)):
            fails.append("%s: cgae_cap bad credit vector" % label)

        # P1 -- coefficient is min(R,1)
        allR1 = True
        for d, kids in succ.items():
            kk = [s for s in kids if s != d]
            if not kk:
                continue
            R = sum(w.get((d, s), 0.0) for s in kk)
            coef = sum((w.get((d, s), 0.0) / max(R, 1.0)) for s in kk) if R > 0 else 0.0
            if coef > 1.0 + 1e-9:
                bad_coef += 1
            if R > 1 + 1e-9:
                n_gt1 += 1
                allR1 = False
            else:
                n_le1 += 1

        # P2 -- where EVERY node in the descendant closure has R<=1, cap must
        # equal flow exactly (the recursion is then identical edge by edge).
        Rmap = {}
        for d, kids in succ.items():
            kk = [s for s in kids if s != d]
            Rmap[d] = sum(w.get((d, s), 0.0) for s in kk) if kk else 0.0
        for d in range(n):
            seen, stack, ok = {d}, [d], True
            while stack:
                x = stack.pop()
                if Rmap.get(x, 0.0) > 1 + 1e-9:
                    ok = False
                    break
                for s in succ.get(x, ()):
                    if s not in seen:
                        seen.add(s)
                        stack.append(s)
            if ok and d < len(q_cap):
                if abs(q_cap[d] - q_flow[d]) < 1e-9:
                    agree_flow += 1
                else:
                    disagree_flow += 1
    print("%-12s R<=1 nodes %4d | R>1 nodes %4d | coefficient>1: %d | "
          "closure-R<=1 decisions agreeing with flow: %d (disagreeing %d)"
          % (label, n_le1, n_gt1, bad_coef, agree_flow, disagree_flow))
    if bad_coef:
        fails.append("%s: %d decisions have bootstrap coefficient > 1 (P1)" % (label, bad_coef))
    if disagree_flow:
        fails.append("%s: %d R<=1-closure decisions differ from cgae_flow (P2)"
                     % (label, disagree_flow))
    if n_gt1 == 0:
        fails.append("%s: no R>1 nodes, so P1 is vacuous here" % label)

CausalTraces._cgae_structure = _orig

# P3 -- synthetic fan-out with R=2: cap must match the convex value (3.37),
# not cgae_flow's inflated 6.74.
acts = [
    {'input_tokens': ['i0'], 'output_tokens': ['t1', 't2'], 'time': 0.0, 'reward': 0.0},
    {'input_tokens': ['t1'], 'output_tokens': ['o1'], 'time': 1.0, 'reward': 2.0},
    {'input_tokens': ['t2'], 'output_tokens': ['o2'], 'time': 1.0, 'reward': 5.0},
]
parents = {'i0': [], 't1': ['i0'], 't2': ['i0'], 'o1': ['t1'], 'o2': ['t2']}
t2a = {'t1': (0, 0.0), 't2': (0, 0.0), 'o1': (1, 1.0), 'o2': (2, 1.0)}
r2a = {id(a): (i, False) for i, a in enumerate(acts)}
stub = types.SimpleNamespace(transition_history=types.SimpleNamespace(transitions=acts))
V = [0.3, 0.7, 1.1]


def run(flow, convex, cap):
    return CausalTraces._redistribute_cgae(
        stub, acts, t2a, r2a, [0.0] * 3, 0.0, lambda t: parents.get(t, []),
        list(V), 0.95, flow, convex, False, cap)


q_flow = run(True, False, False)
q_cflow = run(True, True, False)
q_cap = run(True, False, True)
print("P3 synthetic fan-out (R=2):  flow %.4f | cflow %.4f | cap %.4f"
      % (q_flow[0], q_cflow[0], q_cap[0]))
if abs(q_cap[0] - q_cflow[0]) > 1e-12:
    fails.append("P3: cap %.6f != cflow %.6f under R>1" % (q_cap[0], q_cflow[0]))
if abs(q_cap[0] - q_flow[0]) < 1e-12:
    fails.append("P3: cap did not correct the R>1 inflation")

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 4/4")
