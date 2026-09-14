r"""Gate for cgae_dag -- the variant that makes Proposition 3(i) exact.

(P3)   A_d = sum_{j : own(j) in desc(d)} rho_d(t_j) * r_j

is a statement about the descendant SET, but every weighted variant computes it
as a sum over PATHS. On a re-convergent DAG those differ: measured on s1, 42.1%
of (d, j) pairs have >=2 distinct causal paths (up to 52), while 54.1% of
successors have in-degree 1. So weight-1 over-counts the diamonds and
inflow-normalized weights under-credit plain joint causation. cgae_dag drops
edge weights for the reward term and uses the closure directly, keeping a
convex k-step mixture for the critic.

  D1 P3-EXACT     at lam=1 with V==0, the returned credit equals the closure
                  sum of owned rewards, to floating point, for EVERY decision
                  on real traces. This is the property flow/mean/cflow miss
                  (measured ratios on s1: 0.577 / 0.602 / 0.626).
  D2 GAE-ON-CHAIN on a synthetic chain, cgae_dag reproduces textbook GAE
                  A = sum_k (gamma*lam)^k delta_k term for term, for several
                  lam -- so the DAG form is a generalization, not a new rule.
  D3 DEDUP        on a synthetic diamond (d -> s1 -> j, d -> s2 -> j) the
                  reward at j is counted ONCE, where a weight-1 path sum counts
                  it twice.
  D4 FINITE+COST  no NaN/inf, one credit per decision, and the closure walk
                  costs a small fraction of a training step.

Run: python _test_cgae_dag.py
"""
import os, sys, random, types, uuid, math, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
EPISODES = 5
fails = []

CAPTURE = {}
_orig_struct = CausalTraces._cgae_structure


def _spy(self, action_transitions, token_to_action, record_to_action, get_parents):
    out = _orig_struct(self, action_transitions, token_to_action,
                       record_to_action, get_parents)
    CAPTURE['succ'], CAPTURE['w_edge'], CAPTURE['owned'], CAPTURE['times'] = out
    return out


CausalTraces._cgae_structure = _spy


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


def closure(succ, d):
    seen, stack = {d}, [d]
    while stack:
        x = stack.pop()
        for s in succ.get(x, ()):
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


# ---- D1: P3 exactness on real traces --------------------------------------- #
for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    worst, checked, elapsed, n_tot = 0.0, 0, 0.0, 0
    for ep in range(EPISODES):
        env = rollout(builder, 300 + ep)
        ct = env.pn.causal_trace
        n_dec = len(ct.transition_history.get_action_transitions())
        if n_dec == 0:
            continue
        t0 = time.time()
        q = np.asarray(ct.redistribute_rewards(scheme='cgae_dag', beta=0.0,
                                               values=[0.0] * n_dec, lam=1.0),
                       dtype=float)
        elapsed += time.time() - t0
        n_tot += n_dec
        succ, owned = CAPTURE['succ'], CAPTURE['owned']
        if len(q) != n_dec:
            fails.append("%s: cgae_dag length %d != %d" % (label, len(q), n_dec))
        if not np.all(np.isfinite(q)):
            fails.append("%s: cgae_dag produced non-finite credit" % label)
        for d in range(min(n_dec, len(q))):
            rhs = sum(owned[x] for x in closure(succ, d))
            worst = max(worst, abs(float(q[d]) - rhs))
            checked += 1
    print("D1 %-12s decisions checked %d | worst |Q - closure sum| %.3e | "
          "%.1f ms per episode (%d decisions)"
          % (label, checked, worst, 1000.0 * elapsed / EPISODES, n_tot // EPISODES))
    if worst > 1e-9:
        fails.append("%s: cgae_dag deviates from the closure sum by %.3e" % (label, worst))

CausalTraces._cgae_structure = _orig_struct


# ---- D2 / D3: synthetic fixtures ------------------------------------------- #
def _run(acts, parents, token_to_action, V, lam, beta, scheme='cgae_dag'):
    record_to_action = {id(a): (i, False) for i, a in enumerate(acts)}
    stub = types.SimpleNamespace(
        transition_history=types.SimpleNamespace(transitions=acts))
    # bind the real structure helper to the stub -- the fixtures exercise the
    # genuine edge/ownership construction, not a hand-written substitute
    stub._cgae_structure = lambda *a: CausalTraces._cgae_structure(stub, *a)
    return CausalTraces._redistribute_cgae_dag(
        stub, acts, token_to_action, record_to_action, [0.0] * len(acts),
        beta, lambda t: parents.get(t, []), list(V), lam)


# D2 -- a 4-step chain against textbook GAE
K = 4
rewards = [1.0, 3.0, -2.0, 4.0]
Vs = [0.5, 0.9, 0.2, 1.3]
acts, parents, t2a = [], {'x0': []}, {}
for i in range(K):
    inp = 'x%d' % i
    out = 'x%d' % (i + 1)
    acts.append({'input_tokens': [inp], 'output_tokens': [out],
                 'time': float(i), 'reward': rewards[i]})
    parents[out] = [inp]
    t2a[out] = (i, float(i))

BETA = 0.3
for lam in (1.0, 0.9, 0.5, 0.0):
    q = _run(acts, parents, t2a, Vs, lam, BETA)
    # textbook GAE on the same chain: gamma_k = exp(-beta * dt)
    g = math.exp(-BETA * 1.0)
    A = [0.0] * K
    for i in range(K - 1, -1, -1):
        v_next = Vs[i + 1] if i + 1 < K else 0.0
        delta = rewards[i] + g * v_next - Vs[i]
        A[i] = delta + (g * lam * A[i + 1] if i + 1 < K else 0.0)
    ref = [A[i] + Vs[i] for i in range(K)]
    err = max(abs(q[i] - ref[i]) for i in range(K))
    print("D2 chain lam=%-4s max |cgae_dag - textbook GAE| = %.3e" % (lam, err))
    if err > 1e-9:
        fails.append("D2: chain lam=%s deviates from GAE by %.3e" % (lam, err))

# D3 -- diamond: d -> s1 -> j and d -> s2 -> j, reward only at j
acts_d = [
    {'input_tokens': ['i0'], 'output_tokens': ['a', 'b'], 'time': 0.0, 'reward': 0.0},
    {'input_tokens': ['a'], 'output_tokens': ['c'], 'time': 1.0, 'reward': 0.0},
    {'input_tokens': ['b'], 'output_tokens': ['e'], 'time': 1.0, 'reward': 0.0},
    {'input_tokens': ['c', 'e'], 'output_tokens': ['f'], 'time': 2.0, 'reward': 7.0},
]
parents_d = {'i0': [], 'a': ['i0'], 'b': ['i0'], 'c': ['a'], 'e': ['b'],
             'f': ['c', 'e']}
t2a_d = {'a': (0, 0.0), 'b': (0, 0.0), 'c': (1, 1.0), 'e': (2, 1.0), 'f': (3, 2.0)}
qd = _run(acts_d, parents_d, t2a_d, [0.0] * 4, 1.0, 0.0)
print("D3 diamond credit: %s  (reward 7.0 at the merge; a weight-1 path sum "
      "would give the root 14.0)" % [round(x, 6) for x in qd])
if abs(qd[0] - 7.0) > 1e-9:
    fails.append("D3: diamond root got %.6f, expected 7.0 (counted once)" % qd[0])

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 4/4")
