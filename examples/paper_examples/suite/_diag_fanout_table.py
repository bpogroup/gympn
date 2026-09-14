r"""Fan-out and component structure across every environment -- the paper's
mechanism table, computed with NO training.

Two structural statistics predict, a priori, what the cgae family will do on a
given problem:

  K          reward-bearing causal components. K=1 predicts the null (nothing to
             scope); K>1 is where component-factored credit can pay. This is the
             existing K-diagnostic.

  fan-out    mean |succ(d)| over the causal DAG, and the share of decisions with
             more than one causal successor. This predicts whether the
             AGGREGATION RULE matters at all: at fan-out exactly 1 every cgae
             variant coincides by chain-exactness, so cgae / cgae_flow /
             cgae_cflow / cgae_dag are literally the same algorithm.

Measured so far, and the reason this table belongs in the paper:

    env            fan-out   variants diverge?   spread
    multisite       1.000    no, BIT-IDENTICAL   0.000
    ncopies N=4     1.07     barely              0.955 / 0.936 / 0.888
    s1              1.34     sharply             0.792 / 0.650 / 0.166

i.e. the aggregation rule matters in proportion to measured fan-out -- a
mechanism that predicts its own relevance, the same evidential pattern as the
K-diagnostic. This script extends that from 3 environments to the whole suite so
the relationship is a curve rather than three points.

Also reports max|credit difference| between the variants directly, which is the
sharpest possible statement: 0.0 means they are the same estimator on that env.

Run: python _diag_fanout_table.py [n_rollouts]
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LENGTH = 20
VARIANTS = ['cgae', 'cgae_flow', 'cgae_cflow', 'cgae_dag']

CAP = {}
_orig = CausalTraces._cgae_structure


def _spy(self, acts, t2a, r2a, gp):
    out = _orig(self, acts, t2a, r2a, gp)
    CAP['succ'], CAP['w'], CAP['owned'] = out[0], out[1], out[2]
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


def components(succ, owned, n):
    """Reward-bearing causal components via union-find over shared descendants."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for d, kids in succ.items():
        for s in kids:
            union(d, s)
    comps = {}
    for d in range(n):
        if owned[d] != 0.0:
            comps.setdefault(find(d), 0.0)
            comps[find(d)] += owned[d]
    return comps


def probe(label, maker):
    fans, Rs, Ks, shares = [], [], [], []
    maxdiff = {v: 0.0 for v in VARIANTS}
    for ep in range(EPISODES):
        env = rollout(maker, 300 + ep)
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n == 0:
            continue
        V = [0.3] * n
        creds = {}
        for v in VARIANTS:
            creds[v] = np.asarray(ct.redistribute_rewards(scheme=v, beta=0.1,
                                                          values=V, lam=0.95),
                                  dtype=float)
        base = creds['cgae_cflow']
        for v in VARIANTS:
            if v == 'cgae_cflow':
                continue
            maxdiff[v] = max(maxdiff[v], float(np.max(np.abs(creds[v] - base))))
        succ, w, owned = CAP['succ'], CAP['w'], CAP['owned']
        for d, kids in succ.items():
            kk = [s for s in kids if s != d]
            if not kk:
                continue
            fans.append(len(kk))
            Rs.append(sum(w.get((d, s), 0.0) for s in kk))
        comps = components(succ, owned, n)
        if comps:
            Ks.append(len(comps))
            tot = sum(abs(x) for x in comps.values())
            if tot > 0:
                shares.append(max(abs(x) for x in comps.values()) / tot)
    if not fans:
        print("  %-22s no causal edges" % label)
        return
    F, R = np.asarray(fans), np.asarray(Rs)
    print("  %-22s fan %.3f (>1: %4.1f%%) | R %.3f+-%.3f | K %.2f | top-comp %4.1f%% | "
          "maxdiff vs cflow: cgae %.3g  flow %.3g  dag %.3g"
          % (label, F.mean(), 100 * np.mean(F > 1), R.mean(), R.std(),
             float(np.mean(Ks)) if Ks else float('nan'),
             100 * float(np.mean(shares)) if shares else float('nan'),
             maxdiff['cgae'], maxdiff['cgae_flow'], maxdiff['cgae_dag']))


ENVS = []
try:
    from ncopies_env import make_n_copies
    for N in (1, 2, 4, 8):
        ENVS.append(("ncopies N=%d" % N,
                     (lambda N=N: make_n_copies(N, causal_rl=True, allow_postpone=True,
                                                causal_postpone_tokenflow=True))))
except Exception as e:
    print("ncopies unavailable:", e)
try:
    from multisite_env import make_multisite
    ENVS.append(("multisite", lambda: make_multisite(4, 1, 0, causal_rl=True,
                                                     allow_postpone=False)))
except Exception as e:
    print("multisite unavailable:", e)
try:
    from envs import make_env, HEURISTICS
    for name in sorted(HEURISTICS):
        ENVS.append((name, (lambda nm=name: make_env(nm, causal_rl=True,
                                                     allow_postpone=True,
                                                     causal_postpone_tokenflow=True))))
except Exception as e:
    print("suite envs unavailable:", e)

print("=" * 118)
print("FAN-OUT / COMPONENT TABLE  (%d random rollouts each; no training)" % EPISODES)
print("  fan = mean |succ(d)|;  R = row sum;  K = reward-bearing components;")
print("  maxdiff = max |credit(variant) - credit(cgae_cflow)|  -- 0 means SAME ESTIMATOR")
print("=" * 118)
for label, maker in ENVS:
    try:
        probe(label, maker)
    except Exception as e:
        print("  %-22s FAILED: %s" % (label, str(e)[:70]))
print("=" * 118)
CausalTraces._cgae_structure = _orig
