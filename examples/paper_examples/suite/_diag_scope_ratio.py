r"""How much of the remaining episode does each estimator actually scope to?

The scoping ratio

    S(d) = |desc(d)| / |{decisions after d}|

is what determines whether an estimator is doing credit assignment at all.
S -> 1 means a decision's credit covers essentially the whole remaining episode,
i.e. the estimator has degenerated to the unscoped return-to-go (`mc_q`, the
lineage ABLATION). S -> 1/K means it has isolated one of K concurrent streams.

This is the mathematical content behind two observations:
  * why component-scoping beats PPO where K > 1 (Proposition 2's variance term);
  * why `cgae_dag`, which is EXACTLY the descendant-closure estimand and
    therefore the unbiased one, degenerates on a K=1 environment -- there
    desc(d) is nearly everything, so the exact estimand is approximately the
    unscoped return and exactness buys no scoping at all.

Also reports the effective path-weight mass that `cgae_cflow` retains,
    C(d) = sum_j c_d(j) / |desc(d)|,     c_d(j) = sum over paths of prod(w-hat)
which is the shrinkage factor in Q = sum_j c_d(j) rho owned[j].

Run: python _diag_scope_ratio.py [n_rollouts]
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LENGTH = 20

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


def probe(label, maker):
    S, C = [], []
    for ep in range(EPISODES):
        env = rollout(maker, 300 + ep)
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n < 4:
            continue
        ct.redistribute_rewards(scheme='cgae_dag', beta=0.0,
                                values=[0.0] * n, lam=1.0)
        succ, w = CAP['succ'], CAP['w']
        # normalized (convex) weights, cgae_cflow's kernel
        what = {}
        for d, kids in succ.items():
            kk = [s for s in kids if s != d]
            R = sum(w.get((d, s), 0.0) for s in kk)
            for s in kk:
                what[(d, s)] = (w.get((d, s), 0.0) / R) if R > 0 else (1.0 / max(1, len(kk)))
        for d in range(n):
            later = n - d - 1
            if later <= 0:
                continue
            # descendant closure + total path weight per descendant (forward DP)
            seen, stack = {d}, [d]
            while stack:
                x = stack.pop()
                for s in succ.get(x, ()):
                    if s not in seen:
                        seen.add(s)
                        stack.append(s)
            desc = seen - {d}
            S.append(len(desc) / later)
            mass = [0.0] * n
            mass[d] = 1.0
            for x in range(d, n):
                if mass[x] == 0.0:
                    continue
                for s in succ.get(x, ()):
                    if s > x:
                        mass[s] += mass[x] * what.get((x, s), 0.0)
            if desc:
                C.append(float(np.mean([mass[j] for j in desc])))
    if not S:
        print("  %-24s no data" % label)
        return
    print("  %-24s scope S = %.3f (median %.3f)   cflow path-mass C = %.3f"
          % (label, float(np.mean(S)), float(np.median(S)),
             float(np.mean(C)) if C else float('nan')))


ENVS = []
try:
    from multisite_env import make_multisite
    ENVS.append(("multisite (K=8.0)", lambda: make_multisite(4, 1, 0, causal_rl=True,
                                                             allow_postpone=False)))
except Exception:
    pass
try:
    from ncopies_env import make_n_copies
    for N in (8, 4, 2, 1):
        ENVS.append(("ncopies N=%d" % N,
                     (lambda N=N: make_n_copies(N, causal_rl=True, allow_postpone=True,
                                                causal_postpone_tokenflow=True))))
except Exception:
    pass
try:
    from envs import make_env
    for nm in ["s2_stoch_scaled", "s1_stoch_sequence", "s3_stoch_mixed",
               "s4_stoch_mixed_rework"]:
        ENVS.append((nm, (lambda x=nm: make_env(x, causal_rl=True, allow_postpone=True,
                                                causal_postpone_tokenflow=True))))
except Exception:
    pass

print("=" * 92)
print("SCOPING RATIO  S(d) = |desc(d)| / |decisions after d|   (%d rollouts each)" % EPISODES)
print("  S -> 1  : estimator covers the whole remaining episode == unscoped return (mc_q)")
print("  S -> 1/K: estimator has isolated one of K concurrent streams")
print("=" * 92)
for lab, mk in ENVS:
    try:
        probe(lab, mk)
    except Exception as e:
        print("  %-24s FAILED: %s" % (lab, str(e)[:60]))
print("=" * 92)
CausalTraces._cgae_structure = _orig
