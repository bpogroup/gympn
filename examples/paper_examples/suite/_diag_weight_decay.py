r"""Where do cgae_flow and cgae_cflow actually differ? The single-successor case.

The R-identity in the outline treats fan-out as the driver, but that misses the
dominant case. With exactly ONE causal successor,

    R(d) = w(d->s)          so    what(d->s) = w/R = 1

i.e. cgae_cflow assigns weight 1 and propagates credit backwards UNDIMINISHED,
while cgae_flow assigns w < 1 whenever s has fan-IN (a jointly-caused successor)
and therefore decays credit multiplicatively along a chain as prod(w).

That predicts the two variants diverge most on LONG CHAINS WITH FAN-IN -- which
is the ncopies N=2 regime (fan-out 1.020, scope S=0.472, i.e. deep chains) --
and not only under fan-out. It also predicts the two failure modes are
different in kind:

    R > 1  (fan-out dominant)  -> cgae_flow INFLATES the critic  -> s1 collapse
    R < 1  (fan-in dominant)   -> cgae_cflow OVER-PROPAGATES     -> N=2 collapse

Measured here per environment:
  * distribution of edge weights w and row sums R (share <1, ==1, >1)
  * the effective backward-propagation mass at causal depth k for each variant,
    i.e. how much credit from a reward k steps downstream reaches a decision.

Run: python _diag_weight_decay.py [n_rollouts]
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LENGTH = 20
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
    ws, Rs, singles = [], [], []
    prop = {'flow': np.zeros(9), 'cflow': np.zeros(9), 'cap': np.zeros(9)}
    cnt = np.zeros(9)
    for ep in range(EPISODES):
        env = rollout(maker, 300 + ep)
        ct = env.pn.causal_trace
        n = len(ct.transition_history.get_action_transitions())
        if n < 4:
            continue
        ct.redistribute_rewards(scheme='cgae_dag', beta=0.0, values=[0.0] * n, lam=1.0)
        succ, w = CAP['succ'], CAP['w']
        R = {}
        for d, kids in succ.items():
            kk = [s for s in kids if s != d]
            if not kk:
                continue
            R[d] = sum(w.get((d, s), 0.0) for s in kk)
            singles.append(1 if len(kk) == 1 else 0)
            Rs.append(R[d])
            for s in kk:
                ws.append(w.get((d, s), 0.0))
        # backward propagation mass at causal depth k, from each decision
        for d in range(n):
            mass = {v: {d: 1.0} for v in prop}
            for k in range(1, 9):
                nxt = {v: {} for v in prop}
                for v in prop:
                    for x, mv in mass[v].items():
                        kk = [s for s in succ.get(x, ()) if s != x]
                        if not kk:
                            continue
                        Rx = R.get(x, 0.0)
                        for s in kk:
                            wx = w.get((x, s), 0.0)
                            if v == 'flow':
                                c = wx
                            elif v == 'cflow':
                                c = wx / Rx if Rx > 0 else 0.0
                            else:               # cap: divide only when R > 1
                                c = wx / max(Rx, 1.0) if Rx > 0 else 0.0
                            nxt[v][s] = nxt[v].get(s, 0.0) + mv * c
                mass = nxt
                tot = sum(mass['flow'].values())
                if not mass['flow'] and not mass['cflow']:
                    break
                cnt[k] += 1
                for v in prop:
                    prop[v][k] += sum(mass[v].values())
    W, RR = np.asarray(ws), np.asarray(Rs)
    if W.size == 0:
        print("  %-18s no edges" % label)
        return
    print("  %-18s w: mean %.3f, ==1 %5.1f%%, <1 %5.1f%% | R: <1 %5.1f%%, ==1 %5.1f%%, >1 %5.1f%% | single-succ %5.1f%%"
          % (label, W.mean(), 100 * np.mean(np.abs(W - 1) < 1e-9), 100 * np.mean(W < 1 - 1e-9),
             100 * np.mean(RR < 1 - 1e-9), 100 * np.mean(np.abs(RR - 1) < 1e-9),
             100 * np.mean(RR > 1 + 1e-9), 100 * np.mean(np.asarray(singles) == 1)))
    with np.errstate(invalid='ignore', divide='ignore'):
        line = "      propagation mass by depth  "
        for v in ['flow', 'cflow', 'cap']:
            vals = np.where(cnt[1:6] > 0, prop[v][1:6] / np.maximum(cnt[1:6], 1), np.nan)
            line += "%s=[%s]  " % (v, " ".join("%.2f" % x for x in vals))
        print(line)


ENVS = []
try:
    from multisite_env import make_multisite
    ENVS.append(("multisite", lambda: make_multisite(4, 1, 0, causal_rl=True,
                                                     allow_postpone=False)))
except Exception:
    pass
try:
    from ncopies_env import make_n_copies
    for N in (1, 2, 4, 8):
        ENVS.append(("ncopies N=%d" % N,
                     (lambda N=N: make_n_copies(N, causal_rl=True, allow_postpone=True,
                                                causal_postpone_tokenflow=True))))
except Exception:
    pass
try:
    from envs import make_env
    ENVS.append(("s1_stoch_sequence",
                 lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                  causal_postpone_tokenflow=True)))
except Exception:
    pass

print("=" * 132)
print("EDGE WEIGHTS, ROW SUMS, AND BACKWARD PROPAGATION (%d rollouts each)" % EPISODES)
print("  flow  c=w        (decays as prod(w) on a chain)")
print("  cflow c=w/R      (==1 on a single successor -> NO decay)")
print("  cap   c=w/max(R,1)  (keeps w when R<1, normalizes only when R>1)")
print("=" * 132)
for lab, mk in ENVS:
    try:
        probe(lab, mk)
    except Exception as e:
        print("  %-18s FAILED: %s" % (lab, str(e)[:60]))
print("=" * 132)
CausalTraces._cgae_structure = _orig
