r"""Does each scheme's credit POINT THE RIGHT WAY? (ground-truth validation)

Every metric so far has been internal to the estimators (row sums, P3 identity
ratios, fan-out). This compares them against a MEASURED ground truth, which is
the question a referee actually asks.

GROUND TRUTH. At a decision state with k >= 2 actions, fork the simulator once
per action under COMMON RANDOM NUMBERS -- identical exogenous draws and an
identical continuation policy across branches -- and record the realized SMDP
return-to-go from the fork,

    G(a) = sum over rewards j after the fork of e^{-beta (t_j - t0)} r_j

Because the draws are shared, differences in G across actions are the causal
effect of the action on the objective, with the exogenous component cancelled.
That is the same construction counterfactual.py::fork_cf_advantage uses for its
COMA baseline, and it is what the credit assigned to that decision SHOULD be
ordinally consistent with.

ESTIMATOR SIDE. On each branch, every scheme's emitted credit for the FORKED
decision, Q_scheme[d]. V is held constant across decisions in this harness, so
the additive V in Q = A + V cancels when ranking actions at one state.

METRICS, per scheme, averaged over fork points:
  * SPEARMAN rho between (G(a))_a and (Q(a))_a  -- does the credit order the
    actions the way the objective does?
  * TOP-1 agreement -- does argmax_a Q pick the argmax_a G action? This is the
    decision-relevant one: a policy gradient chases the argmax.
  * reported for ALL actions and for PRODUCTION actions only, since cgae_cflow2
    emits 0 at postpone by construction (data.py substitutes the SMDP-TD form
    there), so including postpone would penalise it for something the estimator
    is not responsible for.

PREDICTION, registered here so it cannot drift after the fact: rho and top-1
should order the schemes the way s1 performance does,

    cgae_cflow (0.792) > cgae (0.650) > cgae_dag (0.386) > cgae_cflow2 (~0.11)

with cgae_flow (0.166) low for its own separate reason (unanchored bootstrap).
If the ordering does NOT appear, the "credit points the right way" story is
wrong and the negative results need separate explanations.

LIMITATION, stated plainly: the continuation policy here is RANDOM, so this
measures counterfactual value under random continuation. That is the regime a
high-entropy policy actually sees early in training, but it is not the same as
the converged-policy counterfactual, and the numbers should not be read as if
it were.

Run: python _diag_credit_vs_counterfactual.py [n_episodes]
"""
import os, sys, copy, math, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

try:
    from scipy import stats as sps
except ImportError:
    sps = None

LENGTH = 20
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_FORKS_PER_EP = 3
BETA = 0.1
SCHEMES = ['cgae_cflow', 'cgae', 'cgae_dag', 'cgae_cflow2', 'cgae_flow']
S1_PERF = {'cgae_cflow': 0.792, 'cgae': 0.650, 'cgae_dag': 0.386,
           'cgae_cflow2': 0.109, 'cgae_flow': 0.166}


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


def is_postpone_flags(pn):
    ad = pn.get_graph_observation().get('actions_dict') or []
    return [bool(e and isinstance(e[0], (list, tuple)) and len(e[0])
                 and e[0][0] == 'postpone') for e in ad]


def ground_truth_return(ct, t0, n_before):
    """SMDP return-to-go realized after the fork, from the trace's own records.

    Uses the transition history rather than env.step's reward channel so the
    clock and the rewards come from one source, the same way the estimators
    read them.
    """
    tot = 0.0
    for tr in ct.transition_history.transitions:
        rv = tr.get('reward', 0.0)
        if rv == 0.0:
            continue
        t = tr.get('time')
        if t is None or float(t) < t0:
            continue
        tot += math.exp(-BETA * max(0.0, float(t) - t0)) * float(rv)
    return tot


def rollout_to_end(env, seed):
    random.seed(seed)
    done, guard = False, 0
    while not done and guard < 10000:
        guard += 1
        k = len(env.pn.get_graph_observation().get('actions_dict') or [])
        if k == 0:
            break
        _, _, done, _, _ = env.step(random.randrange(k))
    return env


def probe(label, builder, n_eps):
    rho = {sc: {'all': [], 'prod': []} for sc in SCHEMES}
    top1 = {sc: {'all': [], 'prod': []} for sc in SCHEMES}
    n_fork = 0
    for ep in range(n_eps):
        random.seed(900 + ep)
        env = build(builder)
        env.reset()
        forks, done, guard = 0, False, 0
        while not done and guard < 10000 and forks < MAX_FORKS_PER_EP:
            guard += 1
            ad = env.pn.get_graph_observation().get('actions_dict') or []
            k = len(ad)
            if k == 0:
                break
            if k >= 2 and random.random() < 0.4:
                snap_pn = copy.deepcopy(env.pn)
                snap_i = env.i
                pp = is_postpone_flags(snap_pn)
                d_idx = len(env.pn.causal_trace.transition_history
                            .get_action_transitions())
                t0 = float(getattr(env.pn, 'clock', 0.0))
                seed0 = random.randrange(2 ** 31 - 1)
                G, Q = [], {sc: [] for sc in SCHEMES}
                ok = True
                for a in range(k):
                    env.pn = copy.deepcopy(snap_pn)
                    env.i = snap_i
                    if len(env.pn.get_graph_observation().get('actions_dict') or []) != k:
                        ok = False
                        break
                    random.seed(seed0)
                    _, _, dn, _, _ = env.step(a)
                    if not dn:
                        rollout_to_end(env, seed0 + 1)
                    ct = env.pn.causal_trace
                    n = len(ct.transition_history.get_action_transitions())
                    if n == 0 or d_idx >= n:
                        ok = False
                        break
                    G.append(ground_truth_return(ct, t0, d_idx))
                    V = [0.3] * n
                    for sc in SCHEMES:
                        q = ct.redistribute_rewards(scheme=sc, beta=BETA,
                                                    values=V, lam=0.95)
                        Q[sc].append(float(q[d_idx]))
                env.pn = copy.deepcopy(snap_pn)
                env.i = snap_i
                env.pn.get_graph_observation()
                if not ok or len(set(np.round(G, 9))) < 2:
                    continue          # ground truth flat -> nothing to rank
                n_fork += 1
                forks += 1
                for sc in SCHEMES:
                    for tag, sel in (('all', list(range(k))),
                                     ('prod', [i for i in range(k) if not pp[i]])):
                        if len(sel) < 2:
                            continue
                        g = [G[i] for i in sel]
                        q = [Q[sc][i] for i in sel]
                        if len(set(np.round(g, 9))) < 2:
                            continue
                        if sps is not None and len(set(np.round(q, 9))) >= 2:
                            r = sps.spearmanr(g, q).correlation
                            if not np.isnan(r):
                                rho[sc][tag].append(float(r))
                        top1[sc][tag].append(
                            float(g[int(np.argmax(q))] >= max(g) - 1e-9))
            ad = env.pn.get_graph_observation().get('actions_dict') or []
            if not ad:
                break
            _, _, done, _, _ = env.step(random.randrange(len(ad)))

    print("=" * 80)
    print("%s -- %d usable fork points  (ground truth: CRN return-to-go, beta=%.2f)"
          % (label, n_fork, BETA))
    print("  %-12s %-22s %-22s %s" % ("scheme", "Spearman rho (all)",
                                      "Spearman rho (prod)", "top-1 agree (prod)"))
    for sc in SCHEMES:
        ra, rp = rho[sc]['all'], rho[sc]['prod']
        tp = top1[sc]['prod']
        print("  %-12s %-22s %-22s %-8s   [s1 perf %.3f]"
              % (sc,
                 "%+.3f (n=%d)" % (np.mean(ra), len(ra)) if ra else "n/a",
                 "%+.3f (n=%d)" % (np.mean(rp), len(rp)) if rp else "n/a",
                 "%.1f%%" % (100.0 * np.mean(tp)) if tp else "n/a",
                 S1_PERF[sc]))


for label, builder in [
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True)),
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True))]:
    probe(label, builder, N_EPISODES)
print("=" * 80)
