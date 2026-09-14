r"""Signal-to-noise of the credit signal -- the surviving explanation.

Three mechanistic hypotheses for the cgae ordering on s1 have already been
falsified (postpone-invariance, DAG sparsity, per-state ranking fidelity --
the last one came out INVERTED: cgae_dag ranks actions best against the measured
counterfactual yet learns worst). What remains is plain bias-variance: a policy
gradient integrates credit over many samples, so what matters is not how
accurate one sample's ranking is but how much of the credit's variation is
BETWEEN ACTIONS (signal the gradient can follow) versus WITHIN AN ACTION across
exogenous draws (noise it must average away).

DESIGN (an F-ratio / one-way ANOVA on the forked decision's credit):

  At a decision state, for each production action a and each of M exogenous
  seeds m, fork and roll out, then read scheme s's credit for the FORKED
  decision, Q[a, m]. Seeds are SHARED across actions (paired / CRN), so:

      signal(s) = SD over a of ( mean over m of Q[a,m] )
      noise(s)  = mean over a of ( SD over m of Q[a,m] )
      SNR(s)    = signal / noise

  signal is the part of the credit that actually depends on the decision;
  noise is what the exogenous stream injects at a FIXED decision. A scheme can
  have a large signal and still be unlearnable if its noise is larger.

PREDICTION, registered before running: SNR should order the schemes the way s1
performance does --

    cgae_cflow (0.792) > cgae (0.650) > cgae_dag (0.386) > cgae_cflow2 (0.106)

with cgae_flow (0.166) low. Specifically cgae_dag, whose closure sum on a
one-blob env is ~the whole episode return, should show the LOWEST SNR despite
being the unbiased variant with the best per-state ranking. If SNR does not
order this way either, then bias-variance is also not the explanation and I am
out of mechanisms -- which is itself worth knowing before anything is written.

Restricted to PRODUCTION actions: cgae_cflow2 emits 0 at postpone by
construction (data.py substitutes the SMDP-TD form), so including postpone would
score it on something the estimator is not responsible for.

Run: python _diag_credit_snr.py [n_episodes] [n_seeds] [max_actions]
"""
import os, sys, copy, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
M_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAX_ACTIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
MAX_FORKS_PER_EP = 2
BETA = 0.1
SCHEMES = ['cgae_cflow', 'cgae', 'cgae_dag', 'cgae_cflow2', 'cgae_flow']
S1_PERF = {'cgae_cflow': 0.792, 'cgae': 0.650, 'cgae_dag': 0.386,
           'cgae_cflow2': 0.106, 'cgae_flow': 0.166}


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


def prod_actions(pn):
    ad = pn.get_graph_observation().get('actions_dict') or []
    return [i for i, e in enumerate(ad)
            if not (e and isinstance(e[0], (list, tuple)) and len(e[0])
                    and e[0][0] == 'postpone')]


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
    snr = {sc: [] for sc in SCHEMES}
    sig = {sc: [] for sc in SCHEMES}
    noi = {sc: [] for sc in SCHEMES}
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
            acts = prod_actions(env.pn)[:MAX_ACTIONS]
            if len(acts) >= 2 and random.random() < 0.5:
                snap_pn = copy.deepcopy(env.pn)
                snap_i = env.i
                d_idx = len(env.pn.causal_trace.transition_history
                            .get_action_transitions())
                seeds = [random.randrange(2 ** 31 - 1) for _ in range(M_SEEDS)]
                # Q[scheme][action][seed]
                Q = {sc: [[None] * M_SEEDS for _ in acts] for sc in SCHEMES}
                ok = True
                for ai, a in enumerate(acts):
                    for mi, sd in enumerate(seeds):
                        env.pn = copy.deepcopy(snap_pn)
                        env.i = snap_i
                        if len(env.pn.get_graph_observation().get('actions_dict') or []) != k:
                            ok = False
                            break
                        random.seed(sd)
                        _, _, dn, _, _ = env.step(a)
                        if not dn:
                            rollout_to_end(env, sd + 1)
                        ct = env.pn.causal_trace
                        n = len(ct.transition_history.get_action_transitions())
                        if n == 0 or d_idx >= n:
                            ok = False
                            break
                        V = [0.3] * n
                        for sc in SCHEMES:
                            q = ct.redistribute_rewards(scheme=sc, beta=BETA,
                                                        values=V, lam=0.95)
                            Q[sc][ai][mi] = float(q[d_idx])
                    if not ok:
                        break
                env.pn = copy.deepcopy(snap_pn)
                env.i = snap_i
                env.pn.get_graph_observation()
                if not ok:
                    continue
                n_fork += 1
                forks += 1
                for sc in SCHEMES:
                    A = np.array(Q[sc], dtype=float)        # actions x seeds
                    per_action_mean = A.mean(axis=1)
                    s = float(np.std(per_action_mean))      # between actions
                    e = float(np.mean(np.std(A, axis=1)))   # within action
                    sig[sc].append(s)
                    noi[sc].append(e)
                    if e > 1e-12:
                        snr[sc].append(s / e)
            ad = env.pn.get_graph_observation().get('actions_dict') or []
            if not ad:
                break
            _, _, done, _, _ = env.step(random.randrange(len(ad)))

    print("=" * 80)
    print("%s -- %d fork points, %d seeds x <=%d production actions"
          % (label, n_fork, M_SEEDS, MAX_ACTIONS))
    print("  %-12s %10s %10s %14s   %s"
          % ("scheme", "signal", "noise", "SNR (sig/noise)", "s1 perf"))
    for sc in SCHEMES:
        if not sig[sc]:
            continue
        v = np.array(snr[sc]) if snr[sc] else np.array([np.nan])
        print("  %-12s %10.4f %10.4f %14s   %.3f"
              % (sc, float(np.mean(sig[sc])), float(np.mean(noi[sc])),
                 "%.4f (med %.4f)" % (float(np.nanmean(v)), float(np.nanmedian(v))),
                 S1_PERF[sc]))


for label, builder in [
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True)),
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True))]:
    probe(label, builder, N_EPISODES)
print("=" * 80)
