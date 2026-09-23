r"""Proposition 5 (the exact rescaling and the cosine law), measured.

This is the script behind the paper's cosine table. It did not previously
exist: the table's numbers were not reproducible from the repository, and its
CV column did not satisfy the identity it claimed to verify (CV=0.195 gives
(1+CV^2)^-1/2 = 0.98151, not the tabulated 0.97823). Everything below is
recomputed from traces.

DEFINITION USED FOR delta-bar. The reward-weighted deficiency is the
rho-weighted form,

    delta-bar_d = sum_e (1 - c_d(e)) rho_de owned[e] / sum_e rho_de owned[e],

which is the form the proof of Proposition 4(ii) consumes (it telescopes rho
along paths) and the form Proposition 5(i) needs for A[d] = m_d A*[d] to be an
identity. Section 5.5 of the manuscript previously stated an unweighted
variant; that was an error and has been corrected. At beta=0, rho == 1 and the
two coincide, which is the regime this script measures -- so the correction
does not change these numbers, but it does make the symbol mean one thing.

WHAT IS MEASURED, per environment, over EPISODES untrained (uniform-random)
rollouts, at lam=1, V==0, beta=0:

  c_d(e)    Pr[the W-hat walk started at d visits e]  (Lemma 1), by forward DP
            over the DAG in topological (clock) order.
  A*[d]     sum over the descendant closure of owned[e]        -- Eq. (6)
  A[d]      sum_e c_d(e) owned[e]                              -- the unrolled
            path-weight form of Eq. (4)
  m_d       A[d]/A*[d] = 1 - delta-bar_d
  CV        coefficient of variation of m under p_d ~ A*[d]^2
  cos       <A, A*> / (||A|| ||A*||), pooled over all decisions of all episodes

and three independent checks that are verifications, not fits:

  (C1) the recursion's own output at lam=1, V==0 equals sum_e c_d(e) owned[e]
       -- i.e. Lemma 1's path-weight reading of Eq. (4) is what the code does;
  (C2) cgae_dag's output equals A*;
  (C3) every causal edge is time-forward (Assumption 3), which is what makes
       rho telescope; violations are counted and reported, not clamped away.

Run: python _diag_cosine_law.py [n_rollouts]
"""
import os, sys, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LENGTH = 20
BETA = 0.0          # Table caption: beta = 0, so rho == 1
LAM = 1.0

CAP = {}
_orig = CausalTraces._cgae_structure


def _spy(self, acts, t2a, r2a, gp):
    out = _orig(self, acts, t2a, r2a, gp)
    CAP['succ'], CAP['w'], CAP['owned'], CAP['times'] = out
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


def normalized_weights(succ, w, n):
    """W-hat: successor-normalized edge weights, Eq. (3). d -> [(s, w-hat)]."""
    What = {}
    for d in range(n):
        S = sorted(s for s in succ.get(d, ()) if s != d)
        if not S:
            continue
        R = sum(w.get((d, s), 0.0) for s in S)
        if R > 0:
            What[d] = [(s, w.get((d, s), 0.0) / R) for s in S]
        else:                       # the equal-share fallback of Alg. 1
            What[d] = [(s, 1.0 / len(S)) for s in S]
    return What


def visit_matrix(What, order, n):
    """c_d(e) for every d, by forward DP over the DAG in topological order.

    For fixed d: c_d(d) = 1 and c_d(e) = sum_{p -> e} c_d(p) w-hat(p -> e),
    accumulated with nodes processed in clock order, which is a topological
    order by Assumption 3.
    """
    pos = {d: i for i, d in enumerate(order)}
    C = np.zeros((n, n))
    for d in range(n):
        c = C[d]
        c[d] = 1.0
        for x in order[pos[d]:]:
            cx = c[x]
            if cx == 0.0:
                continue
            for (s, wt) in What.get(x, ()):
                c[s] += cx * wt
    return C


def descendant_closure(What, order, n):
    """Boolean reachability, for A* -- set semantics, each owner counted once."""
    pos = {d: i for i, d in enumerate(order)}
    D = np.zeros((n, n), dtype=bool)
    for d in range(n):
        r = D[d]
        r[d] = True
        for x in order[pos[d]:]:
            if not r[x]:
                continue
            for (s, _) in What.get(x, ()):
                r[s] = True
    return D


def probe(label, maker):
    m_all, astar_all, dbar_all, cbar_all = [], [], [], []
    c1 = c2 = 0.0
    cmin, cmax, rowdev = 1.0, 0.0, 0.0
    edge_viol = edge_tot = 0
    n_dec = n_used = 0

    for ep in range(EPISODES):
        env = rollout(maker, 300 + ep)
        ct = env.pn.causal_trace
        acts = ct.transition_history.get_action_transitions()
        n = len(acts)
        if n == 0:
            continue
        V0 = [0.0] * n
        # cgae_dag first: it is the only scheme routing through
        # _cgae_structure, which is what the spy captures. _redistribute_cgae
        # builds an inline copy of the same objects, so C1 below is a real
        # check that the two constructions agree, not a tautology.
        q_dag = np.asarray(ct.redistribute_rewards(
            scheme='cgae_dag', beta=BETA, values=V0, lam=LAM), dtype=float)
        succ, w, owned, times = CAP['succ'], CAP['w'], CAP['owned'], CAP['times']
        q_cflow = np.asarray(ct.redistribute_rewards(
            scheme='cgae_cflow', beta=BETA, values=V0, lam=LAM), dtype=float)
        owned = np.asarray(owned, dtype=float)
        n_dec += n

        tm = [0.0 if times[i] is None else float(times[i]) for i in range(n)]
        order = sorted(range(n), key=lambda i: (tm[i], i))

        # (C3) Assumption 3: every causal edge time-forward
        for d, kids in succ.items():
            for s in kids:
                if s == d:
                    continue
                edge_tot += 1
                if not tm[s] > tm[d]:
                    edge_viol += 1

        What = normalized_weights(succ, w, n)
        C = visit_matrix(What, order, n)
        D = descendant_closure(What, order, n)

        A = C @ owned                      # sum_e c_d(e) owned[e]
        Astar = D.astype(float) @ owned    # closure, set semantics

        # (C4) Lemma 1 range: c_d(e) is a probability, and W-hat is
        # row-stochastic on non-sinks.
        cmin = min(cmin, float(C.min()))
        cmax = max(cmax, float(C.max()))
        for d, edges in What.items():
            rowdev = max(rowdev, abs(sum(wt for _, wt in edges) - 1.0))

        c1 = max(c1, float(np.max(np.abs(A - q_cflow))))
        c2 = max(c2, float(np.max(np.abs(Astar - q_dag))))

        # mean c_d(e) over STRICT descendant pairs (c_d(d)=1 is trivial). This
        # is the unweighted pair statistic Section 5.5 quotes; it is NOT
        # delta-bar, which is reward-weighted, and the two differ materially.
        strict = D.copy()
        np.fill_diagonal(strict, False)
        if strict.any():
            cbar_all.append(C[strict])

        keep = Astar != 0.0
        n_used += int(keep.sum())
        m = A[keep] / Astar[keep]
        m_all.append(m)
        astar_all.append(Astar[keep])
        dbar_all.append(1.0 - m)

    if not m_all:
        print("  %-14s no decisions with A* != 0" % label)
        return None

    m = np.concatenate(m_all)
    astar = np.concatenate(astar_all)
    dbar = np.concatenate(dbar_all)
    cbar = np.concatenate(cbar_all) if cbar_all else np.array([1.0])

    p = astar ** 2
    p = p / p.sum()
    Em = float((p * m).sum())
    Em2 = float((p * m * m).sum())
    var = max(Em2 - Em * Em, 0.0)
    CV = float(np.sqrt(var) / Em) if Em != 0 else float('nan')

    A = m * astar
    cos = float(A @ astar / (np.linalg.norm(A) * np.linalg.norm(astar)))
    pred = (1.0 + CV * CV) ** -0.5

    print("  %-14s 1-cbar %.3f | dbar %.3f | CV %.3f | cos %.5f | (1+CV^2)^-1/2 %.5f | "
          "1-cos %.3f | id %.1e | C1 %.1e C2 %.1e | c[%.3f,%.3f] rowdev %.1e | "
          "edges %d viol %d | dec %d/%d"
          % (label, 1 - cbar.mean(), dbar.mean(), CV, cos, pred, 1 - cos,
             abs(cos - pred),
             c1, c2, cmin, cmax, rowdev, edge_tot, edge_viol, n_used, n_dec))
    return dict(env=label, one_minus_cbar=1 - cbar.mean(), dbar=dbar.mean(), CV=CV, cos=cos, pred=pred,
                one_minus=1 - cos, ident=abs(cos - pred), c1=c1, c2=c2,
                edges=edge_tot, viol=edge_viol, cmin=cmin, cmax=cmax,
                rowdev=rowdev)


ENVS = []
from multisite_env import make_multisite
ENVS.append(("multi-site", lambda: make_multisite(4, 1, 0, causal_rl=True,
                                                  allow_postpone=False)))
from ncopies_env import make_n_copies
for N in (4, 8):
    ENVS.append(("N=%d" % N,
                 (lambda N=N: make_n_copies(N, causal_rl=True, allow_postpone=True,
                                            causal_postpone_tokenflow=True))))
from envs import make_env
ENVS.append(("single comp.", lambda: make_env('s1_stoch_sequence', causal_rl=True,
                                              allow_postpone=True,
                                              causal_postpone_tokenflow=True)))

print("=" * 132)
print("PROPOSITION 5 -- exact rescaling and cosine law (%d untrained rollouts, "
      "lam=%.0f, V==0, beta=%.0f)" % (EPISODES, LAM, BETA))
print("=" * 132)
rows = []
for label, maker in ENVS:
    r = probe(label, maker)
    if r:
        rows.append(r)
print("=" * 132)

print("\nLaTeX body for the cosine table:\n")
for r in rows:
    print("%-12s & $%.3f$ & $%.3f$ & $%.5f$ & $%.5f$ & $%.3f$ \\\\"
          % (r['env'], r['dbar'], r['CV'], r['cos'], r['pred'], r['one_minus']))

CausalTraces._cgae_structure = _orig
