r"""Does DAG sparsity explain the cgae ordering on s1? (the "faithfulness loses" test)

Three separate negative results all point the same way on s1, where the
component partition is ONE blob (K=1.0, 100% of reward mass):

    cgae_cflow   0.792   sparse DAG, path-averaged credit
    cgae         0.650   sparse DAG, mean over successors
    cgae_dag     0.386   closure semantics -> desc(d) ~ whole episode
    cgae_cflow2  ~0.11   postpone transparent -> +97 edges/episode

HYPOTHESIS: with a single component, credit LOCALIZATION comes entirely from
the causal DAG being sparse. Anything that raises effective connectivity
diffuses credit over more decisions and destroys the estimator's ability to
DISCRIMINATE between the actions available at a state -- so the mis-specified
DAG's edge-masking was acting as regularization.

That predicts an ordering, so it is falsifiable. Two families of measurement on
the same traces:

STRUCTURAL   edges, mean fan-out, mean |desc(d)| under each DAG convention.
             Tests the "denser" half of the claim.

DISCRIMINATION  the half that actually matters for learning. Fork every action
             at a decision state under common random numbers, compute each
             scheme's emitted credit for the FORKED decision under each action,
             and take the spread across actions. A scheme whose credit barely
             moves when the action changes cannot drive a policy gradient,
             whatever its estimand. Scale-normalized by the RMS credit in that
             episode, since schemes differ in magnitude.

Prediction: discrimination should order cgae_cflow > cgae > cgae_dag >
cgae_cflow2, matching performance. If it does not, the sparsity story is wrong
and the three negative results need separate explanations.

Run: python _diag_sparsity_discrimination.py [n_episodes]
"""
import os, sys, copy, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAX_FORKS_PER_EP = 3
SCHEMES = ['cgae_cflow', 'cgae', 'cgae_dag', 'cgae_cflow2', 'cgae_flow']


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


def structure(ct, skip_pp):
    """Edges + successors under a DAG convention (skip_pp=True is cgae_cflow2's)."""
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

    succ = {}
    n_edges = 0
    for idx, a in enumerate(acts):
        if skip_pp and idx in pp:
            continue
        for tid in list(a.get('input_tokens', ()) or ()):
            for dd in producers(tid, idx):
                if idx not in succ.setdefault(dd, set()):
                    n_edges += 1
                succ[dd].add(idx)
    return succ, n_edges, len(acts)


def desc_stats(succ, n):
    tot = []
    for d in range(n):
        seen, stack = {d}, [d]
        while stack:
            x = stack.pop()
            for s in succ.get(x, ()):
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        tot.append(len(seen) - 1)
    return float(np.mean(tot)) if tot else 0.0


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


def credits(ct, n):
    V = [0.3] * n
    out = {}
    for sc in SCHEMES:
        out[sc] = np.asarray(ct.redistribute_rewards(scheme=sc, beta=0.1,
                                                     values=V, lam=0.95), dtype=float)
    return out


def probe(label, builder, n_eps):
    struct = {'sparse': [], 'dense': []}
    spreads = {sc: [] for sc in SCHEMES}
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
                d_idx = len(env.pn.causal_trace.transition_history
                            .get_action_transitions())
                seed0 = random.randrange(2 ** 31 - 1)
                per_action = {sc: [] for sc in SCHEMES}
                rms = {sc: [] for sc in SCHEMES}
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
                    cr = credits(ct, n)
                    for sc in SCHEMES:
                        per_action[sc].append(float(cr[sc][d_idx]))
                        rms[sc].append(float(np.sqrt(np.mean(cr[sc] ** 2))))
                    if a == 0:      # structural snapshot on one branch
                        for name, skip in (('sparse', False), ('dense', True)):
                            su, ne, nn = structure(ct, skip)
                            struct[name].append((ne, nn, desc_stats(su, nn),
                                                 float(np.mean([len(v) for v in su.values()]))
                                                 if su else 0.0))
                env.pn = copy.deepcopy(snap_pn)
                env.i = snap_i
                env.pn.get_graph_observation()
                if ok:
                    for sc in SCHEMES:
                        sc_rms = float(np.mean(rms[sc])) or 1.0
                        spreads[sc].append((max(per_action[sc]) - min(per_action[sc]))
                                           / (sc_rms if sc_rms > 1e-12 else 1.0))
                    forks += 1
            ad = env.pn.get_graph_observation().get('actions_dict') or []
            if not ad:
                break
            _, _, done, _, _ = env.step(random.randrange(len(ad)))

    print("=" * 78)
    print("%s   (%d fork points)" % (label, len(spreads[SCHEMES[0]])))
    for name in ('sparse', 'dense'):
        if not struct[name]:
            continue
        arr = np.array(struct[name], dtype=float)
        print("  DAG %-7s edges %6.1f | decisions %5.1f | mean |desc(d)| %6.2f | mean fan-out %.3f"
              % (name + ("(cflow/cgae/dag)" if name == 'sparse' else "(cflow2)"),
                 arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean(), arr[:, 3].mean()))
    print("  DISCRIMINATION -- credit spread across actions at one state, /RMS credit:")
    s1_norm = {'cgae_cflow': 0.792, 'cgae': 0.650, 'cgae_dag': 0.386,
               'cgae_cflow2': 0.109, 'cgae_flow': 0.166}
    for sc in SCHEMES:
        v = np.array(spreads[sc])
        if v.size == 0:
            continue
        print("    %-12s mean %.4f   median %.4f      (s1 norm perf %.3f)"
              % (sc, v.mean(), np.median(v), s1_norm[sc]))


for label, builder in [
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True)),
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True))]:
    probe(label, builder, N_EPISODES)
print("=" * 78)
