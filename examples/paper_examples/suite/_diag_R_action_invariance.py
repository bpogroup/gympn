r"""Is R(d) action-measurable? The crux of the cgae_cflow theorem (L1).

cgae_cflow normalizes the causal-successor weights by the row sum

    R(d) = sum_{s in succ(d)} w(d->s)

so the bootstrap is a convex combination. That turns the recursion into
SMDP-GAE for a Markov chain over the provenance DAG, with kernel
what(d->s) = w(d->s)/R(d). The policy-gradient argument needs that kernel to be
a function of the STATE, not of the action taken at d: an action-independent
1/R(d) is a per-state rescaling, which preserves the ascent direction, whereas
an action-dependent one distorts the comparison between the very actions the
gradient is choosing among.

The supporting intuition is that transition arity in a Petri net is structural
-- how many tokens a transition consumes and produces is fixed by the net, not
by which binding fires. But R(d) also depends on DOWNSTREAM structure, which
the action can influence. That is what this measures rather than assumes.

METHOD. Roll out to a decision state with >= 2 available actions. Snapshot the
simulator (deepcopy(pn) + env.i, the mid-episode snapshot counterfactual.py
uses). For each available action: restore the snapshot, reseed the RNG to a
shared value so every branch faces IDENTICAL exogenous draws (CRN), fire that
action, then roll the same random continuation to the end. Recover the causal
structure through the library's own `_cgae_structure`, and read R(d) for the
FORKED decision -- which sits at the same index in every branch, since the
prefix is shared.

Reported per fork point: R under each action, the spread max-min, and the same
for the raw fan-out |succ(d)|. Aggregated: the fraction of fork points where R
is identical across all actions.

  spread ~ 0  ->  L1 holds, cgae_cflow's kernel is action-independent, and the
                  normalization is a per-state rescaling.
  spread >> 0 ->  L1 fails; 1/R(d) is action-dependent and the convex form
                  needs a different justification.

Run: python _diag_R_action_invariance.py [n_episodes]
"""
import os, sys, copy, random, types, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces
from envs import make_env
from ncopies_env import make_n_copies

LENGTH = 20
N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAX_FORKS_PER_EP = 4

CAPTURE = {}
_orig_struct = CausalTraces._cgae_structure


def _spy(self, action_transitions, token_to_action, record_to_action, get_parents):
    out = _orig_struct(self, action_transitions, token_to_action,
                       record_to_action, get_parents)
    CAPTURE['succ'], CAPTURE['w_edge'] = out[0], out[1]
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


def n_decisions(env):
    return len(env.pn.causal_trace.transition_history.get_action_transitions())


def structure(ct, skip_pp):
    """Edge set + inflow shares, rebuilt locally so both DAG conventions can be
    measured on the SAME finished trace: skip_pp=False is cgae_cflow's view,
    skip_pp=True is cgae_cflow2's (postpone transparent)."""
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

    w, succ = {}, {}
    for idx, a in enumerate(acts):
        if skip_pp and idx in pp:
            continue
        contrib, k = {}, 0
        for tid in list(a.get('input_tokens', ()) or ()):
            prods = producers(tid, idx)
            if not prods:
                continue
            k += 1
            for dd in prods:
                contrib[dd] = contrib.get(dd, 0.0) + 1.0 / len(prods)
        if k:
            for dd, c in contrib.items():
                w[(dd, idx)] = c / k
                succ.setdefault(dd, set()).add(idx)
    return w, succ


def rollout_to_end(env, seed):
    random.seed(seed)
    done = False
    guard = 0
    while not done and guard < 10000:
        guard += 1
        obs = env.pn.get_graph_observation()
        acts = obs.get('actions_dict') or {}
        k = len(acts)
        if k == 0:
            break
        _, _, done, _, _ = env.step(random.randrange(k))
    return env


def probe(label, builder, n_eps):
    rows = []
    for ep in range(n_eps):
        random.seed(900 + ep)
        env = build(builder)
        env.reset()
        forks = 0
        done = False
        guard = 0
        while not done and guard < 10000 and forks < MAX_FORKS_PER_EP:
            guard += 1
            obs = env.pn.get_graph_observation()
            acts = obs.get('actions_dict') or {}
            k = len(acts)
            if k == 0:
                break
            if k >= 2 and random.random() < 0.35:
                # ---- fork: same state, every action, common random numbers --- #
                snap_pn = copy.deepcopy(env.pn)
                snap_i = env.i
                d_idx = n_decisions(env)      # the forked decision's index
                base_seed = random.randrange(2 ** 31 - 1)
                Rs, Rs2, fans = [], [], []
                for a in range(k):
                    env.pn = copy.deepcopy(snap_pn)
                    env.i = snap_i
                    obs_b = env.pn.get_graph_observation()
                    if len(obs_b.get('actions_dict') or {}) != k:
                        Rs = []
                        break                  # ordering drift -> discard fork
                    random.seed(base_seed)
                    _, _, dn, _, _ = env.step(a)
                    if not dn:
                        rollout_to_end(env, base_seed + 1)
                    ct_b = env.pn.causal_trace
                    w_b, succ_b = structure(ct_b, skip_pp=False)
                    w_a, succ_a = structure(ct_b, skip_pp=True)
                    kb = [s for s in succ_b.get(d_idx, ()) if s != d_idx]
                    ka = [s for s in succ_a.get(d_idx, ()) if s != d_idx]
                    Rs.append(sum(w_b.get((d_idx, s), 0.0) for s in kb))
                    Rs2.append(sum(w_a.get((d_idx, s), 0.0) for s in ka))
                    fans.append(len(kb))
                # which of the k actions is postpone? entries look like
                # (['postpone'], t, None) vs ([(place, binding), ...], t, trans)
                snap_acts = (copy.deepcopy(snap_pn).get_graph_observation()
                             .get('actions_dict') or [])
                is_pp = [bool(e and isinstance(e[0], (list, tuple))
                              and len(e[0]) and e[0][0] == 'postpone')
                         for e in snap_acts]
                # restore and continue the base trajectory
                env.pn = copy.deepcopy(snap_pn)
                env.i = snap_i
                env.pn.get_graph_observation()
                if len(Rs) == k and k >= 2 and len(is_pp) == k:
                    rows.append((k, Rs, fans, is_pp, Rs2))
                    forks += 1
            obs = env.pn.get_graph_observation()
            acts = obs.get('actions_dict') or {}
            if not acts:
                break
            _, _, done, _, _ = env.step(random.randrange(len(acts)))

    print("=" * 74)
    print("%s -- %d fork points" % (label, len(rows)))
    if not rows:
        print("  no usable fork points")
        return
    spreads = np.array([max(r[1]) - min(r[1]) for r in rows])
    fspreads = np.array([max(r[2]) - min(r[2]) for r in rows])
    print("  ALL actions:")
    print("    R identical across actions : %.1f%% of fork points"
          % (100.0 * float(np.mean(spreads <= 1e-9))))
    print("    R spread (max-min)  mean %.4f   median %.4f   max %.4f"
          % (spreads.mean(), np.median(spreads), spreads.max()))
    print("    fan-out spread  mean %.3f   max %d"
          % (fspreads.mean(), int(fspreads.max())))

    # PRODUCTION actions only -- postpone re-emits the whole marking under
    # token-flow, so its causal successor set is the entire downstream, which
    # inflates R. This is the same pathology lrq2 fixes by giving postpone the
    # SMDP-TD advantage instead of a lineage credit.
    prod = [[v for v, p in zip(r[1], r[3]) if not p] for r in rows]
    prod = [v for v in prod if len(v) >= 2]
    if prod:
        ps = np.array([max(v) - min(v) for v in prod])
        print("  PRODUCTION actions only (postpone excluded), %d fork points:" % len(prod))
        print("    R identical across actions : %.1f%% of fork points"
              % (100.0 * float(np.mean(ps <= 1e-9))))
        print("    R spread (max-min)  mean %.4f   median %.4f   max %.4f"
              % (ps.mean(), np.median(ps), ps.max()))
    pp = [ (r[1][i], [v for j,v in enumerate(r[1]) if not r[3][j]])
           for r in rows for i,p in enumerate(r[3]) if p ]
    if pp:
        ratio = [a / (np.mean(b) if np.mean(b) else 1.0) for a, b in pp if b]
        print("  postpone R / mean(production R): mean %.2f  median %.2f  (n=%d)"
              % (float(np.mean(ratio)), float(np.median(ratio)), len(ratio)))
    # cgae_cflow2 view: postpone transparent, and postpone is no longer a
    # comparable action here (it gets the SMDP-TD advantage instead), so the
    # meaningful spread is over PRODUCTION actions under the new DAG.
    p2 = [[v for v, pp_ in zip(r[4], r[3]) if not pp_] for r in rows]
    p2 = [v for v in p2 if len(v) >= 2]
    if p2:
        s2 = np.array([max(v) - min(v) for v in p2])
        print("  cgae_cflow2 (postpone TRANSPARENT), production actions, %d fork points:"
              % len(p2))
        print("    R identical across actions : %.1f%% of fork points"
              % (100.0 * float(np.mean(s2 <= 1e-9))))
        print("    R spread (max-min)  mean %.4f   median %.4f   max %.4f"
              % (s2.mean(), np.median(s2), s2.max()))
    allR = np.array([v for r in rows for v in r[1]])
    print("  R overall: mean %.3f  SD %.3f" % (allR.mean(), allR.std()))
    print("  sample fork points (R per action; * = postpone):")
    for r in rows[:6]:
        s = ", ".join("%.3f%s" % (v, "*" if p else "") for v, p in zip(r[1], r[3]))
        print("      k=%-3d [%s]" % (r[0], s))


for label, builder in [
        ("ncopies N=4", lambda: make_n_copies(4, causal_rl=True, allow_postpone=True,
                                              causal_postpone_tokenflow=True)),
        ("s1", lambda: make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True,
                                causal_postpone_tokenflow=True))]:
    probe(label, builder, N_EPISODES)
print("=" * 74)
CausalTraces._cgae_structure = _orig_struct
