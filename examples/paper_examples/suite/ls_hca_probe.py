r"""Prototype of FORKFREE_LINEAGE_RETHINK.md Idea 1: Lineage-Structured
Hindsight Credit Assignment (LS-HCA) -- the fork-free dual of the DAG-replay
counterfactual (m2_counterfactual_probe.py / m4_counterfactual_probe.py).

    A(s,a) =  Sum_{r in PURE(d)}      r * disc                     # exact, keep
            + Sum_{r in CONTESTED(d)} r * disc * (1 - pi(a|s)/hhat(a|s,r))
            + 0 * Sum_{r in EXOGENOUS}                              # dropped, exact

PURE/EXOGENOUS are read off the provenance DAG's STATIC structure (no
estimation). CONTESTED is the only place we estimate: hhat(a|s, "reward-type r
realized in d's lineage") is fit from a batch of FACTUAL trajectories only --
no simulator forks.

The decisive experiment (FORKFREE_LINEAGE_RETHINK.md, bottom section):

  1. M2 (assembly_probe.make_shared_r): the shared-resource join where ccf/lrq
     flip. Does LS-HCA's d1(use_R - standalone) keep mc_q's unbiased SIGN?
  2. M4 (assembly_probe.make_two_chains): the abundant-resource RCPSP motif
     where s_ccf is merely conservative (keeps r_b). Does LS-HCA's empirical
     hhat drive r_b's correction factor to ~0 (factoring it out), matching cf
     and beating s_ccf, while staying unbiased on the r_hi/r_lo effect?

Because M2/M4 are deterministic nets, a "batch of factual trajectories" only
needs one trajectory per target action (more episodes would just repeat the
same (action -> lineage-membership) pair) -- this is the noiseless limit a
logistic/empirical fit converges to as the batch grows; see fit_hindsight().
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import types, uuid
import gympn
from simpn.simulator import SimToken
from gympn.environment import AEPN_Env
from assembly_probe import make_shared_r, make_two_chains, run_forced


# --------------------------------------------------------------------------- #
# Static classification: PURE(d) / CONTESTED(d) reward-types for a decision  #
# --------------------------------------------------------------------------- #

def _reach_by_action_type(pn):
    """Static forward-reachability of each action TYPE to reward-transition
    TYPES (topological, action-invariant). Same walk as the library's
    CausalTraces._static_component_reward_types(), but returns the PER-ACTION
    reach sets un-merged -- LS-HCA needs the PURE/CONTESTED split relative to
    one decision, not s_ccf's all-decisions component union."""
    trans = list(pn.actions) + list(pn.events)
    consumers = {}
    for t in trans:
        for p in t.incoming:
            consumers.setdefault(p._id, []).append(t)
    reward_types = set(pn.reward_functions.keys())

    def reaches(t):
        reached, seen_t, seen_p, stack = set(), set(), set(), list(t.outgoing)
        if t._id in reward_types:
            reached.add(t._id)
        while stack:
            p = stack.pop()
            if p._id in seen_p:
                continue
            seen_p.add(p._id)
            for ct in consumers.get(p._id, []):
                if ct._id in reward_types:
                    reached.add(ct._id)
                if ct._id not in seen_t:
                    seen_t.add(ct._id); stack.extend(ct.outgoing)
        return reached

    return {a._id: reaches(a) for a in pn.actions}


def classify(pn, target_action_names):
    """PURE(d) = reward-types ONLY d's own actions can statically reach.
    CONTESTED(d) = reward-types d shares reachability with some OTHER action
    type. EXOGENOUS(d) is everything else (not returned -- callers just don't
    credit it). This is a sound over-approximation: reach() flows through any
    place a reward could structurally pass, including other actions' resource
    hand-backs, so it can mark something CONTESTED that is actually pure (see
    M4 below) -- costs a wasted hhat fit, never a bias (FORKFREE_LINEAGE_
    RETHINK.md's "safe degradation" argument)."""
    reach = _reach_by_action_type(pn)
    reach_d = set().union(*(reach[a] for a in target_action_names))
    others = [a for a in reach if a not in target_action_names]
    other = set().union(*(reach[a] for a in others)) if others else set()
    pure = reach_d - other
    contested = reach_d & other
    return pure, contested


# --------------------------------------------------------------------------- #
# Realized lineage membership (per reward instance, for ONE decision index)  #
# --------------------------------------------------------------------------- #

def _lineage_membership(pn, target_idx=0):
    """For every realized reward instance in this trajectory:
    (value, time, reward_type, is-in-target_idx's-realized-token-lineage).
    Same token-parent walk as the library's _redistribute_lrq, specialised to
    a single decision index instead of "which decisions does this belong to"."""
    ct = pn.causal_trace
    action_transitions = ct.transition_history.get_action_transitions()
    token_to_action = {}
    record_to_action = {}
    for idx, act in enumerate(action_transitions):
        record_to_action[id(act)] = idx
        for tid in act.get('output_tokens', []):
            token_to_action[tid] = idx

    def get_parents(tid):
        t = ct.token_history.get_token(tid)
        return t.get('parents', []) if t else []

    base = pn._get_string_before_last_dot
    out = []
    for tr in ct.transition_history.transitions:
        reward = tr.get('reward', 0.0)
        if reward == 0.0:
            continue
        rtype = base(getattr(tr.get('transition'), '_id', ''))
        firing_idx = record_to_action.get(id(tr))
        found = set()
        if firing_idx is not None:
            found.add(firing_idx)
        seen, stack = set(), list(tr.get('input_tokens', []))
        while stack:
            tid = stack.pop()
            if tid in seen:
                continue
            seen.add(tid)
            hit = token_to_action.get(tid)
            if hit is not None:
                found.add(hit)
            for p in get_parents(tid):
                if p not in seen:
                    stack.append(p)
        out.append((reward, tr.get('time'), rtype, target_idx in found))
    return out


# --------------------------------------------------------------------------- #
# Run helper: like assembly_probe.run_forced, but also hands back the pn     #
# --------------------------------------------------------------------------- #

def run_forced_pn(choice_name, make_fn, beta=0.0, length=12, **rw):
    gympn.seed_everything(0)
    pn = make_fn(causal_rl=True, **rw)
    pn.length = length
    for p in pn.places:
        for t in p.marking:
            setattr(t, '_id', str(uuid.uuid4()))
    pn.causal_trace.flush()
    sen = types.SimpleNamespace(_id="__initial__")
    toks = [t for p in pn.places for t in p.marking]
    for t in toks:
        pn.causal_trace.register_token(t, sen, [], time=0)
    pn.causal_trace.register_transition(sen, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn); env.reset()
    total = 0.0
    forced = False
    for _ in range(30):
        acts = env.pn.pn_actions
        if not acts:
            break
        idx = 0
        for i, a in enumerate(acts):
            tname = getattr(a[2], '_id', getattr(a[2], 'name', '')) if isinstance(a, tuple) and len(a) > 2 and a[2] is not None else ''
            if choice_name in str(tname):
                idx = i; forced = True; break
        _, r, done, _, _ = env.step(idx)
        total += float(r)
        if done:
            break
    return total, env.pn, forced


# --------------------------------------------------------------------------- #
# Hindsight model: fit hhat(a | s, reward-type r realized in d's lineage)    #
# --------------------------------------------------------------------------- #

def fit_hindsight(make_fn, target_names, contested, beta, **rw):
    """Estimate hhat from a batch of FACTUAL trajectories only (no forks): one
    per target action (M2/M4 are deterministic, so more episodes per action
    would just repeat the same lineage-membership outcome -- this IS the
    noiseless limit of a logistic fit as batch size grows). pi is uniform over
    the target alternatives (the behavior policy used to collect the batch)."""
    n = len(target_names)
    pi = {a: 1.0 / n for a in target_names}
    z_by_action = {}
    for a in target_names:
        _, pn, forced = run_forced_pn(a, make_fn, beta=beta, **rw)
        assert forced, f"could not force {a}"
        mem = _lineage_membership(pn, target_idx=0)
        z_by_action[a] = {rtype: z for (_, _, rtype, z) in mem if rtype in contested}

    hhat = {}
    for rtype in contested:
        for z in (True, False):
            support = [a for a in target_names if z_by_action[a].get(rtype, False) == z]
            mass = sum(pi[a] for a in support)
            for a in target_names:
                if a in support:
                    hhat[(a, rtype, z)] = pi[a] / mass
                else:
                    hhat[(a, rtype, z)] = 1e-6  # a never realizes this z -> ~0
    return hhat, pi


# --------------------------------------------------------------------------- #
# LS-HCA credit for one forced trajectory                                    #
# --------------------------------------------------------------------------- #

def ls_hca_credit(pn, target_action, pure, contested, hhat, pi, beta):
    ct = pn.causal_trace
    u = ct.transition_history.get_action_transitions()[0].get('time')

    def disc(t_j):
        if beta == 0.0 or t_j is None or u is None:
            return 1.0
        return math.exp(-beta * max(0.0, float(t_j) - float(u)))

    mem = _lineage_membership(pn, target_idx=0)
    total, bd = 0.0, {'pure': 0.0, 'contested': 0.0}
    for (rv, t_j, rtype, z) in mem:
        if u is not None and t_j is not None and t_j < u:
            continue
        if rtype in pure:
            c = rv * disc(t_j)
            total += c; bd['pure'] += c
        elif rtype in contested:
            h = hhat.get((target_action, rtype, z), pi[target_action])
            factor = 1.0 - pi[target_action] / max(h, 1e-9)
            c = rv * disc(t_j) * factor
            total += c; bd['contested'] += c
        # else: EXOGENOUS -- dropped, exact.
    return total, bd


# --------------------------------------------------------------------------- #
# Probe driver                                                               #
# --------------------------------------------------------------------------- #

def probe(name, make_fn, target_names, actionA, actionB, beta=0.0, **rw):
    pn0 = make_fn(causal_rl=True, **rw)
    pure, contested = classify(pn0, target_names)
    print(f"\n=== {name}  (beta={beta}, {rw}) ===")
    print(f"    PURE(d)={sorted(pure)}   CONTESTED(d)={sorted(contested)}")

    hhat, pi = fit_hindsight(make_fn, target_names, contested, beta, **rw)
    for rtype in sorted(contested):
        parts = ", ".join(f"hhat({a}|z={int(z)})={hhat[(a, rtype, z)]:.2f}"
                          for a in target_names for z in (True, False))
        print(f"    {rtype:>10}: {parts}")

    _, pnA, fA = run_forced_pn(actionA, make_fn, beta=beta, **rw)
    _, pnB, fB = run_forced_pn(actionB, make_fn, beta=beta, **rw)
    assert fA and fB
    credA, bdA = ls_hca_credit(pnA, actionA, pure, contested, hhat, pi, beta)
    credB, bdB = ls_hca_credit(pnB, actionB, pure, contested, hhat, pi, beta)

    _, credsA, _ = run_forced(actionA, make_fn, beta=beta, **rw)
    _, credsB, _ = run_forced(actionB, make_fn, beta=beta, **rw)
    ref = float(credsA['mc_q'][0] - credsB['mc_q'][0])

    diff = credA - credB
    sign_ok = (diff > 0) == (ref > 0) if abs(ref) > 1e-9 else True
    print(f"    LS-HCA credit({actionA})={credA:+.3f}  (pure={bdA['pure']:+.3f}, contested={bdA['contested']:+.3f})")
    print(f"    LS-HCA credit({actionB})={credB:+.3f}  (pure={bdB['pure']:+.3f}, contested={bdB['contested']:+.3f})")
    print(f"    LS-HCA d({actionA[:1]}-{actionB[:1]}) = {diff:+.3f}   "
          f"mc_q(unbiased ref) = {ref:+.3f}   [{'SIGN MATCH' if sign_ok else 'FLIP'}]")
    for s in ('lrq', 'ccf', 's_ccf'):
        d = float(credsA[s][0] - credsB[s][0])
        flip = abs(ref) > 1e-9 and (d > 0) != (ref > 0)
        print(f"    {s:>6} d = {d:+.3f}   [{'FLIP' if flip else 'ok'}]")
    return diff, ref, bdA, bdB


if __name__ == "__main__":
    print("LS-HCA (Idea 1, FORKFREE_LINEAGE_RETHINK.md): fork-free, provenance-scoped")
    print("Hindsight Credit Assignment. Decisive experiment: M2 sign-recovery, M4 r_b-drop.")

    # ---- 1. M2: shared-resource join where ccf/lrq flip. ----
    print("\n" + "=" * 70)
    print("M2 shared-R join -- does LS-HCA's d1(use_R - standalone) keep the sign?")
    print("=" * 70)
    for b in (0.0, 0.3):
        probe("M2 shared-R join", make_shared_r, {'use_R', 'standalone'},
              'use_R', 'standalone', beta=b, r_join=5, r1=5, r2=10)

    # ---- 2. M4: abundant resource where s_ccf is merely conservative. ----
    print("\n" + "=" * 70)
    print("M4 abundant shared resource -- does LS-HCA drop r_b (contested-but-uncontended)?")
    print("=" * 70)
    diff, ref, bdA, bdB = probe("M4 abundant resource", make_two_chains, {'A_hi', 'A_lo'},
                                'A_hi', 'A_lo', beta=0.0, r_hi=5, r_lo=2, r_b=4)
    print(f"\n    true effect (r_hi - r_lo) = +3.0; r_b=4 should NOT appear in the (A-B) "
          f"difference if LS-HCA factors it out (it's identical under both actions, so any "
          f"correctly-factored contested term cancels in the difference regardless of its "
          f"absolute size -- the diagnostic is the ABSOLUTE credit vs mc_q's r_b-carrying level).")
    _, credsA, _ = run_forced('A_hi', make_two_chains, beta=0.0, r_hi=5, r_lo=2, r_b=4)
    mcq_level = float(credsA['mc_q'][0])
    print(f"    mc_q(A_hi) [carries r_b]        = {mcq_level:.2f}")
    print(f"    LS-HCA credit(A_hi) [pure+contested] = {(bdA['pure']+bdA['contested']):.2f}   "
          f"[{'factors r_b out' if (bdA['pure']+bdA['contested']) < mcq_level - 1e-6 else 'keeps r_b (conservative)'}]")