
# gympn/planners/dcl_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import torch

@dataclass
class PlannerConfig:
    horizon: int = 5
    rollouts_per_action: int = 32
    gamma: float = 1.0
    temperature: float = 1.0    # softmax temp for target_pi
    use_crn: bool = True
    use_lineage: bool = True
    beta: float = 0.0
    """SMDP time-discount rate: a reward at clock t is discounted by
    exp(-beta*(t - t0)) from the plan's clock t0, NOT by flat per-step gamma.
    This is the ONLY thing that penalises postpone for the wall-clock it
    burns; with flat gamma the planner collapses to always-postpone on
    queueing envs (the entire project's postpone-attractor history; beta=0.5
    is the established fix). 0.0 = legacy flat gamma."""
    value_bootstrap: bool = True
    """Bootstrap the rollout tail with the critic when the horizon is reached
    without termination. Essential on delayed-reward envs where a short
    horizon otherwise sees no reward (the X15 failure). Mirrors cfpk."""

    # ---- structural lineage use (compute_target_pi_lineage) ----------------
    # These are the jobs a scalar advantage cannot express, which is why they
    # live in the planner rather than in a credit scheme. All are opt-out.
    lineage_share: bool = True
    """Harvest a rollout for every INDEPENDENT candidate it happens to fire,
    not just the one it started with. Two candidates that consume disjoint
    tokens do not interact, so one simulation estimates both. This is the
    saving no model-free method can express: the lineage is what tells you
    whose reward is whose inside a single trajectory."""

    lineage_prune: bool = True
    """Two-phase budget allocation: probe every candidate cheaply, then spend
    the remaining rollouts only on candidates whose own lineage actually
    produced reward in-horizon. Candidates that cannot earn within the
    horizon are not worth distinguishing from each other by simulation."""

    lineage_truncate: bool = True
    """Stop a rollout once the candidate's descendant set is closed (no live
    descendant token remains in the marking). Its causal influence has ended,
    so every later reward is off-lineage. EXACT for the restricted estimand,
    hence only applied together with lineage_tally."""

    lineage_tally: bool = False
    """Score candidates by their lineage-restricted return instead of the raw
    return. Lower variance (X13: -11% paired SE, p=.002) but BIASED wherever
    a decision's value lies in what it prevented elsewhere -- on the s1
    foreclosure env this cost 8% of final performance. Default OFF: the
    planner's whole advantage is that it evaluates alternatives directly, so
    it does not need the lineage to carry the opportunity cost. Turn on only
    for direct-dominated envs (grid/E1)."""

    probe_frac: float = 0.25
    """Fraction of the per-action budget spent in the pruning probe phase."""

def _shape_reward(reward: float, info: Dict[str, Any], lineage: Optional[object]) -> float:
    # If you prefer episodic redistribution in buffer.finish(), just return reward here.
    if lineage is None:
        return float(reward)
    if isinstance(info, dict) and 'eligibility_credits' in info:
        #disabled for now
        pass
        #return float(reward) + float(info['eligibility_credits'])
    return float(reward)


def _disc_at(env, t0, cfg, step_idx):
    """Discount factor for a reward observed right now. SMDP e^{-beta*(t-t0)}
    keyed on the sim clock when cfg.beta>0 (penalises postpone for the time it
    burns), else the legacy flat gamma**step_idx."""
    if cfg.beta > 0.0:
        t = float(getattr(env.pn, 'clock', t0))
        return float(np.exp(-cfg.beta * max(0.0, t - t0)))
    return float(cfg.gamma ** step_idx)


def _sample_continuation(actor, s, enabled_t, cfg):
    """Pick a continuation action by sampling the CURRENT policy over the
    enabled positions (falls back to uniform if the actor can't score them).

    Uniform continuation was the X15 failure: on a delayed-reward env a
    random tail swamps the first-action signal, so the target policy came out
    near-uniform and the distilled greedy policy fell below random. A
    policy-guided tail is the standard MCTS/DCL choice and gives each
    candidate's value the credit of a competent continuation."""
    if actor is None:
        return int(np.random.choice(enabled_t))
    try:
        probs = actor(s).flatten()
        idx = torch.as_tensor(enabled_t, dtype=torch.long)
        p = torch.clamp(probs[idx], min=1e-8)
        p = (p / p.sum()).cpu().numpy()
        return int(np.random.choice(enabled_t, p=p))
    except Exception:
        return int(np.random.choice(enabled_t))


def _value_tail(critic, s, disc, cfg):
    """Discounted critic estimate of the return beyond the rollout horizon.

    Without it a short horizon on a delayed-reward env sees mostly zero
    reward and cannot rank first actions (the other half of the X15 failure).
    Mirrors cfpk's value-tail bootstrap. Zero when no critic is available or
    the rollout already terminated."""
    if critic is None or not cfg.value_bootstrap:
        return 0.0
    try:
        v = critic(s)
        return float(disc) * float(v.reshape(-1).mean().item())
    except Exception:
        return 0.0


# gympn/planners/dcl_planner.py
@torch.no_grad()
def compute_target_pi(env, state_graph, actor, cfg: PlannerConfig, lineage=None,
                      critic=None):
    enabled = env.enabled_actions(state_graph)  # positions in pn.pn_actions
    A = len(enabled)
    if A == 0:
        return np.array([]), {'means': [], 'stds': []}, enabled
    if A == 1:
        return np.array([1.0], dtype=np.float32), {'means': [0.0], 'stds': [0.0]}, enabled

    seeds = [np.random.randint(0, 2**31-1) for _ in range(cfg.rollouts_per_action)] if cfg.use_crn else [None] * cfg.rollouts_per_action
    means = np.zeros(A, dtype=np.float32)
    stds  = np.zeros(A, dtype=np.float32)
    snapshot = env.get_state() if hasattr(env, "get_state") else None

    for i, env_idx0 in enumerate(enabled):
        rets = []
        for sd in seeds:
            if snapshot is not None and hasattr(env, "set_state"):
                env.set_state(snapshot)
            if sd is not None and hasattr(env, "set_seed"):
                env.set_seed(sd)

            t0 = float(getattr(env.pn, 'clock', 0.0))
            # Step 0: take the candidate env position
            s, rwd, done, trunc, info = env.step(env_idx0)
            total = _disc_at(env, t0, cfg, 0) * _shape_reward(rwd, info, lineage)

            # Steps 1..H-1: continue under the CURRENT POLICY (not uniform)
            for t in range(1, cfg.horizon):
                if done or trunc:
                    break
                enabled_t = env.enabled_actions(s)  # positions in pn_actions
                if len(enabled_t) == 0:
                    break
                a_pos = _sample_continuation(actor, s, enabled_t, cfg)
                s, rwd, done, trunc, info = env.step(a_pos)
                total += _disc_at(env, t0, cfg, t) * _shape_reward(rwd, info, lineage)

            # Bootstrap the tail with the critic when we ran out of horizon
            # rather than terminating (delayed reward lives beyond H).
            if not done and not trunc:
                total += _value_tail(critic, s, _disc_at(env, t0, cfg, cfg.horizon), cfg)

            rets.append(total)

        means[i] = float(np.mean(rets))
        stds[i]  = float(np.std(rets))

    logits = means / max(cfg.temperature, 1e-8)
    logits -= logits.max()
    exps = np.exp(logits)
    target_pi = exps / (exps.sum() if exps.sum() > 0 else 1.0)

    # CRITICAL: restore the env to the planning state. Rollouts mutated it via
    # set_state/step; without this the caller resumes the real episode from a
    # random rollout-terminal state, corrupting every trajectory (the X15
    # below-random bug). compute_target_pi_lineage already does this.
    if snapshot is not None and hasattr(env, "set_state"):
        env.set_state(snapshot)

    return target_pi.astype(np.float32), {'means': means.tolist(), 'stds': stds.tolist()}, enabled


# ----------------------------------------------------------------------- #
# Structural lineage planner                                              #
# ----------------------------------------------------------------------- #

def _binding_signature(binding):
    """Identity of a candidate that survives across steps: the transition it
    fires plus the tokens it consumes. Positional indices into pn_actions are
    NOT stable between steps, but token ids are, so this is how a candidate
    enabled at t0 is recognised again later inside a rollout."""
    if binding is None or (isinstance(binding[0], list) and binding[0] == ['postpone']):
        return None
    try:
        tr = getattr(binding[2], '_id', None)
        toks = tuple(sorted(t._id for _, t in binding[0]))
        return (tr, toks)
    except Exception:
        return None


def _consumed_tokens(sig):
    return frozenset(sig[1]) if sig else frozenset()


def _live_descendant(pn, desc):
    """Is any descendant token still present in the marking? While one is,
    the decision's causal influence is unfinished."""
    for place in pn.places:
        for tok in place.marking:
            if getattr(tok, '_id', None) in desc:
                return True
    return False


def _lineage_rollout(env, snapshot, seed, first_idx, cfg, t0_sigs,
                     indep_of_first, actor=None, critic=None):
    """One rollout. Returns (total_return, {sig: lineage_return}, earned_flag).

    ``t0_sigs`` maps a candidate's signature -> its index among the enabled
    actions at t0. Any such candidate fired during this rollout AND
    independent of the starting one is harvested: its own lineage-restricted
    return is a legitimate estimate for it, because disjoint lineages do not
    interact (lineage_share).
    """
    from gympn.counterfactual import descendants_of

    env.set_state(snapshot)
    if seed is not None:
        env.set_seed(seed)
    pn = env.pn
    ct = getattr(pn, 'causal_trace', None)
    traced = ct is not None
    if traced:
        ct.flush()
    t_clock = float(getattr(pn, 'clock', 0.0))

    t0 = t_clock
    s, rwd, done, trunc, info = env.step(first_idx)
    total = _disc_at(env, t0, cfg, 0) * float(rwd)
    # roots of the candidate we started with
    roots = {}
    if traced and ct.transition_history.transitions:
        rec0 = ct.transition_history.transitions[0]
        roots[0] = set(rec0.get('output_tokens', ()))

    harvest = {}          # sig -> (root token ids, index among enabled)
    step_i = 0
    for t in range(1, cfg.horizon):
        step_i = t
        if done or trunc:
            break
        if (cfg.lineage_truncate and cfg.lineage_tally and traced and roots
                and not _live_descendant(pn, descendants_of(ct, roots[0]))):
            # candidate's influence is spent; nothing further can be in-lineage
            break
        enabled_t = env.enabled_actions(s)
        if not enabled_t:
            break
        a_pos = _sample_continuation(actor, s, enabled_t, cfg)

        sig_now = None
        if cfg.lineage_share and traced:
            try:
                sig_now = _binding_signature(pn.pn_actions[a_pos])
            except Exception:
                sig_now = None
        n_before = len(ct.transition_history.transitions) if traced else 0

        s, rwd, done, trunc, info = env.step(a_pos)
        total += _disc_at(env, t0, cfg, t) * float(rwd)

        if (sig_now is not None and sig_now in t0_sigs
                and sig_now in indep_of_first and sig_now not in harvest
                and traced and len(ct.transition_history.transitions) > n_before):
            rec = ct.transition_history.transitions[n_before]
            harvest[sig_now] = set(rec.get('output_tokens', ()))

    # tally lineage-restricted returns
    lin_returns = {}
    earned = False
    if traced:
        def _tally(root_ids):
            desc = descendants_of(ct, root_ids)
            tot = 0.0
            for tr in ct.transition_history.transitions:
                r = tr.get('reward', 0.0)
                if r and any(tid in desc for tid in tr.get('input_tokens', ())):
                    tot += r
            return tot

        if roots:
            own = _tally(roots[0])
            lin_returns['__self__'] = own
            earned = own != 0.0
        for sig, rts in harvest.items():
            lin_returns[sig] = _tally(rts)

    # value-tail bootstrap for the RAW total only (the tail is off-lineage
    # by construction, so it must not enter the restricted tallies)
    if not done and not trunc:
        total += _value_tail(critic, s, _disc_at(env, t0, cfg, cfg.horizon), cfg)

    return total, lin_returns, earned


@torch.no_grad()
def compute_target_pi_lineage(env, state_graph, actor, cfg: PlannerConfig,
                              lineage=None, critic=None):
    """DCL target policy with the lineage used STRUCTURALLY.

    Four uses, none of which has any expression as a scalar advantage:
      sharing    - one rollout scores every independent candidate it fires
      pruning    - budget goes to candidates that can actually earn in-horizon
      truncation - stop once the candidate's descendant set is closed
      tally      - optional lineage-restricted scoring (biased on foreclosure
                   envs; see PlannerConfig.lineage_tally)

    Falls back to plain rollout scoring when the env records no causal trace.
    """
    enabled: List[int] = env.enabled_actions(state_graph)
    A = len(enabled)
    if A == 0:
        return np.array([], dtype=np.float32), {'means': [], 'stds': []}, enabled
    if A == 1:
        return (np.array([1.0], dtype=np.float32),
                {'means': [0.0], 'stds': [0.0], 'rollouts': 0}, enabled)

    snapshot = env.get_state()
    bindings = list(getattr(env.pn, 'pn_actions', []))
    sigs = [(_binding_signature(bindings[j]) if j < len(bindings) else None)
            for j in enabled]
    t0_sigs = {s: i for i, s in enumerate(sigs) if s is not None}
    consumed = [_consumed_tokens(s) for s in sigs]
    # candidate j is independent of candidate i when they consume no common
    # token: neither disables or delays the other, so their lineages are
    # separable and one rollout can score both
    indep = [{sigs[j] for j in range(A)
              if j != i and sigs[j] is not None
              and not (consumed[i] & consumed[j])}
             for i in range(A)]

    samples: List[List[float]] = [[] for _ in range(A)]
    earned_any = [False] * A
    n_rollouts = 0

    def _seeds(k):
        return ([int(np.random.randint(0, 2 ** 31 - 1)) for _ in range(k)]
                if cfg.use_crn else [None] * k)

    def _run(idx_list, k):
        nonlocal n_rollouts
        seeds = _seeds(k)
        for i in idx_list:
            for sd in seeds:
                tot, lin, earned = _lineage_rollout(
                    env, snapshot, sd, enabled[i], cfg, t0_sigs, indep[i],
                    actor=actor, critic=critic)
                n_rollouts += 1
                score = (lin.get('__self__', 0.0) if cfg.lineage_tally else tot)
                samples[i].append(float(score))
                earned_any[i] = earned_any[i] or earned
                if cfg.lineage_share:
                    for sig, val in lin.items():
                        if sig == '__self__':
                            continue
                        j = t0_sigs.get(sig)
                        if j is not None and j != i:
                            # only legitimate as a score under the restricted
                            # estimand; a raw total would double-count the
                            # starting candidate's own contribution
                            if cfg.lineage_tally:
                                samples[j].append(float(val))

    total_budget = max(1, int(cfg.rollouts_per_action))
    if cfg.lineage_prune and total_budget >= 4:
        probe = max(1, int(round(total_budget * cfg.probe_frac)))
        _run(range(A), probe)
        # keep only candidates that demonstrably earn in-lineage; if the
        # lineage says none of them can, the probe already ranked them and
        # extra rollouts would only resolve off-lineage differences
        live = [i for i in range(A) if earned_any[i]]
        if not live or len(live) == A:
            live = list(range(A))
        rest = total_budget - probe
        if rest > 0 and live:
            extra = max(1, int(round(rest * A / max(1, len(live)))))
            _run(live, extra)
    else:
        _run(range(A), total_budget)

    means = np.array([float(np.mean(s)) if s else 0.0 for s in samples],
                     dtype=np.float32)
    stds = np.array([float(np.std(s)) if len(s) > 1 else 0.0 for s in samples],
                    dtype=np.float32)

    logits = means / max(cfg.temperature, 1e-8)
    logits -= logits.max()
    exps = np.exp(logits)
    target_pi = exps / (exps.sum() if exps.sum() > 0 else 1.0)

    env.set_state(snapshot)
    return (target_pi.astype(np.float32),
            {'means': means.tolist(), 'stds': stds.tolist(),
             'rollouts': n_rollouts,
             'samples': [len(s) for s in samples],
             'pruned': int(A - sum(earned_any)) if cfg.lineage_prune else 0},
            enabled)


# ----- Optional: Sequential Halving flavor for large action sets -----

@torch.no_grad()
def sequential_halving_target_pi(env, state_graph, actor, cfg: PlannerConfig,
                                 rounds: int = 3, lineage: Optional[object] = None
                                ) -> Tuple[np.ndarray, Dict[str, Any], List[int]]:
    enabled: List[int] = env.enabled_actions(state_graph)
    if len(enabled) <= 1:
        return (np.array([1.0], dtype=np.float32) if enabled else np.array([], dtype=np.float32),
                {'rounds': []}, enabled)

    S: List[int] = list(enabled)  # survivors
    round_stats: List[Dict[str, Any]] = []
    snapshot = env.get_state() if hasattr(env, "get_state") else None

    for r in range(rounds):
        seeds = [np.random.randint(0, 2**31 - 1) for _ in range(cfg.rollouts_per_action)]
        means: List[float] = []
        for a0 in S:
            rets: List[float] = []
            for sd in seeds:
                if snapshot is not None:
                    env.set_state(snapshot)
                if hasattr(env, "set_seed"):
                    env.set_seed(sd)

                s, rwd, done, trunc, info = env.step(a0)
                total = _shape_reward(rwd, info, lineage)
                disc = cfg.gamma
                for t in range(1, cfg.horizon):
                    if done or trunc: break
                    probs = actor(s)
                    enabled_t = env.enabled_actions(s)
                    if len(enabled_t) == 0: break
                    p_vec = torch.clamp(probs.flatten()[enabled_t], min=1e-8)
                    p_vec = (p_vec / p_vec.sum()).cpu().numpy()
                    a_t = int(np.random.choice(enabled_t, p=p_vec))
                    s, rwd, done, trunc, info = env.step(a_t)
                    total += disc * _shape_reward(rwd, info, lineage)
                    disc  *= cfg.gamma
                rets.append(float(total))
            means.append(float(np.mean(rets)))

        round_stats.append({'round': r, 'actions': S.copy(), 'means': means.copy()})
        # keep top half
        idx = np.argsort(means)[::-1]
        keep = max(1, int(np.ceil(len(S) / 2)))
        S = [S[i] for i in idx[:keep]]
        if len(S) == 1:
            break

    # final target over last survivors
    last_means = np.array(round_stats[-1]['means'], dtype=np.float32)
    logits = last_means / max(cfg.temperature, 1e-8)
    logits -= logits.max()
    exps = np.exp(logits)
    target = exps / (exps.sum() if exps.sum() > 0 else 1.0)
    return target.astype(np.float32), {'rounds': round_stats}, S
