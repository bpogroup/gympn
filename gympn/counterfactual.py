"""G1 — forked counterfactual preferences (CAUSAL_LINEAGE_RETHINK.md §7.2).

At a visited decision state, fork the white-box simulator once for the taken
action and once for one alternative (the policy's highest-probability
non-taken action), run 2-3 common-random-number sibling suffixes each with a
greedy continuation, and record the paired gap in beta-discounted remaining
reward. A preference (winner, loser) is emitted only when the mean paired
gap exceeds ``gate`` times its own paired standard error — the SNR gate
that turns clustered/noisy gaps into NO signal instead of a noisy gradient.

Everything here is PN-formalism-general: bindings come from the env's
``actions_dict``, the alternative is chosen from the policy's own logits,
the continuation is the greedy current policy, and no token-value semantics
are read anywhere.

Cost controls: suffixes are truncated once the branch clock advances
``lookahead`` beyond the fork time (with a e^{-beta*dt} * V(s) value-tail
bootstrap from the agent's critic — principled because the gap is measured
in beta-discounted units, so the truncated tail carries e^{-beta*lookahead}
of the mass), and forks are capped per episode.

Snapshot mechanics mirror suite/diag_s1_counterfactual.py (validated
2026-07-18): deepcopy(env.pn) + env.i is a complete mid-episode snapshot,
and env.pn.get_graph_observation() after a restore rebuilds ``pn_actions``
coherently with the restored token identities.
"""
import copy
import math
import random as _random

import numpy as np
import torch

_TRACE_ERR = (
    "[cf] lineage mode is on but the env is not recording a causal trace. "
    "The ENV must be built with causal_rl=True (trace recording is gated "
    "on it in simulator.fire/postpone, and causal_trace is only created "
    "when it is set); the agent may still be non-causal. See "
    "run_suite.train_cell's env_causal.")


def _discounted_suffix(agent, env, obs, first_idx, t0, beta, lookahead,
                       value_tail=True):
    """Fire ``first_idx`` then continue greedily; return the beta-discounted
    remaining reward from clock t0, value-tail-bootstrapped at truncation.

    ``value_tail=False`` drops the V(s) bootstrap (plain truncation). Kept as
    an independent switch so the lineage ablation can be isolated: the
    lineage variant must drop the tail by construction (V predicts the FULL
    return), so a lineage-vs-raw comparison would otherwise confound the two
    changes. V is itself a learned, noisy function that differs between
    branches, so it contributes variance to the paired gap on its own.
    """
    obs2, r, done, _, _ = env.step(first_idx)
    disc = r * math.exp(-beta * max(0.0, float(env.pn.clock) - t0))
    while not done:
        if float(env.pn.clock) - t0 > lookahead:
            v = 0.0
            if value_tail:
                try:
                    with torch.no_grad():
                        v = float(agent.value_model(obs2).reshape(-1)[0])
                except Exception:
                    v = 0.0
            disc += math.exp(-beta * (float(env.pn.clock) - t0)) * v
            break
        a = agent.act(obs2, deterministic=True)
        obs2, r, done, _, _ = env.step(a)
        disc += r * math.exp(-beta * max(0.0, float(env.pn.clock) - t0))
    return disc


def _lineage_return(ct, roots, root_rec_id, t0, beta):
    """Discounted sum of ONLY those branch rewards causally descended from the
    forked decision (the lineage restriction, applied inside the fork).

    ``roots`` = ids of the tokens the forked action produced. A token is a
    descendant if any of its parents is; since ``token_history.tokens`` is
    insertion-ordered and a parent is always registered before its child, one
    forward pass computes the whole descendant set. A rewarding transition
    counts iff it consumed a descendant (or is the forked firing itself).

    This is the same ancestry test ``_redistribute_lrq`` applies over a
    factual episode, keyed on one decision instead of all of them.
    """
    desc = descendants_of(ct, roots)

    total = 0.0
    for tr in ct.transition_history.transitions:
        rew = tr.get("reward", 0.0)
        if not rew:
            continue
        if id(tr) != root_rec_id and not any(
                tid in desc for tid in tr.get("input_tokens", ())):
            continue
        t = tr.get("time")
        dt = max(0.0, float(t) - t0) if t is not None else 0.0
        total += rew * math.exp(-beta * dt)
    return total


def descendants_of(ct, roots):
    """Forward-closure of ``roots`` over the token DAG.

    ``token_history.tokens`` is insertion-ordered and a parent is always
    registered before its child, so one forward pass suffices. Shared by the
    fork machinery and the DCL planner.
    """
    desc = set(roots)
    for tid, rec in ct.token_history.tokens.items():
        if tid in desc:
            continue
        for p in rec.get("parents", ()):
            if p in desc:
                desc.add(tid)
                break
    return desc


def _branch_tally(ct, roots, root_rec_id, t0, beta):
    """One branch, decomposed: (total, lineage, feats).

    ``lineage`` is the direct effect (rewards descended from the forked
    decision); ``total - lineage`` is the indirect / opportunity-cost effect.
    ``feats`` are LINEAGE-DERIVED conditioning statistics for the indirect
    channel: how long and how heavily this decision ties the system up. They
    are near-deterministic under CRN (service durations are shared draws),
    which is the whole point — Rao-Blackwellizing on a statistic no smoother
    than the reward itself would buy nothing.
    """
    desc = descendants_of(ct, roots)

    total = lin = 0.0
    t_close = t0
    for tr in ct.transition_history.transitions:
        t = tr.get("time")
        dt = max(0.0, float(t) - t0) if t is not None else 0.0
        disc = math.exp(-beta * dt)
        in_lin = (id(tr) == root_rec_id or
                  any(tid in desc for tid in tr.get("input_tokens", ())))
        rew = tr.get("reward", 0.0)
        if rew:
            total += rew * disc
            if in_lin:
                lin += rew * disc
        if in_lin and t is not None:
            t_close = max(t_close, float(t))

    disc_desc = 0.0
    for tid in desc:
        rec = ct.token_history.tokens.get(tid)
        if rec and rec.get("time") is not None:
            disc_desc += math.exp(
                -beta * max(0.0, float(rec["time"]) - t0))

    feats = {
        # promptness with which the decision's own causal activity closes —
        # i.e. how soon whatever it occupied comes back into play
        "occ_disc": math.exp(-beta * (t_close - t0)),
        "occ_dur": float(t_close - t0),
        "n_desc": float(len(desc)),
        "disc_desc": disc_desc,
    }
    return total, lin, feats


# Differenced occupancy statistics, plus their interaction with the system's
# CONGESTION at the fork. The interaction matters because occupancy only has
# an opportunity cost when something else is waiting for the capacity: tying a
# resource up for 3 time units is expensive under a long queue and free under
# an empty one. Congestion is shared by both branches (it is a property of the
# fork state), so it enters only as a scale on the differenced terms.
FEAT_KEYS = ("occ_disc", "occ_dur", "n_desc", "disc_desc",
             "occ_dur_x_cong", "occ_disc_x_cong")


def _decomp_suffix(agent, env, obs, first_idx, t0, beta, lookahead):
    """Branch rollout returning the decomposed tally (no value tail: V
    predicts the full unrestricted return, which would contaminate the
    direct channel)."""
    ct = getattr(env.pn, 'causal_trace', None)
    if ct is None:
        raise RuntimeError(_TRACE_ERR)
    ct.flush()

    obs2, r, done, _, _ = env.step(first_idx)
    trans = ct.transition_history.transitions
    if not trans:
        raise RuntimeError(_TRACE_ERR)
    roots = set(trans[0].get("output_tokens", ()))
    root_rec_id = id(trans[0])

    while not done and float(env.pn.clock) - t0 <= lookahead:
        a = agent.act(obs2, deterministic=True)
        obs2, r, done, _, _ = env.step(a)

    return _branch_tally(ct, roots, root_rec_id, t0, beta)


def resolve_decomp_preferences(records, gate, ridge=1.0, min_r2=0.05,
                               fit_pool=None):
    """Turn a batch of decomposed fork records into preferences.

    The direct channel is used as sampled (low variance). The indirect
    channel is NOT: its per-fork sample is the noisy term, so it is replaced
    by a ridge regression on the lineage-derived occupancy features, POOLED
    across the batch — the fitted value has far lower variance than the
    sample it replaces, which is where the reduction comes from.

    Safety floor: if the fit does not generalize (held-out R^2 < ``min_r2``)
    the regression is discarded and the raw total gap is used instead, i.e.
    the method degrades to plain (non-decomposed) counterfactual preferences
    rather than inventing a new way to be biased.

    ``fit_pool`` (default: ``records``) is the sample the regression is fit
    and validated on. It is normally a ROLLING WINDOW over recent epochs:
    the occupancy->opportunity-cost relationship is a property of the
    system's congestion, which drifts slowly, whereas one epoch's forks are
    far too few to validate a 5-parameter fit (held-out R^2 goes negative
    from variance alone). Pooling buys the sample size; the window keeps it
    adaptive.

    Returns (prefs, stats).
    """
    if not records:
        return [], {"n": 0, "mode": "empty", "r2": float("nan"),
                    "se": float("nan"), "pass_rate": 0.0, "n_fit": 0}

    pool = list(fit_pool) if fit_pool else list(records)

    def _design(recs):
        m = np.array([[r["phi"][k] for k in FEAT_KEYS] for r in recs], float)
        return np.hstack([m, np.ones((len(m), 1))])   # intercept

    X = _design(pool)
    y = np.array([r["gap_ind"] for r in pool], float)

    def _fit(xs, ys):
        A = xs.T @ xs + ridge * np.eye(xs.shape[1])
        return np.linalg.solve(A, xs.T @ ys)

    # 2-fold held-out R^2 — the honest test of whether the conditioning
    # statistic actually predicts the opportunity-cost channel.
    r2 = float("nan")
    if len(pool) >= 8:
        idx = np.arange(len(pool))
        scores = []
        for hold in (idx % 2 == 0, idx % 2 == 1):
            tr, te = ~hold, hold
            if tr.sum() < 2 or te.sum() < 2:
                continue
            w = _fit(X[tr], y[tr])
            resid = y[te] - X[te] @ w
            var = ((y[te] - y[te].mean()) ** 2).sum()
            if var > 1e-12:
                scores.append(1.0 - (resid ** 2).sum() / var)
        if scores:
            r2 = float(np.mean(scores))

    use_model = np.isfinite(r2) and r2 >= min_r2
    prefs, passed = [], 0
    if use_model:
        w = _fit(X, y)
        resid_std = float(np.std(y - X @ w, ddof=1)) if len(y) > 1 else 0.0
        se_pred = resid_std / math.sqrt(len(y))       # pooled, hence small
        pred = _design(records) @ w                   # applied to THIS epoch
        for rec, p in zip(records, pred):
            est = rec["gap_direct"] + float(p)
            se = math.sqrt(rec["se_direct"] ** 2 + se_pred ** 2)
            if abs(est) > 1e-9 and abs(est) > gate * se:
                passed += 1
                w_i, l_i = ((rec["action"], rec["alt"]) if est > 0
                            else (rec["alt"], rec["action"]))
                prefs.append({"state": rec["state"], "winner": w_i,
                              "loser": l_i, "gap": est, "se": se})
        mean_se = float(np.mean([
            math.sqrt(r["se_direct"] ** 2 + se_pred ** 2) for r in records]))
    else:
        for rec in records:
            est, se = rec["gap_total"], rec["se_total"]
            if abs(est) > 1e-9 and abs(est) > gate * se:
                passed += 1
                w_i, l_i = ((rec["action"], rec["alt"]) if est > 0
                            else (rec["alt"], rec["action"]))
                prefs.append({"state": rec["state"], "winner": w_i,
                              "loser": l_i, "gap": est, "se": se})
        mean_se = float(np.mean([r["se_total"] for r in records]))

    return prefs, {"n": len(records), "mode": "model" if use_model else "raw",
                   "r2": r2, "se": mean_se, "n_fit": len(pool),
                   "pass_rate": passed / len(records)}


def _lineage_suffix(agent, env, obs, first_idx, t0, beta, lookahead):
    """Lineage-restricted counterpart of ``_discounted_suffix``.

    Same rollout, but the tally counts only rewards in the forked decision's
    causal lineage. Two variance sources are then removed at once: CRN
    pairing cancels the noise SHARED by the branches, the lineage restriction
    removes reward from CONCURRENT, causally-unrelated activity that the raw
    return-to-go sums indiscriminately (the lrq-vs-mc_q effect, 28W/0L on the
    grid, applied inside the counterfactual instead of over a factual trace).

    No value-tail bootstrap at truncation: V(s) predicts the FULL future
    return, so adding it would re-inject exactly the concurrent-reward
    contamination the restriction removes. The tail is dropped for both
    branches alike, so the paired gap stays comparable.
    """
    ct = getattr(env.pn, 'causal_trace', None)
    if ct is None:
        raise RuntimeError(_TRACE_ERR)
    ct.flush()   # branch-local trace: ancestry roots at the fork, not earlier

    obs2, r, done, _, _ = env.step(first_idx)
    trans = ct.transition_history.transitions
    if not trans:
        raise RuntimeError(_TRACE_ERR)
    # The action fires before run_evolutions, so record 0 is the forked firing.
    roots = set(trans[0].get("output_tokens", ()))
    root_rec_id = id(trans[0])

    while not done and float(env.pn.clock) - t0 <= lookahead:
        a = agent.act(obs2, deterministic=True)
        obs2, r, done, _, _ = env.step(a)

    return _lineage_return(ct, roots, root_rec_id, t0, beta)


def maybe_fork(agent, env, state, action, logpis, cfg, rng=_random):
    """Possibly fork the current decision state into (taken, alternative).

    Returns (forked, pref, diag): ``forked`` is True when a fork was executed
    (for the per-episode budget), ``pref`` is a preference record
    {'state', 'winner', 'loser', 'gap', 'se'} or None (gate not passed), and
    ``diag`` is {'gap', 'se', 'passed'} mechanism telemetry for every executed
    fork (None when no fork ran). The env and the ``rng`` stream are restored
    to their pre-fork state either way.
    """
    acts = state.get('actions_dict') if isinstance(state, dict) else None
    pn = getattr(env, 'pn', None)
    if acts is None or pn is None or len(acts) < 2:
        return False, None, None
    if rng.random() > cfg['fork_prob']:
        return False, None, None

    # Alternative = highest-probability action that was not taken (general:
    # logits exist for every binding, postpone included).
    order = torch.argsort(logpis.detach().reshape(-1), descending=True).tolist()
    alt = next((i for i in order if i != action), None)
    if alt is None:
        return False, None, None

    beta = cfg['beta']
    reps = cfg['reps']
    base_seed = rng.randrange(2 ** 31 - 1)
    snap_pn = copy.deepcopy(env.pn)
    snap_i = env.i
    snap_rnd = rng.getstate()
    t0 = float(getattr(env.pn, 'clock', 0.0))
    n_actions = len(acts)

    lineage = cfg.get('lineage', False)
    decompose = cfg.get('decompose', False)
    value_tail = cfg.get('value_tail', True)

    if decompose:
        # Decomposed mode: the preference cannot be decided here, because the
        # indirect channel is resolved by a regression POOLED over the epoch's
        # forks. Return a record; resolve_decomp_preferences() finishes it.
        try:
            tallies = {action: [], alt: []}
            for idx in (action, alt):
                for rep in range(reps):
                    env.pn = copy.deepcopy(snap_pn)
                    env.i = snap_i
                    obs_b = env.pn.get_graph_observation()
                    if len(obs_b['actions_dict']) != n_actions:
                        return True, None, None
                    rng.seed(base_seed + rep)
                    tallies[idx].append(_decomp_suffix(
                        agent, env, obs_b, idx, t0, beta, cfg['lookahead']))

            tot = {k: np.array([t[0] for t in v]) for k, v in tallies.items()}
            lin = {k: np.array([t[1] for t in v]) for k, v in tallies.items()}

            def _se(d):
                return (float(d.std(ddof=1) / math.sqrt(len(d)))
                        if len(d) > 1 else float('inf'))

            d_dir = lin[action] - lin[alt]
            d_tot = tot[action] - tot[alt]
            d_ind = d_tot - d_dir
            base_keys = ("occ_disc", "occ_dur", "n_desc", "disc_desc")
            phi = {k: float(np.mean([t[2][k] for t in tallies[action]])
                            - np.mean([t[2][k] for t in tallies[alt]]))
                   for k in base_keys}
            try:
                cong = float(sum(len(p.marking) for p in snap_pn.places))
            except Exception:
                cong = 0.0
            phi["occ_dur_x_cong"] = phi["occ_dur"] * cong
            phi["occ_disc_x_cong"] = phi["occ_disc"] * cong

            rec = {'state': state, 'action': action, 'alt': alt,
                   'gap_direct': float(d_dir.mean()), 'se_direct': _se(d_dir),
                   'gap_ind': float(d_ind.mean()),
                   'gap_total': float(d_tot.mean()), 'se_total': _se(d_tot),
                   'phi': phi}
            return True, rec, {'gap': float(d_tot.mean()), 'se': _se(d_tot),
                               'passed': False}
        finally:
            env.pn = copy.deepcopy(snap_pn)
            env.i = snap_i
            rng.setstate(snap_rnd)
            env.pn.get_graph_observation()

    try:
        returns = {action: [], alt: []}
        for idx in (action, alt):
            for rep in range(reps):
                env.pn = copy.deepcopy(snap_pn)
                env.i = snap_i
                obs_b = env.pn.get_graph_observation()
                if len(obs_b['actions_dict']) != n_actions:
                    # restored ordering unexpectedly diverged — abort fork
                    return True, None, None
                rng.seed(base_seed + rep)  # identical across idx => exact CRN
                if lineage:
                    val = _lineage_suffix(agent, env, obs_b, idx, t0, beta,
                                          cfg['lookahead'])
                else:
                    val = _discounted_suffix(agent, env, obs_b, idx, t0, beta,
                                             cfg['lookahead'],
                                             value_tail=value_tail)
                returns[idx].append(val)

        dif = np.asarray(returns[action]) - np.asarray(returns[alt])
        gap = float(dif.mean())
        se = (float(dif.std(ddof=1) / math.sqrt(len(dif)))
              if len(dif) > 1 else float('inf'))
        pref = None
        # SNR gate: exact zero gaps (branches coupled) carry no preference.
        passed = abs(gap) > 1e-9 and abs(gap) > cfg['gate'] * se
        if passed:
            w, l = (action, alt) if gap > 0 else (alt, action)
            pref = {'state': state, 'winner': w, 'loser': l,
                    'gap': gap, 'se': se}
        return True, pref, {'gap': gap, 'se': se, 'passed': bool(passed)}
    finally:
        env.pn = copy.deepcopy(snap_pn)
        env.i = snap_i
        rng.setstate(snap_rnd)
        env.pn.get_graph_observation()  # rebuild pn_actions for the caller