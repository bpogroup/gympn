
"""R4-use-1 diagnostic (CAUSAL_LINEAGE_RETHINK.md): exact counterfactual
action gaps on s1 via snapshot + common-random-number branching.

Question this answers: at decision states s1 actually visits, how large is
the TRUE gap  Q(s, a) - Q(s, b)  between same-stage assignment categories
(match / cross / generalist), measured by branching the white-box simulator
and continuing with the anchor heuristic — and does that gap exceed its own
Monte-Carlo noise? If the true gaps are noise-level, no consumer interface
can win s1 at this budget (the extraction gap is fundamental); if they are
large where the lineage credits are tiny (~0.06-0.26, clustered), the
foreclosure/opportunity-cost term F1 is confirmed as the binding failure
and R4/R5 (counterfactual / shadow-price lineage) become the method.

Method per branch state (a decision point in a heuristic-driven episode):
  snapshot = deepcopy(env.pn), env.i, random.getstate()
  for each available action CATEGORY (one representative binding each):
      for rep in range(R):
          restore snapshot; random.seed(f(ep, step, rep))   # SAME seed
          fire the category's binding; continue with the heuristic to the
          horizon; record remaining reward (undiscounted + e^{-beta*dt}).
  restore snapshot + random state; continue the base episode.
The rep seed is identical ACROSS categories -> exact common random numbers:
per-rep differences are paired, and the paired std prices the noise.

Registered prediction (CAUSAL_LINEAGE_RETHINK.md R4): s1's true
counterfactual match-vs-cross gaps exceed 3x their paired standard error.

Run:  python diag_s1_counterfactual.py [n_episodes] [reps]   (~minutes)
"""
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite")

from gympn.environment import AEPN_Env
from stoch_envs import make_s1_stoch_sequence, s1_heuristic

BETA = 0.5          # suite SMDP discount rate — for the discounted variant
HORIZON = 20        # suite env_length for s1
MAX_BRANCH_STATES_PER_EP = 4
BRANCH_PROB = 0.6   # probability a candidate decision state becomes a branch


def classify(binding):
    """Category of a non-postpone binding: (stage, pairing) or None."""
    if binding[0] == ['postpone']:
        return None
    tr_id = str(getattr(binding[2], '_id', ''))
    stage = 's1' if tr_id.startswith('start1') else \
            's2' if tr_id.startswith('start2') else None
    if stage is None:
        return None
    task = emp = None
    for _, tok in binding[0]:
        v = getattr(tok, 'value', None)
        if isinstance(v, dict):
            if 'task_type' in v:
                task = v['task_type']
            elif 'code_employee' in v:
                emp = v['code_employee']
    if task is None or emp is None:
        return None
    pair = 'gen' if emp == 2 else ('match' if task == emp else 'cross')
    return f"{stage}_{pair}"


def heuristic_index(acts):
    """Index into actions_dict of the anchor heuristic's choice."""
    choice = s1_heuristic(None, None, bindings=acts)
    if choice is None:
        return len(acts) - 1  # only postpone available
    for i, b in enumerate(acts):
        if b is choice:
            return i
    raise RuntimeError("heuristic returned a binding not in actions_dict")


def fresh_obs(env):
    """Rebuild pn_actions coherently after a snapshot restore (mirrors
    what AEPN_Env.reset does after its own deepcopy)."""
    return env.pn.get_graph_observation()


def play_out(env, obs, t0):
    """Continue with the heuristic to the horizon; return (undisc, disc)
    remaining reward from clock t0."""
    undisc = disc = 0.0
    done = False
    while not done:
        acts = obs['actions_dict']
        idx = heuristic_index(acts)
        obs, r, done, _, _ = env.step(idx)
        if r:
            undisc += r
            disc += r * math.exp(-BETA * max(0.0, env.pn.clock - t0))
    return undisc, disc


def run(n_episodes=30, reps=8):
    t_start = time.time()
    states = []          # one record per branch state
    base_returns = []

    for ep in range(n_episodes):
        random.seed(ep)
        np.random.seed(ep)
        pn = make_s1_stoch_sequence(causal_rl=False, allow_postpone=True)
        pn.length = HORIZON
        env = AEPN_Env(pn)
        obs = env.reset()

        done, step_i, n_branched, ep_ret = False, 0, 0, 0.0
        while not done:
            acts = obs['actions_dict']
            cats = {}
            for i, b in enumerate(acts):
                c = classify(b)
                if c is not None and c not in cats:
                    cats[c] = i

            # branch candidate: >=2 categories within the same stage
            stages = defaultdict(list)
            for c in cats:
                stages[c.split('_')[0]].append(c)
            candidate = any(len(v) >= 2 for v in stages.values())

            if (candidate and n_branched < MAX_BRANCH_STATES_PER_EP
                    and random.random() < BRANCH_PROB):
                n_branched += 1
                snap_pn = copy.deepcopy(env.pn)
                snap_i = env.i
                snap_rnd = random.getstate()
                t0 = env.pn.clock

                rec = {"ep": ep, "step": step_i, "clock": t0,
                       "cats": sorted(cats),
                       "heur_choice": classify(acts[heuristic_index(acts)]),
                       "returns": {}}
                for cat, idx0 in cats.items():
                    per_rep = []
                    for rep in range(reps):
                        env.pn = copy.deepcopy(snap_pn)
                        env.i = snap_i
                        obs_b = fresh_obs(env)
                        # re-locate this category's representative binding in
                        # the RESTORED pn (token identities changed)
                        idx = next(i for i, b in enumerate(obs_b['actions_dict'])
                                   if classify(b) == cat)
                        random.seed(1_000_000 * (ep + 1) + 1000 * step_i + rep)
                        obs2, r, done_b, _, _ = env.step(idx)
                        u = r * 1.0
                        d = r * math.exp(-BETA * max(0.0, env.pn.clock - t0))
                        if not done_b:
                            u2, d2 = play_out(env, obs2, t0)
                            u, d = u + u2, d + d2
                        per_rep.append((u, d))
                    rec["returns"][cat] = per_rep
                states.append(rec)

                # restore and continue the base episode
                env.pn = copy.deepcopy(snap_pn)
                env.i = snap_i
                random.setstate(snap_rnd)
                obs = fresh_obs(env)
                acts = obs['actions_dict']

            idx = heuristic_index(acts)
            obs, r, done, _, _ = env.step(idx)
            ep_ret += r
            step_i += 1
        base_returns.append(ep_ret)

    # ------------------------------------------------------------------ #
    # analysis                                                           #
    # ------------------------------------------------------------------ #
    print(f"\n[diag] {len(states)} branch states from {n_episodes} episodes, "
          f"{reps} CRN reps, {time.time()-t_start:.0f}s")
    print(f"[diag] heuristic base return: {np.mean(base_returns):.2f} "
          f"± {np.std(base_returns):.2f}  (anchor ~14.8)")

    def pair_table(kind):
        k = 0 if kind == "undisc" else 1
        print(f"\n=== paired counterfactual gaps ({kind}), same-stage pairs, "
              f"CRN-paired over {reps} reps ===")
        print(f"{'pair':<24}{'n_states':>9}{'mean_gap':>10}{'med_gap':>9}"
              f"{'mean_SE':>9}{'med|g|/SE':>11}{'sig@2SE':>9}{'sign+':>7}")
        agg = defaultdict(list)
        for rec in states:
            by_stage = defaultdict(list)
            for c in rec["returns"]:
                by_stage[c.split('_')[0]].append(c)
            for stage, cs in by_stage.items():
                for a in cs:
                    for b in cs:
                        if a >= b:   # each unordered pair once, a<b lexicographic
                            continue
                        ra = [x[k] for x in rec["returns"][a]]
                        rb = [x[k] for x in rec["returns"][b]]
                        dif = np.array(ra) - np.array(rb)
                        se = dif.std(ddof=1) / math.sqrt(len(dif)) if len(dif) > 1 else float('nan')
                        agg[f"{a} - {b}"].append((dif.mean(), se))
        order = ["s1_match - s1_cross" if "s1_cross - s1_match" not in agg else "s1_cross - s1_match"]
        for key in sorted(agg):
            gaps = np.array([g for g, _ in agg[key]])
            ses = np.array([s for _, s in agg[key]])
            ratio = np.abs(gaps) / np.where(ses > 0, ses, np.nan)
            sig = np.mean(np.abs(gaps) > 2 * ses)
            signpos = np.mean(gaps > 0)
            print(f"{key:<24}{len(gaps):>9}{gaps.mean():>10.3f}"
                  f"{np.median(gaps):>9.3f}{np.nanmean(ses):>9.3f}"
                  f"{np.nanmedian(ratio):>11.2f}{sig:>9.0%}{signpos:>7.0%}")
        return agg

    pair_table("undisc")
    agg_d = pair_table("disc")

    # the registered prediction: match vs cross, either stage
    print("\n=== verdict vs registered prediction "
          "(match-vs-cross gaps > 3x paired SE) ===")
    for key, entries in sorted(agg_d.items()):
        if "match" in key and "cross" in key:
            gaps = np.array([g for g, _ in entries])
            ses = np.array([s for _, s in entries])
            ratio = np.abs(gaps) / np.where(ses > 0, ses, np.nan)
            frac3 = np.nanmean(ratio > 3)
            print(f"  {key}: median |gap|/SE = {np.nanmedian(ratio):.2f}, "
                  f"states with |gap|>3SE: {frac3:.0%}  "
                  f"({'PREDICTION HOLDS' if np.nanmedian(ratio) > 3 else 'BELOW 3x'})")

    out = {"n_episodes": n_episodes, "reps": reps, "beta": BETA,
           "horizon": HORIZON, "base_returns": base_returns,
           "states": [{**rec, "returns": {c: [list(x) for x in v]
                                          for c, v in rec["returns"].items()}}
                      for rec in states]}
    path = r"C:\Users\lobia\PycharmProjects\gympn\examples\paper_examples\suite\diag_s1_counterfactual.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"\n[diag] raw per-state data -> {path}")


if __name__ == "__main__":
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    run(n_ep, reps)