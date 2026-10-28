"""The rollout planner over the heuristic on the 100 held-out days (seeds
888000+i): the heuristic itself and Sequential Halving with 32 rollouts per
option (DCL's lookahead), hidden state redrawn in every rollout (fair).
Per day: return, decisions changed, rollout steps simulated, seconds. Every
finished day is appended to results/planner_heldout_<env>_n<N>.jsonl at once,
so the run resumes where it stopped.

usage: python scripts/planner_heldout.py env=bpic15|insurer [N=1] [workers=8]
                                         [anchors=<eval_anchors.json>]
anchors: normalize with these held-out anchors (e.g. those of the DCL run on
the same env), so the planner and the networks share one scale.
"""
import json
import os
import random
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))

opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ENV, N = opts.get("env", "bpic15"), int(opts.get("N", 1))
WORKERS = int(opts.get("workers", 8))
OUT = SUITE / "results" / f"planner_heldout_{ENV}_n{N}.jsonl"
BASE, DAYS = 888_000, 100
RULES = {"heuristic": None, "sh32": dict(sh_budget=32)}


def _setup():
    from heldout import make_env
    if ENV.startswith("bpic15"):
        from bpic15_env import BPIC15_BUILDERS as B, BPIC15_HEURISTICS as H, bpic15_resample_hidden as rs
    else:
        from insurer_env import INSURER_BUILDERS as B, INSURER_HEURISTICS as H, insurer_resample_hidden as rs
    return make_env(B[ENV], N, 20, marking_certificate=True), H[ENV], rs


def day(task):
    name, i = task
    torch.set_num_threads(1)
    from gympn.policy_iteration import RolloutPlanner, HeuristicBase
    env, heur, resample = _setup()
    base = HeuristicBase(heur)
    pl = None
    if RULES[name] is not None:
        kw = dict(RULES[name]); kw.setdefault("rollouts", 8)
        pl = RolloutPlanner(base, seed=1, resample=resample, **kw)
    random.seed(BASE + i); np.random.seed(BASE + i)
    env.reset()
    t0 = time.time()
    total, step, changed = 0.0, 0, 0
    while env.pn.pn_actions:
        if pl is None:
            a = base(env)
        else:
            a = pl.choose(env, tag=(i, step)); changed += a != pl.last["base"]
        _, r, done, _, _ = env.step(a, build_obs=False)
        total += r; step += 1
        if done:
            break
    rec = {"rule": name, "day": i, "ret": total, "changed": changed, "steps": pl.steps[0] if pl else 0,
           "saved": pl.steps[1] if pl else 0, "sec": time.time() - t0, "decisions": step}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def main():
    done = set()
    if OUT.exists():
        done = {(json.loads(l)["rule"], json.loads(l)["day"]) for l in open(OUT)}
    tasks = [(r, i) for i in range(DAYS) for r in RULES if (r, i) not in done]
    t0 = time.time()
    with mp.get_context("spawn").Pool(WORKERS) as pool:   # as on Windows; fork + torch threads can hang
        for k, _ in enumerate(pool.imap_unordered(day, tasks), 1):
            if k % 10 == 0:
                print(f"  {k}/{len(tasks)} days ({(time.time() - t0) / 60:.0f} min)", flush=True)
    from scipy import stats
    res = [json.loads(l) for l in open(OUT)]
    h = np.array([z["ret"] for z in sorted([z for z in res if z["rule"] == "heuristic"], key=lambda z: z["day"])], float)
    n = lambda v: v
    if "anchors" in opts:
        anc = json.load(open(opts["anchors"]))["heldout"]
        n = lambda v: (v - anc["random_mean"]) / (anc["heuristic_mean"] - anc["random_mean"])
    for r in RULES:
        x = sorted([z for z in res if z["rule"] == r], key=lambda z: z["day"])
        if len(x) < DAYS or r == "heuristic":
            continue
        ret = np.array([z["ret"] for z in x]); d = ret - h
        print(f"{r:9s}: {ret.mean():.2f} = {n(ret.mean()):.3f}; vs heuristic {d.mean():+.2f}/day "
              f"({int((d > 0).sum())}/{int((d < 0).sum())} better/worse, p={stats.ttest_rel(ret, h).pvalue:.2g}); "
              f"{np.mean([z['changed'] for z in x]):.1f} changes/day; rollout steps/day "
              f"{np.mean([z['steps'] for z in x]) / 1e3:.0f}k; {np.mean([z['sec'] for z in x]) / 60:.1f} worker-min/day")


if __name__ == "__main__":
    main()
