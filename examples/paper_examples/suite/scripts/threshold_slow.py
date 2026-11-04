"""Threshold policies on the slow-mismatch insurer (Lin and Kumar, 1984): assign
a mismatched underwriter only when at least k tasks of its type are waiting,
and wait otherwise (envs/insurer_env.py, make_threshold_heuristic).

Every threshold k = 1, 2, ... and the waiting heuristic (k = inf) are scored on
the 20 evaluation days (on which the best k is selected) and on the 100
held-out days, normalized with the anchors of the slow-mismatch runs (0 =
random, 1 = waiting heuristic), with a paired test against the waiting
heuristic over the held-out days.

usage: python scripts/threshold_slow.py [N=1] [kmax=6] [anchors=<eval_anchors.json>]
Writes results/threshold_slow_n<N>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
N, KMAX = int(opts.get("N", 1)), int(opts.get("kmax", 6))
ANCH = Path(opts.get("anchors", SUITE / "results" / f"pi_insurer_slow_n{N}_R8_d40_mc_ppc_fair_mrg_clkn_sh32"
                     / "eval_anchors.json"))

from heldout import make_env, score_solver, EVAL_SEED, EVAL_DAYS, HELDOUT_SEED, HELDOUT_DAYS  # noqa: E402
from insurer_env import INSURER_BUILDERS, make_threshold_heuristic, make_wait_heuristic  # noqa: E402

build = INSURER_BUILDERS["insurer_slow"]
mismatch = build.uw_mismatch
anc = json.loads(ANCH.read_text())
norm = {s: (lambda x, a=anc[s]: (np.asarray(x) - a["random_mean"]) / (a["heuristic_mean"] - a["random_mean"]))
        for s in ("eval", "heldout")}
env = make_env(build, N, 20, True, 'component', marking_certificate=True)

rules = {f"k={k}": make_threshold_heuristic(mismatch, k) for k in range(1, KMAX + 1)}
rules["k=inf (waiting heuristic)"] = make_wait_heuristic(mismatch)
out = {}
for name, h in rules.items():
    ev = score_solver(env, EVAL_SEED, EVAL_DAYS, 'heuristic', h)
    ho = score_solver(env, HELDOUT_SEED, HELDOUT_DAYS, 'heuristic', h)
    out[name] = {"eval": ev, "heldout": ho, "eval_norm": float(np.mean(norm["eval"](ev))),
                 "heldout_norm": float(np.mean(norm["heldout"](ho)))}
    print(f"{name:28s} eval {out[name]['eval_norm']:.3f}   held-out {out[name]['heldout_norm']:.3f}", flush=True)

wait = np.asarray(anc["heldout"]["heuristic"])
best = max((n for n in out if n.startswith("k=") and "inf" not in n), key=lambda n: out[n]["eval_norm"])
ho = np.asarray(out[best]["heldout"])
t = stats.ttest_rel(ho, wait)
better, worse = int(np.sum(ho > wait)), int(np.sum(ho < wait))
print(f"\nbest threshold on the evaluation days: {best}; held-out {out[best]['heldout_norm']:.3f}, "
      f"{(ho - wait).mean():+.2f} cases per shift vs the waiting heuristic, "
      f"better on {better} days, worse on {worse}, p = {t.pvalue:.3g}")
out["_selected"] = {"best": best, "heldout_norm": out[best]["heldout_norm"], "diff_per_shift": float((ho - wait).mean()),
                    "better_days": better, "worse_days": worse, "p_paired": float(t.pvalue), "anchors": str(ANCH)}
(SUITE / "results" / f"threshold_slow_n{N}.json").write_text(json.dumps(out, indent=1))
