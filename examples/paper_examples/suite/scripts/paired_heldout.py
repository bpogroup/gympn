"""Paired comparison of run_pi arms on the 100 held-out days.

Every run_pi summary stores the greedy return of the selected and the final
network on each held-out day. For every arm, the per-day return is averaged
over its training seeds, and two arms are compared day by day (paired t-test
and sign counts), next to Welch's t-test over the seed means.

usage: python scripts/paired_heldout.py env=insurer_alloc N=2 [which=final|selected]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SUITE = Path(__file__).resolve().parents[1]
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ENV, N, WHICH = opts.get("env", "insurer_alloc"), int(opts.get("N", 2)), opts.get("which", "final")
BASE = f"pi_{ENV}_n{N}_R8_d40_mc_fair_mrg_clkn_sh32"
CF = "-".join(["40"] * (3 * N if ENV == "insurer" else N))
ARMS = {
    "DCL": (BASE, lambda f: not f.name.startswith("summary_tau0.25_rn")),
    "component-local loss": (BASE, lambda f: f.name.startswith("summary_tau0.25_rn")),
    "factored, dynamic": (f"{BASE}_cf{CF}_dyn", lambda f: True),
}


def load(d, keep):
    runs = {}
    for f in sorted((SUITE / "results" / d).glob("summary_tau0.25*.json")):
        if keep(f):
            s = json.loads(f.read_text())
            runs[f.stem] = np.array(s[f"heldout_{WHICH}"])
    return runs


anc = json.loads((SUITE / "results" / BASE / "eval_anchors.json").read_text())["heldout"]
lo, hi = anc["random_mean"], anc["heuristic_mean"]
norm = lambda x: (x - lo) / (hi - lo)  # noqa: E731

arms = {k: load(*v) for k, v in ARMS.items()}
print(f"{ENV} N={N}, {WHICH} networks, held-out anchors random {lo:.2f} heuristic {hi:.2f}")
for k, runs in arms.items():
    seeds = [norm(r.mean()) for r in runs.values()]
    print(f"  {k:22s} {len(runs)} seeds  mean {np.mean(seeds):.3f}  "
          f"({', '.join(f'{x:.3f}' for x in seeds)})")
names = list(arms)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = arms[names[j]], arms[names[i]]
        da, db = np.mean(list(a.values()), axis=0), np.mean(list(b.values()), axis=0)
        diff = da - db
        t = stats.ttest_rel(da, db)
        w = stats.ttest_ind([norm(r.mean()) for r in a.values()],
                            [norm(r.mean()) for r in b.values()], equal_var=False)
        print(f"  {names[j]} - {names[i]}: {diff.mean():+.2f} cases/day "
              f"({norm(da.mean()) - norm(db.mean()):+.3f} normalized), "
              f"days better/worse {int((diff > 0).sum())}/{int((diff < 0).sum())}, "
              f"paired p={t.pvalue:.3g}; Welch over seeds p={w.pvalue:.3g}")
