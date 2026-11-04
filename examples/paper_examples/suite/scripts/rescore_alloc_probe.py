"""Score the 150-epoch allocation runs of results/alloc_long_probe (Table
tab:alloc of the paper) on the 100 held-out days of envs/heldout.py, the days
on which run_pi and run_bpm score their policies, so that they can be compared
with the runs in the DCL observation (options, clock node).

For every PPO and NF-GAE cell: the selected network (best_policy.pth, chosen on
the evaluation episodes during training), the final one (snap_ep150.pth), and
the mean over the last 10 snapshots (epochs 123-150, the analogue of the
table's "mean of the last 10 evaluations"). Component agents are not scored
(their policy is not a single network; rescore_heldout skips them too).
Writes results/alloc_long_probe/heldout_probe.json and prints a summary.

usage: python scripts/rescore_alloc_probe.py [workers=12]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))
sys.path.insert(0, str(SUITE / "scripts"))

from heldout import make_env, anchors, score_policy, HELDOUT_SEED, HELDOUT_DAYS  # noqa: E402
from rescore_heldout import registries, cell_args  # noqa: E402

PROBE = SUITE / "results" / "alloc_long_probe"
GROUPS = {   # table row -> directory pattern
    "PPO": r"ppo_s\d_ep150_beta0\.05_fv$",
    "NF-GAE, marking, forward only": r"nfgae_mc_s\d_ep150_beta0\.05$",
    "NF-GAE, marking, component": r"nfgae_mc_s\d_ep150_beta0\.05_fv$",
}
LAST = [f"snap_ep{e:03d}.pth" for e in range(123, 151, 3)]


def _score(cell_dir):
    import torch
    torch.set_num_threads(1)
    builders, _ = registries()
    tr = next((Path(cell_dir) / "train").iterdir())
    ca = cell_args(tr)
    if "--global_view" not in (tr / "args.txt").read_text():
        ca["global_view"] = False      # trained before the whole-net view existed: forward edges only
    env = make_env(builders["insurer_alloc"], 1, 20, False, 'global',
                   ca["flat_obs"], ca["global_view"], ca["marking_certificate"])
    out = {}
    for key, files in (("selected", ["best_policy.pth"]), ("final", ["snap_ep150.pth"]), ("last10", LAST)):
        days = []
        for f in files:
            pol = torch.load(tr / f, weights_only=False).eval()
            days.append(score_policy(env, pol, HELDOUT_SEED, HELDOUT_DAYS))
        out[key] = np.mean(days, axis=0).tolist()
    return str(cell_dir), out


if __name__ == "__main__":
    w = int(dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a).get("workers", 12))
    builders, heur = registries()
    ho = anchors(make_env(builders["insurer_alloc"], 1, 20), heur["insurer_alloc"], HELDOUT_SEED, HELDOUT_DAYS)
    cells = [d for d in sorted(PROBE.iterdir()) if d.is_dir()
             and any(re.match(p, d.name) for p in GROUPS.values())]
    fo = PROBE / "heldout_probe.json"
    res = json.loads(fo.read_text()) if fo.exists() else {}
    todo = [str(d) for d in cells if str(d) not in res]
    if todo:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
            for cell, out in ex.map(_score, todo):
                res[cell] = out
                fo.write_text(json.dumps(res))
    norm = lambda v: (v - ho["random_mean"]) / (ho["heuristic_mean"] - ho["random_mean"])  # noqa: E731
    print(f"held-out anchors random {ho['random_mean']:.2f} heuristic {ho['heuristic_mean']:.2f}")
    for row, pat in GROUPS.items():
        rs = [v for k, v in res.items() if re.match(pat, Path(k).name)]
        line = "  ".join(f"{key} {np.mean([norm(np.mean(r[key])) for r in rs]):.3f}"
                         for key in ("selected", "final", "last10"))
        print(f"  {row:32s} n={len(rs):2d}  {line}")
