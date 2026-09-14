r"""Quick smoke of 'ls_hca' on s1_stoch_sequence (the stochastic env every
other lineage scheme has been benchmarked on -- see causal-stability-suite
memory: lrq2 ~0.28, ppo/lcv0 ~0.76, cfpk ~0.849). Small budget, 1-2 seeds --
NOT a real result, just: does it run, does it move off lrq2's floor, is the
hhat table actually getting populated. Compares 'lrq2' (fork-free lineage
baseline, no correction) vs 'ls_hca' (same lineage baseline + the fork-free
hindsight correction on contested reward-types) head to head.

Run: python run_ls_hca_smoke_s1.py [epochs] [episodes_per_epoch] [seeds]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import train_cell, compute_baselines  # noqa: E402

METHODS = ("lrq2", "ls_hca")
ENV = "s1_stoch_sequence"


def main(epochs=4, episodes_per_epoch=8, seeds=2):
    cfg = stoch_config()
    cfg.envs = [ENV]
    cfg.epochs = epochs
    cfg.episodes_per_epoch = episodes_per_epoch
    cfg.test_freq = 1
    cfg.seeds = seeds
    cfg.output_dir = Path("suite_results_ls_hca_smoke")

    out = cfg.output_dir
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    bpath = out / f"baselines_{ENV}.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
    else:
        b = compute_baselines(ENV, cfg)
        bpath.write_text(json.dumps(b))
    r, h = b["random_mean"], b["heuristic_mean"]
    print(f"[smoke] {ENV}: random={r:.2f} heuristic(anchor)={h:.2f}", flush=True)

    def norm(v):
        return (v - r) / (h - r) if h != r else 0.0

    results = {m: [] for m in METHODS}
    t0 = time.time()
    for method in METHODS:
        for seed in range(seeds):
            cell = out / "cells" / f"{ENV}__{method}__s{seed}.json"
            if cell.exists():
                m = json.loads(cell.read_text())
            else:
                ts = time.time()
                m = train_cell(ENV, method, seed, cfg, logdir)
                m["minutes"] = (time.time() - ts) / 60.0
                cell.write_text(json.dumps(m))
            gf = m.get("greedy_final")
            nv = norm(gf) if gf is not None else None
            results[method].append(nv)
            extra = ""
            if method == "ls_hca":
                recs = m.get("ls_hca_records_curve", [])
                sizes = m.get("ls_hca_hhat_size_curve", [])
                extra = f"  ls_hca_records={recs}  hhat_size={sizes}"
            print(f"[smoke] {method} s{seed}: greedy_final={gf} norm={nv} "
                  f"({m.get('minutes', 0):.1f} min){extra}", flush=True)

    print(f"\n[smoke] === {ENV}, {epochs} epochs x {episodes_per_epoch} eps/epoch, "
          f"{seeds} seeds ({(time.time()-t0)/60:.1f} min total) ===")
    for method in METHODS:
        vs = [v for v in results[method] if v is not None]
        if vs:
            print(f"  {method:<8} norm_final = {np.mean(vs):.3f} +- {np.std(vs):.2f}  "
                  f"(n={len(vs)}, values={[round(v, 3) for v in vs]})")
    lrq2_v = [v for v in results["lrq2"] if v is not None]
    hca_v = [v for v in results["ls_hca"] if v is not None]
    if lrq2_v and hca_v:
        print(f"  ls_hca - lrq2 (mean norm_final) = {np.mean(hca_v) - np.mean(lrq2_v):+.3f}  "
              f"(reference: lrq2c lifted lrq2's ~0.28 toward >=0.76 on this env)")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    main(epochs, episodes, seeds)