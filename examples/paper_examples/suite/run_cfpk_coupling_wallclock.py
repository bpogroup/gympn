r"""Real-env wall-clock comparison: cfpk WITHOUT vs WITH coupling truncation
(gympn/counterfactual.py's _paired_coupled_suffixes, cf_config
['coupling_truncate']) -- the empirical follow-up to
_test_coupling_truncation.py's controlled-net proof (18->12 env.step calls,
exact gap/se match). CFPK_EXPLAINED.md's documented cost baseline (~8.5
min/cell vs ~2.4 min/cell no-fork, ~3.6x) was measured on the full paper grid
protocol (paper_config: 30 epochs x 20 episodes/epoch, length=10); this script
reuses that EXACT protocol on one representative JOINT (contested-decision,
fork-rich) topology so the wall-clock numbers are apples-to-apples with that
baseline, at a seed count (3, not 10) that keeps this a quick real check
rather than a full paper-scale run.

Both arms use method="cfpk" (so _make_args's cf_fork_prob/anneal/etc all
resolve identically); they're distinguished by a SEPARATE logdir_base per
arm (not a fake method tag -- unlike run_phi_shaping_s1_full.py, cfpk's
fork behavior is itself gated on the literal method string "cfpk", so it
must stay literal here).

Run: python run_cfpk_coupling_wallclock.py [num_workers]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import paper_config  # noqa: E402
from run_suite import (train_cell, compute_baselines,  # noqa: E402
                       _default_workers, _threads_per_worker)

ENV = "g_exclusive_choice_joint"   # joint/contested topology -> genuine forks
SEEDS = 12
ARMS = ("cfpk_off", "cfpk_on")
COUPLING = {"cfpk_off": False, "cfpk_on": True}
OUT = Path("suite_results_cfpk_coupling_wallclock")


def _make_cfg(arm):
    cfg = paper_config()
    cfg.envs = [ENV]
    cfg.methods = ["cfpk"]
    cfg.seeds = SEEDS
    cfg.cf_coupling_truncate = COUPLING[arm]
    cfg.output_dir = OUT / arm
    return cfg


def _worker(payload):
    import torch
    arm, seed, cfg, logdir, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        t0 = time.time()
        m = train_cell(ENV, "cfpk", seed, cfg, logdir)
        m["minutes"] = (time.time() - t0) / 60.0
        m["arm"] = arm
        return (arm, seed, m, None)
    except Exception:
        import traceback
        return (arm, seed, None, traceback.format_exc())


def main(num_workers=None):
    if num_workers is None:
        num_workers = _default_workers()

    cells_dir = OUT / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    logdir = str(OUT / "train")
    base_cfg = paper_config()

    bpath = OUT / f"baselines_{ENV}.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
    else:
        b = compute_baselines(ENV, base_cfg)
        bpath.write_text(json.dumps(b))
    r, h = b["random_mean"], b["heuristic_mean"]
    print(f"[cfpk_ct] {ENV}: random={r:.2f} heuristic(anchor)={h:.2f}  "
          f"num_workers={num_workers}", flush=True)

    def norm(v):
        return (v - r) / (h - r) if (v is not None and h != r) else None

    pending = []
    for arm in ARMS:
        cfg = _make_cfg(arm)
        for seed in range(SEEDS):
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if not cell.exists():
                pending.append((arm, seed, cfg))
    print(f"[cfpk_ct] {len(pending)} cells pending "
          f"({len(ARMS)} arms x {SEEDS} seeds, {base_cfg.epochs} epochs x "
          f"{base_cfg.episodes_per_epoch} eps/epoch -- the same protocol scale "
          f"CFPK_EXPLAINED.md's 3.6x baseline was measured at)", flush=True)

    tpw = _threads_per_worker(num_workers)
    t0 = time.time()
    if num_workers <= 1 or not pending:
        results = [_worker((arm, seed, cfg, logdir, tpw)) for (arm, seed, cfg) in pending]
        for arm, seed, m, err in results:
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if err is not None:
                print(f"[cfpk_ct] !!! {arm} s{seed} FAILED:\n{err}", flush=True)
                continue
            cell.write_text(json.dumps(m))
            print(f"[cfpk_ct] <<< {arm} s{seed} greedy_final={m.get('greedy_final')} "
                  f"({m['minutes']:.1f} min)", flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        payloads = [(arm, seed, cfg, logdir, tpw) for (arm, seed, cfg) in pending]
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                arm, seed, m, err = fut.result()
                if err is not None:
                    print(f"[cfpk_ct] !!! {arm} s{seed} FAILED:\n{err}", flush=True)
                    continue
                cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
                cell.write_text(json.dumps(m))
                print(f"[cfpk_ct] <<< {arm} s{seed} greedy_final={m.get('greedy_final')} "
                      f"({m['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    print(f"\n[cfpk_ct] === {ENV}, {base_cfg.epochs} epochs x "
          f"{base_cfg.episodes_per_epoch} eps/epoch, {SEEDS} seeds "
          f"({(time.time()-t0)/60:.1f} min this pass) ===")

    per_min, per_norm, per_coupled = {}, {}, {}
    for arm in ARMS:
        mins, finals, coupled = [], [], []
        for seed in range(SEEDS):
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if cell.exists():
                mm = json.loads(cell.read_text())
                if mm.get("minutes") is not None:
                    mins.append(mm["minutes"])
                gf = mm.get("greedy_final")
                if gf is not None:
                    finals.append(norm(gf))
        per_min[arm] = mins
        per_norm[arm] = finals
        if mins:
            print(f"  {arm:<10} mean wall-clock={np.mean(mins):.2f} min/cell "
                  f"(n={len(mins)}, values={[round(x,2) for x in mins]})")
        if finals:
            print(f"  {arm:<10} norm_final={np.mean(finals):.3f}+-{np.std(finals):.2f} "
                  f"(n={len(finals)})")

    if per_min.get("cfpk_off") and per_min.get("cfpk_on"):
        off_mean = np.mean(per_min["cfpk_off"])
        on_mean = np.mean(per_min["cfpk_on"])
        speedup = off_mean / on_mean if on_mean > 0 else float("nan")
        print(f"\n  WALL-CLOCK: cfpk_off={off_mean:.2f} min/cell  "
              f"cfpk_on={on_mean:.2f} min/cell  speedup={speedup:.2f}x")
    if per_norm.get("cfpk_off") and per_norm.get("cfpk_on"):
        d = np.mean(per_norm["cfpk_on"]) - np.mean(per_norm["cfpk_off"])
        print(f"  OUTCOME (should be ~0, coupling truncation is exact): "
              f"norm_final(on) - norm_final(off) = {d:+.3f}")


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)
