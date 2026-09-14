r"""Full-budget stoch_config() convergence-speed comparison: plain 'ppo_clip'
vs 'ppo_clip' + potential-based reward shaping (gympn/potential.py) on
s1_stoch_sequence -- the first empirical (not just safety) test of whether the
structural-backlog Phi actually helps, now that the mechanism itself is
theorem-safe and unit-tested (see causal-stability-suite memory, 2026-07-31).

phi_coef is an orthogonal knob on SuiteConfig, not a `method` string, so this
script runs two arms (both method='ppo_clip', differing only in phi_coef)
rather than sweeping it inside run_suite's normal envs x methods x seeds grid.

Reuses run_suite.py's train_cell + baseline machinery directly (same
resumable, cached-baseline approach as every other real run in this project).

Run: python run_phi_shaping_s1_full.py [num_workers]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import (train_cell, compute_baselines,  # noqa: E402
                       _default_workers, _threads_per_worker)

ENV = "s1_stoch_sequence"
# arm tags are OUR bookkeeping only -- all call train_cell with method=<arm>,
# which _make_args resolves to plain PPO regardless of the exact string (see
# _worker's docstring). "ppo_clip" (phi_coef=0.0) cells are REUSED from the
# phi_coef=1.0 run (resumable cache, same baseline); "ppo_clip_phi1" is that
# run's OLD phi_coef=1.0 arm, kept under its own tag so it isn't silently
# overwritten/confused with the new phi_coef=0.2 arm below (0W/10L vs
# ppo_clip, hurt both finals and convergence -- see causal-stability-suite
# memory 2026-07-31/08-01). "ppo_clip_phi02" is the new, weaker-coefficient
# arm being tested now, on the hypothesis that phi_coef=1.0's Phi(s_0)~4.99
# (same order as a full episode's raw return) injected too much
# arrival-driven, exogenous variance into the per-step reward for the value
# function to fit well in a finite budget -- 0.2 shrinks that injected
# magnitude by 5x while keeping the theorem-safety property (any phi_coef is
# safe; only the FINITE-SAMPLE learning dynamics are in question here).
# phi_coef=0.2 (10 seeds) came back statistically indistinguishable from
# plain PPO (norm_final +0.027, 5W/5L, p=.484) with a mild reach-rate nudge
# (8/10 vs 6/10 seeds hit 95% of anchor) -- neutral, not a clean win.
# phi_coef=0.5 (10 seeds) trended WORSE (norm_final -0.042, reach-rate only
# 4/10) -- reach-rate is now confirmed MONOTONIC in phi_coef (8/10 -> 6/10
# baseline -> 4/10 -> 1/10 as coef 0.2 -> 0.0 -> 0.5 -> 1.0), reinforcing the
# arrival-driven-variance mechanism over the whole dial, not just the
# endpoints. Now testing the REDESIGN instead of shrinking the coefficient
# further: "ppo_clip_phi1_cap1" caps each place's own contribution at 1
# token (see gympn/potential.py's topology_potential docstring) -- turns Phi
# from "total weighted backlog VOLUME" (unboundedly sensitive to s1's
# exogenous per-timestep arrivals swelling waiting1's queue) into "which
# stages currently have ANY work" (bounded). Tested at phi_coef=1.0
# specifically -- the coefficient that failed WORST uncapped (0W/10L) -- so
# a clean rescue is the most informative single result: if capping fixes
# THIS, it confirms the mechanism AND unlocks the larger, more informative
# shaping magnitude that phi_coef=1.0 gives, instead of settling for
# phi_coef=0.2's weak, statistically-neutral signal.
ARMS = ("ppo_clip", "ppo_clip_phi1_cap1")
PHI = {"ppo_clip": 0.0, "ppo_clip_phi1": 1.0, "ppo_clip_phi02": 0.2,
      "ppo_clip_phi05": 0.5, "ppo_clip_phi1_cap1": 1.0}
CAP = {"ppo_clip_phi1_cap1": 1.0}   # None (absent) = uncapped, every other arm
PHI_DECAY = 0.9
OUT = Path("suite_results_phi_s1")


def _make_cfg(arm):
    cfg = stoch_config()
    cfg.envs = [ENV]
    cfg.methods = ["ppo_clip"]
    cfg.phi_coef = PHI[arm]
    cfg.phi_decay = PHI_DECAY
    cfg.phi_cap = CAP.get(arm)
    cfg.output_dir = OUT / arm
    return cfg


def _worker(payload):
    """Module-level (picklable) worker for ProcessPoolExecutor's spawn context
    -- a closure defined inside main() cannot be pickled.

    Passes `arm` (not the literal "ppo_clip") as train_cell's `method` --
    _make_args's checkpoint dir name is cfg.cell_id(env, method, seed), so
    both arms sharing "ppo_clip" would collide on the SAME logdir path
    (confirmed by a smoke run: FileExistsError racing two workers writing
    s1_stoch_sequence__ppo_clip__s0 simultaneously). Safe to use an arbitrary
    arm string here: _make_args's "algorithm" field is hardcoded to
    "ppo-clip" (not derived from `method`), and "ppo_clip_phi" isn't in the
    causal-scheme method set, so it still resolves to causal_rl=False, plain
    PPO -- only phi_coef (already baked into `cfg`) differs.
    """
    import torch
    arm, seed, cfg, logdir, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        t0 = time.time()
        m = train_cell(ENV, arm, seed, cfg, logdir)
        m["minutes"] = (time.time() - t0) / 60.0
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
    base_cfg = stoch_config()

    bpath = OUT / f"baselines_{ENV}.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
    else:
        b = compute_baselines(ENV, base_cfg)
        bpath.write_text(json.dumps(b))
    r, h = b["random_mean"], b["heuristic_mean"]
    print(f"[phi_s1] {ENV}: random={r:.2f} heuristic(anchor)={h:.2f}  "
          f"phi_decay={PHI_DECAY}  num_workers={num_workers}", flush=True)

    def norm(v):
        return (v - r) / (h - r) if (v is not None and h != r) else None

    pending = []
    for arm in ARMS:
        cfg = _make_cfg(arm)
        for seed in range(cfg.seeds):
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if not cell.exists():
                pending.append((arm, seed, cfg))
    print(f"[phi_s1] {len(pending)} cells pending "
          f"({len(ARMS)} arms x {base_cfg.seeds} seeds)", flush=True)

    tpw = _threads_per_worker(num_workers)
    t0 = time.time()
    if num_workers <= 1 or not pending:
        results = [_worker((arm, seed, cfg, logdir, tpw)) for (arm, seed, cfg) in pending]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        payloads = [(arm, seed, cfg, logdir, tpw) for (arm, seed, cfg) in pending]
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            results = []
            for fut in as_completed(futs):
                arm, seed, m, err = fut.result()
                if err is not None:
                    print(f"[phi_s1] !!! {arm} s{seed} FAILED:\n{err}", flush=True)
                    continue
                cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
                cell.write_text(json.dumps(m))
                print(f"[phi_s1] <<< {arm} s{seed} greedy_final={m.get('greedy_final')} "
                      f"({m['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)
                results.append((arm, seed, m, None))

    if num_workers <= 1:
        for arm, seed, m, err in results:
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if err is not None:
                print(f"[phi_s1] !!! {arm} s{seed} FAILED:\n{err}", flush=True)
                continue
            cell.write_text(json.dumps(m))
            print(f"[phi_s1] <<< {arm} s{seed} greedy_final={m.get('greedy_final')} "
                  f"({m['minutes']:.1f} min)", flush=True)

    print(f"\n[phi_s1] === {ENV}, {base_cfg.epochs} epochs x "
          f"{base_cfg.episodes_per_epoch} eps/epoch, {base_cfg.seeds} seeds "
          f"({(time.time()-t0)/60:.1f} min this pass) ===")
    per = {}
    for arm in ARMS:
        vals, finals = [], []
        for seed in range(base_cfg.seeds):
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if cell.exists():
                mm = json.loads(cell.read_text())
                gf = mm.get("greedy_final")
                if gf is not None:
                    vals.append(norm(gf))
                    finals.append(gf)
        per[arm] = vals
        if vals:
            print(f"  {arm:<14} norm_final={np.mean(vals):.3f}+-{np.std(vals):.2f} "
                  f"raw={[round(x,1) for x in finals]} n={len(vals)}", flush=True)
    base_arm, phi_arm = ARMS[0], ARMS[1]
    if per.get(base_arm) and per.get(phi_arm):
        common_n = min(len(per[base_arm]), len(per[phi_arm]))
        d = [per[phi_arm][i] - per[base_arm][i] for i in range(common_n)]
        w = sum(x > 0 for x in d)
        try:
            from scipy import stats
            _, p = stats.ttest_rel(per[phi_arm][:common_n], per[base_arm][:common_n])
            p_str = f" paired-t p={p:.3f}"
        except Exception:
            p_str = ""
        print(f"  PAIRED {phi_arm} - {base_arm}: mean={np.mean(d):+.3f} "
              f"{w}W/{common_n-w}L (n={common_n}){p_str}", flush=True)

    # convergence-speed signal: epochs to reach 95% of the anchor
    print("\n  --- convergence speed (epochs to 95% of heuristic anchor) ---")
    for arm in ARMS:
        eps_list = []
        for seed in range(base_cfg.seeds):
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if not cell.exists():
                continue
            mm = json.loads(cell.read_text())
            curve = mm.get("greedy_curve", [])
            epochs_axis = mm.get("greedy_epochs", [])
            target = 0.95 * h
            hit = next((e for e, v in zip(epochs_axis, curve) if v >= target), None)
            if hit is not None:
                eps_list.append(hit)
        if eps_list:
            print(f"  {arm:<14} reached 95% in {np.mean(eps_list):.1f}+-{np.std(eps_list):.1f} "
                  f"epochs (n={len(eps_list)}/{base_cfg.seeds})")
        else:
            print(f"  {arm:<14} never reached 95% of anchor in any seed")


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)
