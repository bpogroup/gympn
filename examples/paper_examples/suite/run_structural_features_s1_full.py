r"""Full-budget stoch_config() comparison: plain 'ppo_clip' vs 'ppo_clip' +
structural (conflict-graph-derived) INPUT features
(gympn/simulator.py's GymProblem.use_structural_features) on
s1_stoch_sequence -- the first empirical training-outcome comparison for
this mechanism (part 3 of the post-phi-shaping consolidation plan; see
causal-stability-suite memory 2026-08-01). Unlike every credit/reward
mechanism tried in this arc, this one only adds INPUT to the GNN encoder
(per action-type: is it in a structural conflict, and with how many other
actions) -- no bias-variance tradeoff, no GAE-bootstrap fragility. s1's
start1/start2 decision (which shares the 3-employee pool) is exactly the
kind of structurally-contested decision this feature is meant to flag.

Same two-arm-via-method-tag pattern as run_phi_shaping_s1_full.py (not
run_cfpk_coupling_wallclock.py's separate-logdir-base pattern): plain PPO's
behavior doesn't depend on the literal method string beyond the
causal-scheme membership check, so an arbitrary non-causal tag is safe and
keeps both arms' cell/logdir names distinct without colliding.

Run: python run_structural_features_s1_full.py [num_workers]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import (train_cell, compute_baselines,  # noqa: E402
                       _default_workers, _threads_per_worker)

ENV = "s1_stoch_sequence"
ARMS = ("ppo_clip", "ppo_clip_struct")   # tags for OUR bookkeeping only; both
                                        # call train_cell with method='ppo_clip'
USE_STRUCT = {"ppo_clip": False, "ppo_clip_struct": True}
OUT = Path("suite_results_struct_s1")


def _make_cfg(arm):
    cfg = stoch_config()
    cfg.envs = [ENV]
    cfg.methods = ["ppo_clip"]
    cfg.use_structural_features = USE_STRUCT[arm]
    cfg.output_dir = OUT / arm
    return cfg


def _worker(payload):
    """Module-level (picklable) worker -- see run_phi_shaping_s1_full.py's
    _worker docstring for why this can't be a closure inside main()."""
    import torch
    arm, seed, cfg, logdir, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        t0 = time.time()
        m = train_cell(ENV, "ppo_clip", seed, cfg, logdir)
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
    base_cfg = stoch_config()

    bpath = OUT / f"baselines_{ENV}.json"
    if bpath.exists():
        b = json.loads(bpath.read_text())
    else:
        b = compute_baselines(ENV, base_cfg)
        bpath.write_text(json.dumps(b))
    r, h = b["random_mean"], b["heuristic_mean"]
    print(f"[struct_s1] {ENV}: random={r:.2f} heuristic(anchor)={h:.2f}  "
          f"num_workers={num_workers}", flush=True)

    def norm(v):
        return (v - r) / (h - r) if (v is not None and h != r) else None

    pending = []
    for arm in ARMS:
        cfg = _make_cfg(arm)
        for seed in range(cfg.seeds):
            cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
            if not cell.exists():
                pending.append((arm, seed, cfg))
    print(f"[struct_s1] {len(pending)} cells pending "
          f"({len(ARMS)} arms x {base_cfg.seeds} seeds)", flush=True)

    tpw = _threads_per_worker(num_workers)
    t0 = time.time()
    if num_workers <= 1 or not pending:
        for arm, seed, cfg in pending:
            r_, s_, m, err = _worker((arm, seed, cfg, logdir, tpw))
            cell = cells_dir / f"{ENV}__{r_}__s{s_}.json"
            if err is not None:
                print(f"[struct_s1] !!! {r_} s{s_} FAILED:\n{err}", flush=True)
                continue
            cell.write_text(json.dumps(m))
            print(f"[struct_s1] <<< {r_} s{s_} greedy_final={m.get('greedy_final')} "
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
                    print(f"[struct_s1] !!! {arm} s{seed} FAILED:\n{err}", flush=True)
                    continue
                cell = cells_dir / f"{ENV}__{arm}__s{seed}.json"
                cell.write_text(json.dumps(m))
                print(f"[struct_s1] <<< {arm} s{seed} greedy_final={m.get('greedy_final')} "
                      f"({m['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    print(f"\n[struct_s1] === {ENV}, {base_cfg.epochs} epochs x "
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
            print(f"  {arm:<16} norm_final={np.mean(vals):.3f}+-{np.std(vals):.2f} "
                  f"raw={[round(x,1) for x in finals]} n={len(vals)}", flush=True)

    base_arm, struct_arm = ARMS[0], ARMS[1]
    if per.get(base_arm) and per.get(struct_arm):
        common_n = min(len(per[base_arm]), len(per[struct_arm]))
        d = [per[struct_arm][i] - per[base_arm][i] for i in range(common_n)]
        w = sum(x > 0 for x in d)
        try:
            _, p = stats.ttest_rel(per[struct_arm][:common_n], per[base_arm][:common_n])
            p_str = f" paired-t p={p:.3f}"
        except Exception:
            p_str = ""
        print(f"  PAIRED {struct_arm} - {base_arm}: mean={np.mean(d):+.3f} "
              f"{w}W/{common_n-w}L (n={common_n}){p_str}", flush=True)

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
            print(f"  {arm:<16} reached 95% in {np.mean(eps_list):.1f}+-{np.std(eps_list):.1f} "
                  f"epochs (n={len(eps_list)}/{base_cfg.seeds})")
        else:
            print(f"  {arm:<16} never reached 95% of anchor in any seed")


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)
