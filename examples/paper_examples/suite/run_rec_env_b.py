"""Run the REC redistribution scheme on env b (b_sequence_disjoint) with
causal_beta=0.2 and postpone ENABLED — the env where flow_dag/causal-TD0 used to
collapse to always-postpone.

REC is return-equivalent (CAUSAL_REDISTRIBUTION_SOUND_SCHEME.md), so it should
reach and HOLD the optimum without the collapse. We report deterministic (greedy)
returns vs the per-env random/optimum baselines, over several seeds.

Usage:
    python run_rec_env_b.py            # full: 5 seeds, 30 epochs
    python run_rec_env_b.py smoke      # 1 seed, 6 epochs (pipeline check)
"""
import os
import sys
import json

# Pin BLAS/OpenMP threads before numpy import (see run_suite.py for the why):
# this module imports numpy ahead of run_suite, so it must set the pin itself.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SuiteConfig                       # noqa: E402
from envs import make_env, perfect_heuristic         # noqa: E402
from run_suite import compute_baselines, _make_args, _extract_metrics, _set_seed  # noqa: E402

ENV = "b_sequence_disjoint"


def train_rec_cell(seed, cfg, logdir_base):
    _set_seed(seed)
    env = make_env(ENV, causal_rl=True, allow_postpone=cfg.allow_postpone)
    args = _make_args(ENV, "causal_td0", seed, cfg, logdir_base)
    args["causal_scheme"] = "rec"          # <-- the scheme under test
    args["name"] = f"{ENV}__rec_b{cfg.causal_beta}__s{seed}"
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=cfg.length, args_dict=args)
    finally:
        sys.argv = saved
    history = getattr(env, "training_history", {}) or {}
    m = _extract_metrics(history, cfg)
    m.update({"env": ENV, "method": "rec", "seed": seed})
    return m


def _rec_seed_worker(payload):
    """Top-level picklable worker: train one seed with threads pinned to 1."""
    import torch
    torch.set_num_threads(1)
    seed, cfg, logdir_base = payload
    try:
        return (seed, train_rec_cell(seed, cfg, logdir_base), None)
    except Exception:
        import traceback
        return (seed, None, traceback.format_exc())


def main(mode, num_workers=None):
    smoke = (mode == "smoke")
    cfg = SuiteConfig(
        envs=[ENV], methods=["causal_td0"],
        seeds=(1 if smoke else 5),
        epochs=(6 if smoke else 30),
        episodes_per_epoch=(8 if smoke else 20),
        test_freq=(2 if smoke else 5),
        baseline_episodes=(10 if smoke else 20),
        causal_beta=0.2, causal_self_credit=0.5, causal_lam=0.0,
        allow_postpone=True,
        output_dir=__import__("pathlib").Path("rec_env_b_results"),
    )
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logdir_base = str((out / "train").resolve())
    os.makedirs(logdir_base, exist_ok=True)

    base = compute_baselines(ENV, cfg)
    rnd, opt = base["random_mean"], base["heuristic_mean"]
    print(f"\n[rec/env-b] random={rnd:.2f}  optimum(heuristic)={opt:.2f}  "
          f"beta={cfg.causal_beta} postpone={cfg.allow_postpone}\n")

    # Seed-level parallelism: seeds are independent runs (Tier 1.1). Default to
    # min(seeds, physical_cores) workers; override with the 2nd CLI arg / num_workers.
    if num_workers is None:
        from run_suite import _physical_core_count
        num_workers = min(cfg.seeds, _physical_core_count())

    def _record(m):
        gf, gb, drift = m["greedy_final"], m["greedy_best"], m["greedy_drift"]
        norm = (gf - rnd) / (opt - rnd) if (opt - rnd) else float("nan")
        print(f"[rec/env-b] seed {m['seed']}: greedy_final={gf:.2f} best={gb:.2f} "
              f"drift={drift:.2f} normalized={norm:.2f}  curve={['%.1f'%x for x in m['greedy_curve']]}")
        (out / f"seed{m['seed']}.json").write_text(json.dumps(m, indent=2))
        return (m["seed"], gf, gb, drift, norm, m["greedy_curve"])

    rows = []
    if num_workers <= 1:
        for s in range(cfg.seeds):
            print(f"[rec/env-b] === seed {s} ===")
            rows.append(_record(train_rec_cell(s, cfg, logdir_base)))
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[rec/env-b] running {cfg.seeds} seeds across {num_workers} processes")
        ctx = mp.get_context("spawn")
        payloads = [(s, cfg, logdir_base) for s in range(cfg.seeds)]
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_rec_seed_worker, p) for p in payloads]
            for fut in as_completed(futs):
                seed, m, err = fut.result()
                if err is not None or m is None:
                    print(f"[rec/env-b] !!! seed {seed} FAILED:\n{err}")
                    continue
                rows.append(_record(m))
    rows.sort(key=lambda r: r[0])

    gfs = [r[1] for r in rows]
    norms = [r[4] for r in rows]
    collapsed = sum(1 for r in rows if (r[2] or 0) == 0 or (r[1] or 0) == 0)
    print("\n================ REC on env b (beta=0.2) ================")
    print(f" seeds                 : {cfg.seeds}")
    print(f" random / optimum      : {rnd:.2f} / {opt:.2f}")
    print(f" greedy_final per seed : {['%.2f'%x for x in gfs]}")
    print(f" mean greedy_final     : {np.mean(gfs):.2f}  (normalized {np.mean(norms):.2f})")
    print(f" collapsed seeds       : {collapsed}/{cfg.seeds}  (greedy hit 0)")
    print("========================================================")
    (out / "summary.json").write_text(json.dumps({
        "env": ENV, "scheme": "rec", "causal_beta": cfg.causal_beta,
        "random": rnd, "optimum": opt,
        "greedy_final": gfs, "normalized_mean": float(np.mean(norms)),
        "collapsed": collapsed, "seeds": cfg.seeds,
    }, indent=2))


if __name__ == "__main__":
    # Usage: python run_rec_env_b.py [mode] [num_workers]
    _mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    _workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(_mode, num_workers=_workers)