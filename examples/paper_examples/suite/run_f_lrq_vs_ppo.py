"""Env f (loop, disjoint): LRQ vs plain PPO, multi-seed.

Question: does LRQ converge faster and hold the optimum more stably than plain
PPO-clip on the rework-loop topology? Judged per the suite protocol on the
deterministic (greedy) eval curves, aggregated over seeds.

Uses the suite's shared hyperparameters (SuiteConfig) so the numbers are
comparable to earlier suite runs. The one LRQ-specific requirement is
token-flow postpone (causal_postpone_tokenflow=True) on the causal env, which
the suite's builders don't thread — patched in here via envs.GymProblem.

Resumable: each (method, seed) cell writes cells/<method>_s<seed>.json and is
skipped if present.

Run: python run_f_lrq_vs_ppo.py [n_seeds] [n_workers]
"""
import os
import sys
import json
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import SuiteConfig  # noqa: E402
import envs  # noqa: E402
from run_suite import (compute_baselines, _set_seed,  # noqa: E402
                       _default_workers, _threads_per_worker)

ENV = "f_loop_disjoint"
METHODS = ("ppo", "lrq")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f_lrq_vs_ppo")
CELLS = os.path.join(OUT, "cells")


def make_env_f(causal_rl: bool):
    """Env f, with token-flow postpone enabled iff the run is causal (LRQ
    requires postpone to be a lineage member; plain PPO ignores the flag)."""
    orig = envs.GymProblem
    if causal_rl:
        def patched(*a, **kw):
            kw.setdefault("causal_postpone_tokenflow", True)
            return orig(*a, **kw)
        envs.GymProblem = patched
    try:
        return envs.make_env(ENV, causal_rl=causal_rl, allow_postpone=True)
    finally:
        envs.GymProblem = orig


def run_one(method: str, seed: int, cfg: SuiteConfig) -> dict:
    causal = (method == "lrq")
    _set_seed(seed)
    env = make_env_f(causal_rl=causal)
    args = {
        "episodes": cfg.episodes_per_epoch,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "max_episode_length": None,
        "policy_lr": cfg.policy_lr,
        "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr,
        "value_updates": cfg.value_updates,
        "gam": cfg.gam,
        "lam": cfg.lam,
        "eps": cfg.ppo_eps,
        "vf_coeff": 0.5,
        "ent_bonus": cfg.ent_bonus,
        "causal_rl": causal,
        "causal_scheme": "lrq",
        "causal_beta": cfg.causal_beta,
        "causal_mu": 0.0,
        "algorithm": "ppo-clip",
        "verbose": 1,
        "use_gpu": False,
        "agent_seed": int(seed),
        "use_wandb": False,
        "open_tensorboard": False,
        "test_in_train": True,
        "test_freq": cfg.test_freq,
        "save_freq": 1_000_000,
        "name": f"{ENV}__{method}__s{seed}",
        "datetag": False,
        "logdir": os.path.join(OUT, "train"),
    }
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=cfg.length, args_dict=args)
    finally:
        sys.argv = saved_argv
    history = getattr(env, "training_history", {}) or {}
    return {
        "method": method,
        "seed": seed,
        "sampled": [float(x) for x in history.get("mean_returns", [])],
        "greedy": [float(x) for x in history.get("test_mean_returns", [])],
        "entropy": [float(x) for x in history.get("policy_ent", [])],
        "minutes": (time.time() - t0) / 60.0,
    }


def _cell_worker(payload):
    """Top-level (picklable) worker: one (method, seed) training cell."""
    import torch
    method, seed, cfg, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (method, seed, run_one(method, seed, cfg), None)
    except Exception:
        import traceback
        return (method, seed, None, traceback.format_exc())


def _seed_cells_from_single_run():
    """Salvage the earlier single-seed combined JSON into per-cell files."""
    combined = os.path.join(OUT, f"{ENV}_s0.json")
    if not os.path.exists(combined):
        return
    payload = json.loads(open(combined).read())
    for method, blob in payload.get("results", {}).items():
        cell = os.path.join(CELLS, f"{method}_s0.json")
        if not os.path.exists(cell):
            raw = dict(blob["raw"])
            raw.setdefault("seed", 0)
            with open(cell, "w") as f:
                json.dump(raw, f, indent=2)
            print(f"[f-test] salvaged {cell} from the single-seed run", flush=True)


def aggregate(cells: dict, base: dict, cfg: SuiteConfig) -> dict:
    """Cross-seed aggregation on the normalized scale (0=random, 1=optimum)."""
    rnd, heu = base["random_mean"], base["heuristic_mean"]

    def norm(x):
        return (x - rnd) / (heu - rnd)

    n_test = cfg.epochs // cfg.test_freq
    greedy_epochs = [cfg.test_freq * (i + 1) for i in range(n_test)]
    out = {}
    for method in METHODS:
        runs = [cells[(method, s)] for s in sorted(s for m, s in cells if m == method)]
        sampled = np.array([[norm(v) for v in r["sampled"]] for r in runs])
        greedy = np.array([[norm(v) for v in r["greedy"][:n_test]] for r in runs])

        def first_reach(curve, epochs, thr=0.9):
            for e, v in zip(epochs, curve):
                if v >= thr:
                    return e
            return None

        t90 = [first_reach(g, greedy_epochs) for g in greedy]
        drift = [float(np.max(g) - g[-1]) for g in greedy]
        out[method] = {
            "n_seeds": len(runs),
            "sampled_mean_curve": [round(float(v), 3) for v in sampled.mean(axis=0)],
            "sampled_std_curve": [round(float(v), 3) for v in sampled.std(axis=0)],
            "greedy_epochs": greedy_epochs,
            "greedy_mean_curve": [round(float(v), 3) for v in greedy.mean(axis=0)],
            "greedy_std_curve": [round(float(v), 3) for v in greedy.std(axis=0)],
            "greedy_final_per_seed": [round(float(g[-1]), 3) for g in greedy],
            "greedy_best_per_seed": [round(float(np.max(g)), 3) for g in greedy],
            "greedy_final_mean": round(float(greedy[:, -1].mean()), 3),
            "greedy_best_mean": round(float(greedy.max(axis=1).mean()), 3),
            "greedy_drift_per_seed": [round(d, 3) for d in drift],
            "greedy_drift_mean": round(float(np.mean(drift)), 3),
            "epochs_to_90pct_greedy_per_seed": t90,
            "sampled_final_mean": round(float(sampled[:, -1].mean()), 3),
            "sampled_tail_std_mean": round(
                float(np.mean([np.std(s[-5:]) for s in sampled])), 3),
            "entropy_final_per_seed": [round(r["entropy"][-1], 3) for r in runs
                                       if r.get("entropy")],
        }
    return out


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else min(4, _default_workers())
    os.makedirs(CELLS, exist_ok=True)
    cfg = SuiteConfig()

    print(f"[f-test] baselines for {ENV} ...", flush=True)
    base = compute_baselines(ENV, cfg)
    print(f"[f-test] random={base['random_mean']:.2f}  "
          f"heuristic(optimum)={base['heuristic_mean']:.2f}", flush=True)

    _seed_cells_from_single_run()

    pending, cells = [], {}
    for method in METHODS:
        for seed in range(n_seeds):
            cell_path = os.path.join(CELLS, f"{method}_s{seed}.json")
            if os.path.exists(cell_path):
                cells[(method, seed)] = json.loads(open(cell_path).read())
            else:
                pending.append((method, seed))
    print(f"[f-test] {len(cells)} cells cached, {len(pending)} to train, "
          f"{n_workers} workers", flush=True)

    t0 = time.time()
    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(n_workers)
        payloads = [(m, s, cfg, tpw) for (m, s) in pending]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_cell_worker, p) for p in payloads]
            for fut in as_completed(futs):
                method, seed, res, err = fut.result()
                if err is not None:
                    print(f"[f-test] !!! {method}_s{seed} FAILED:\n{err}", flush=True)
                    continue
                cells[(method, seed)] = res
                with open(os.path.join(CELLS, f"{method}_s{seed}.json"), "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[f-test] <<< {method}_s{seed} done "
                      f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} min total)",
                      flush=True)

    have = {m: sorted(s for mm, s in cells if mm == m) for m in METHODS}
    print(f"[f-test] aggregating: {have}", flush=True)
    agg = aggregate(cells, base, cfg)
    payload = {"env": ENV, "n_seeds": n_seeds, "baselines": base, "aggregate": agg}
    out_json = os.path.join(OUT, f"{ENV}_agg_{n_seeds}seeds.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(agg, indent=2), flush=True)
    print(f"\n[f-test] wrote {out_json}", flush=True)
