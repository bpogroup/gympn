"""Run the causal-RL stability suite.

Resumable: every (env, method, seed) cell writes its own JSON under
<output_dir>/cells/ and is SKIPPED if that file already exists. Per-env Random
and perfect-Heuristic baselines are computed once and cached under
<output_dir>/baselines/. Safe to Ctrl-C and re-run; it picks up where it left off.

Usage:
    python run_suite.py            # full config (8 envs x 2 methods x 10 seeds)
    python run_suite.py smoke      # fast end-to-end pipeline check
"""
import os
import sys
import json
import time
import random
import traceback
from pathlib import Path

# Pin BLAS/OpenMP to one thread *before* numpy/torch import. The suite runs one
# process per (env, method, seed) cell, so each worker must be single-threaded;
# otherwise N worker processes x (BLAS thread pool of up to cpu_count) massively
# oversubscribe the cores and the machine thrashes to a near-halt (heavy
# parallel/loop envs were the casualties in the ppo-vs-rec post-mortem). Pool
# children re-import this module, so they inherit the pin. setdefault => an
# explicit OMP_NUM_THREADS=... in the environment still wins.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch


def _physical_core_count() -> int:
    """Physical (not logical) CPU cores — the right cap for CPU-bound cells.

    os.cpu_count() returns logical cores; on a hyperthreaded machine that double-
    counts, so one heavyweight training process per logical core 2x-oversubscribes
    the real cores. Prefer psutil's physical count; fall back to logical//2, then 1.
    """
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def _default_workers() -> int:
    """Half the physical cores. This GNN (HGT) workload is memory-bandwidth-bound:
    profiling showed ~3x throughput loss at one heavyweight process per physical
    core. Running ~physical/2 workers — each handed the remaining cores as torch
    intra-op threads (see _threads_per_worker) — matches total throughput with far
    less contention, and keeps the laptop usable. >=1."""
    return max(1, _physical_core_count() // 2)


def _threads_per_worker(num_workers: int) -> int:
    """Fair share of physical cores for each worker's torch intra-op pool, so
    W workers x T threads ~= physical cores. The autograd backward (the dominant
    cost, ~49%) threads well; giving each of W=physical/2 workers T=2 threads uses
    all cores with half as many memory-churning processes. >=1."""
    phys = _physical_core_count()
    return max(1, phys // max(1, num_workers))

# Allow running as a plain script from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SuiteConfig, smoke_config, smoke8_config, seeds5_config,  # noqa: E402
                    paper_config, stoch_config)
from envs import make_env, perfect_heuristic, HEURISTICS           # noqa: E402


def _env_length(cfg: SuiteConfig, env_name: str) -> int:
    """Per-env horizon override (stochastic tier), else the global default."""
    return int(getattr(cfg, "env_length", {}).get(env_name, cfg.length))


def _set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_baselines(env_name: str, cfg: SuiteConfig) -> dict:
    """Random (lower bound) and perfect-heuristic (optimum) on the canonical
    no-postpone env, averaged over cfg.baseline_episodes deterministic sims."""
    from gympn.solvers import RandomSolver, HeuristicSolver

    length = _env_length(cfg, env_name)
    heuristic = HEURISTICS.get(env_name, perfect_heuristic)

    def _avg(make_solver, seed0):
        rewards = []
        for i in range(cfg.baseline_episodes):
            _set_seed(seed0 + i)
            env = make_env(env_name, causal_rl=False, allow_postpone=False)
            rewards.append(float(env.testing_run(solver=make_solver(), length=length)))
        return float(np.mean(rewards)), float(np.std(rewards))

    rnd_mean, rnd_std = _avg(lambda: RandomSolver(), seed0=10_000)
    heu_mean, heu_std = _avg(lambda: HeuristicSolver(heuristic), seed0=20_000)
    return {
        "env": env_name,
        "random_mean": rnd_mean, "random_std": rnd_std,
        "heuristic_mean": heu_mean, "heuristic_std": heu_std,
    }


def _net_kwargs(cfg: SuiteConfig) -> dict:
    """Optional HGT size overrides, emitted only when set on the config so that
    default runs are unchanged (train.py falls back to hidden=256/L3/residual)."""
    kw = {}
    if getattr(cfg, "net_hidden_size", None) is not None:
        kw["hidden_size"] = cfg.net_hidden_size
    if getattr(cfg, "net_num_layers", None) is not None:
        kw["num_layers"] = cfg.net_num_layers
    if getattr(cfg, "net_residual", None) is not None:
        kw["residual"] = cfg.net_residual
    return kw


def _make_args(env_name: str, method: str, seed: int, cfg: SuiteConfig, logdir_base: str) -> dict:
    causal = method in ("lrq", "lrq2", "lrq3", "lqi", "lcv", "lva", "mc_q")
    # lcv0: LCV's exact c_hat=0 limiting case -- plain SMDP-GAE PPO (the
    # STANDARD, non-causal_rl path with the per-sojourn discount switched on).
    # No lineage machinery at all, so it stays OUT of the causal_rl set above.
    smdp_discount = (method == "lcv0")
    net_kw = _net_kwargs(cfg)
    extra = {}
    if net_kw:
        # Same size for actor and critic keeps the comparison fair and cheap.
        extra["policy_kwargs"] = dict(net_kw)
        extra["value_kwargs"] = dict(net_kw)
    return {
        **extra,
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
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_rl": causal,
        "causal_scheme": method if method in ("lrq2", "lrq3", "lqi", "lcv", "lva", "mc_q") else "lrq",
        "causal_beta": cfg.causal_beta,
        "causal_mu": getattr(cfg, "causal_mu", 0.0),
        "causal_aux_coef": getattr(cfg, "causal_aux_coef", 0.5),
        "smdp_discount": smdp_discount,
        "rudder_enabled": (method == "rudder"),
        "rudder_hidden_dim": getattr(cfg, "rudder_hidden_dim", 64),
        "rudder_training_freq": getattr(cfg, "rudder_training_freq", 1),
        "rudder_redistribution_method": getattr(cfg, "rudder_redistribution_method", "contribution"),
        "algorithm": "ppo-clip",
        "verbose": 1,
        "use_gpu": False,
        "agent_seed": int(seed),
        "use_wandb": False,
        "open_tensorboard": False,
        "test_in_train": True,
        "test_freq": cfg.test_freq,
        "test_episodes": getattr(cfg, "test_episodes", 10),
        "save_freq": 1_000_000,
        "name": cfg.cell_id(env_name, method, seed),
        "datetag": False,
        "logdir": logdir_base,
    }


def _extract_metrics(history: dict, cfg: SuiteConfig) -> dict:
    """Pull the stability/performance signals out of training_history."""
    def arr(key):
        return list(map(float, history[key])) if key in history else []

    sampled = arr("mean_returns")
    n_test = cfg.epochs // cfg.test_freq
    greedy = arr("test_mean_returns")[:n_test]          # drop the trailing unused slot
    ent = arr("policy_ent")
    test_epochs = [cfg.test_freq * (i + 1) for i in range(len(greedy))]

    return {
        # LCV mechanism telemetry (zeros / absent for other schemes).
        "cv_coef_curve": arr("cv_coef"),
        "cv_var_reduction_curve": arr("cv_var_reduction"),
        "sampled_curve": sampled,
        "sampled_best": max(sampled) if sampled else None,
        "sampled_final": sampled[-1] if sampled else None,
        "greedy_curve": greedy,
        "greedy_epochs": test_epochs,
        "greedy_best": max(greedy) if greedy else None,
        "greedy_final": greedy[-1] if greedy else None,
        # H1 stability signal: how far the greedy policy fell from its peak.
        "greedy_drift": (max(greedy) - greedy[-1]) if greedy else None,
        "entropy_curve": ent,
        "entropy_final": ent[-1] if ent else None,
    }


def _train_cell_worker(payload):
    """Top-level (picklable) worker for parallel cell training.

    Each cell is an independent (env, method, seed) run that shares nothing, so
    cells parallelize cleanly across processes. We pin intra-op threads to 1 so N
    worker processes do not oversubscribe the cores (which would erase the gain).
    """
    import torch
    env_name, method, seed, cfg, logdir_base, threads = payload
    # Each worker gets a fair share of cores for torch intra-op (backward)
    # parallelism; BLAS stays pinned to 1 via the env (set at module import).
    torch.set_num_threads(max(1, int(threads)))
    try:
        metrics = train_cell(env_name, method, seed, cfg, logdir_base)
        return (env_name, method, seed, metrics, None)
    except Exception:
        import traceback
        return (env_name, method, seed, None, traceback.format_exc())


def train_cell(env_name: str, method: str, seed: int, cfg: SuiteConfig, logdir_base: str) -> dict:
    _set_seed(seed)
    causal = method in ("lrq", "lrq2", "lrq3", "lqi", "lcv", "lva", "mc_q")
    # LRQ requires token-flow postpone (postpone must be a lineage member).
    # lrq2/mc_q don't need it, but get the IDENTICAL env config so the
    # variants differ in the credit computation only. Plain PPO/RUDDER ignore
    # the flag.
    env = make_env(env_name, causal_rl=causal,
                   allow_postpone=cfg.allow_postpone,
                   causal_postpone_tokenflow=causal)
    args = _make_args(env_name, method, seed, cfg, logdir_base)

    # training_run internally calls parse_args(); neutralize our own argv so it
    # only sees defaults (then args_dict overrides).
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=_env_length(cfg, env_name), args_dict=args)
    finally:
        sys.argv = saved_argv

    history = getattr(env, "training_history", {}) or {}
    metrics = _extract_metrics(history, cfg)
    metrics.update({"env": env_name, "method": method, "seed": seed})
    return metrics


def run_suite(cfg: SuiteConfig, num_workers: int = 1):
    out = Path(cfg.output_dir)
    cells_dir = out / "cells"
    base_dir = out / "baselines"
    logdir_base = str((out / "train").resolve())
    for d in (cells_dir, base_dir, Path(logdir_base)):
        d.mkdir(parents=True, exist_ok=True)

    total = len(cfg.envs) * len(cfg.methods) * cfg.seeds
    done = skipped = failed = 0
    t0 = time.time()
    print(f"[suite] {len(cfg.envs)} envs x {len(cfg.methods)} methods x {cfg.seeds} seeds "
          f"= {total} cells -> {out}  (num_workers={num_workers})")

    # --- baselines (cached), computed once up front so workers don't race them ---
    baselines_by_env = {}
    for env_name in cfg.envs:
        bpath = base_dir / f"{env_name}.json"
        if bpath.exists():
            baselines = json.loads(bpath.read_text())
        else:
            print(f"[suite] baselines for {env_name} ...")
            baselines = compute_baselines(env_name, cfg)
            bpath.write_text(json.dumps(baselines, indent=2))
        baselines_by_env[env_name] = baselines
        print(f"[suite] {env_name}: random={baselines['random_mean']:.2f} "
              f"heuristic(optimum)={baselines['heuristic_mean']:.2f}")

    # --- enumerate pending cells (skip already-finished ones: resumable) ---
    pending = []
    for env_name in cfg.envs:
        for method in cfg.methods:
            for seed in range(cfg.seeds):
                cid = cfg.cell_id(env_name, method, seed)
                if (cells_dir / f"{cid}.json").exists():
                    skipped += 1
                    continue
                pending.append((env_name, method, seed))

    def _write_result(env_name, method, seed, metrics, err):
        nonlocal done, failed
        cid = cfg.cell_id(env_name, method, seed)
        if err is not None or metrics is None:
            failed += 1
            print(f"[suite] !!! {cid} FAILED:\n{err}")
            return
        metrics["baselines"] = baselines_by_env[env_name]
        (cells_dir / f"{cid}.json").write_text(json.dumps(metrics, indent=2))
        done += 1
        print(f"[suite] <<< {cid}  greedy_final={metrics.get('greedy_final')}  "
              f"({done + failed}/{len(pending)} cells, {(time.time()-t0)/60:.1f} min)")

    if num_workers <= 1:
        # Sequential (unchanged behaviour).
        for (env_name, method, seed) in pending:
            cid = cfg.cell_id(env_name, method, seed)
            print(f"[suite] >>> {cid}")
            try:
                metrics = train_cell(env_name, method, seed, cfg, logdir_base)
                _write_result(env_name, method, seed, metrics, None)
            except Exception:
                _write_result(env_name, method, seed, None, traceback.format_exc())
    else:
        # Parallel: one process per cell, pool sized to num_workers. spawn +
        # threads=1 per worker (set in _train_cell_worker) avoids oversubscription.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(num_workers)
        print(f"[suite] {num_workers} workers x {tpw} torch-thread(s) each "
              f"(physical cores={_physical_core_count()})")
        payloads = [(e, m, s, cfg, logdir_base, tpw) for (e, m, s) in pending]
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_train_cell_worker, p) for p in payloads]
            for fut in as_completed(futs):
                env_name, method, seed, metrics, err = fut.result()
                _write_result(env_name, method, seed, metrics, err)

    print(f"[suite] complete: {done} trained, {skipped} skipped, {failed} failed, "
          f"{(time.time()-t0)/60:.1f} min total")


if __name__ == "__main__":
    # Usage: python run_suite.py [mode] [num_workers]
    #   mode in {smoke, smoke8, full}; num_workers also via env SUITE_WORKERS.
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if len(sys.argv) > 2:
        workers = int(sys.argv[2])
    else:
        workers = int(os.environ.get("SUITE_WORKERS", _default_workers()))
    cfg = {"smoke": smoke_config, "smoke8": smoke8_config, "seeds5": seeds5_config,
           "paper": paper_config, "stoch": stoch_config,
           "full": SuiteConfig}.get(mode, SuiteConfig)()
    print(f"[suite] mode = {mode}")
    run_suite(cfg, num_workers=workers)