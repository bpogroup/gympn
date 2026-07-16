"""Head-to-head: shapley_dag (Route B) vs flow_dag (Route A) causal redistribution.

Both run with postpone ON.
  - flow_dag         : sink-less postpone + SMDP discount (causal_beta=0.2),
                       self_credit=0.5 (validated suite default = "causal_td0")
  - shapley_dag      : token-flow postpone + Shapley credit, causal_beta=0.0
                       (the original "intended" config — turned out to be UNSTABLE
                       on the joint env because beta=0; kept for the ablation)
  - shapley_dag_b02  : token-flow postpone + Shapley credit + causal_beta=0.2
                       (the FIX — Shapley credit and SMDP continuation discount are
                       complementary; this is the recommended shapley_dag config)

See SHAPLEY_VS_FLOW_DAG_RESULTS.md for the result and the ablation.

Reports greedy (argmax) eval curves, final, peak, drift (peak-final, the H1
stability signal), all normalized per env as (policy-random)/(heuristic-random).

Usage:
    python compare_shapley_vs_flow.py            # default: a,b  x 3 seeds x 20 epochs
    python compare_shapley_vs_flow.py smoke      # 1 seed, 3 epochs (pipeline check)
"""
import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import make_env, perfect_heuristic  # noqa: E402


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---- methods: name -> (causal_scheme, causal_beta, tokenflow, self_credit) ----
METHODS = {
    "flow_dag":    dict(scheme="flow_dag",    beta=0.2, tokenflow=False, self_credit=0.5),
    "shapley_dag": dict(scheme="shapley_dag", beta=0.0, tokenflow=True,  self_credit=0.5),
    # Ablation: Shapley credit PLUS the SMDP continuation discount, to isolate
    # foreclosure (Gap D) from the missing continuation discount as the cause of
    # the env-a (joint) instability.
    "shapley_dag_b02": dict(scheme="shapley_dag", beta=0.2, tokenflow=True, self_credit=0.5),
}


def compute_baselines(env_name, length, n):
    from gympn.solvers import RandomSolver, HeuristicSolver

    def _avg(make_solver, seed0):
        rs = []
        for i in range(n):
            _set_seed(seed0 + i)
            env = make_env(env_name, causal_rl=False, allow_postpone=False)
            rs.append(float(env.testing_run(solver=make_solver(), length=length)))
        return float(np.mean(rs))

    return {
        "random": _avg(lambda: RandomSolver(), 10_000),
        "heuristic": _avg(lambda: HeuristicSolver(perfect_heuristic), 20_000),
    }


def train_cell(env_name, method, seed, cfg):
    m = METHODS[method]
    _set_seed(seed)
    env = make_env(env_name, causal_rl=True, allow_postpone=True)
    # Enable token-flow postpone on the PN (and its causal trace) when requested.
    env.causal_postpone_tokenflow = m["tokenflow"]
    if getattr(env, "causal_trace", None) is not None:
        env.causal_trace.postpone_tokenflow = m["tokenflow"]

    args = {
        "episodes": cfg["episodes"], "epochs": cfg["epochs"], "batch_size": 32,
        "max_episode_length": None,
        "policy_lr": 3e-4, "policy_updates": 5, "value_lr": 3e-4, "value_updates": 10,
        "gam": 0.99, "lam": 0.95, "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01,
        "causal_rl": True, "causal_scheme": m["scheme"], "causal_gamma": 0.9,
        "causal_lam": 0.0, "causal_self_credit": m["self_credit"], "causal_beta": m["beta"],
        "algorithm": "ppo-clip", "verbose": 1, "use_gpu": False,
        "agent_seed": int(seed), "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": cfg["test_freq"], "save_freq": 10**9,
        "name": f"{env_name}__{method}__s{seed}", "datetag": False,
        "logdir": cfg["logdir"],
    }
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        env.training_run(length=cfg["length"], args_dict=args)
    finally:
        sys.argv = saved

    h = getattr(env, "training_history", {}) or {}
    greedy = list(map(float, h.get("test_mean_returns", [])))[: cfg["epochs"] // cfg["test_freq"]]
    return {
        "greedy_curve": greedy,
        "greedy_final": greedy[-1] if greedy else None,
        "greedy_best": max(greedy) if greedy else None,
        "greedy_drift": (max(greedy) - greedy[-1]) if greedy else None,
    }


ALL_ENVS = [
    "a_sequence_joint", "b_sequence_disjoint", "c_parallel_joint",
    "d_parallel_disjoint", "e_loop_joint", "f_loop_disjoint",
    "g_exclusive_choice_joint", "h_exclusive_choice_disjoint",
]


def main(mode):
    if mode == "smoke":
        cfg = dict(envs=["b_sequence_disjoint"], seeds=1, epochs=3, episodes=4,
                   test_freq=1, length=10, baseline_n=5,
                   methods=list(METHODS), out="compare_shapley_results")
    elif mode == "ablation":
        # Isolate the foreclosure cause on the joint env: Shapley + SMDP discount.
        cfg = dict(envs=["a_sequence_joint"], seeds=3, epochs=15,
                   episodes=12, test_freq=3, length=10, baseline_n=20,
                   methods=["shapley_dag_b02"], out="compare_shapley_ablation")
    elif mode == "suite8":
        # Full 8-env head-to-head with MATCHED causal_beta=0.2:
        #   flow_dag (Route A) vs shapley_dag_b02 (Route B, the corrected config).
        cfg = dict(envs=list(ALL_ENVS), seeds=3, epochs=15,
                   episodes=12, test_freq=3, length=10, baseline_n=20,
                   methods=["flow_dag", "shapley_dag_b02"], out="compare_shapley_suite8")
    else:
        cfg = dict(envs=["a_sequence_joint", "b_sequence_disjoint"], seeds=3, epochs=15,
                   episodes=12, test_freq=3, length=10, baseline_n=20,
                   methods=["flow_dag", "shapley_dag"], out="compare_shapley_results")

    out = Path(cfg["out"])
    cells_dir = out / "cells"
    base_dir = out / "baselines"
    for d in (out, cells_dir, base_dir):
        d.mkdir(parents=True, exist_ok=True)
    cfg["logdir"] = str((out / "train").resolve())

    t0 = time.time()
    total = len(cfg["envs"]) * len(cfg["methods"]) * cfg["seeds"]
    done = skipped = 0
    for env_name in cfg["envs"]:
        # --- baselines (cached per env) ---
        bpath = base_dir / f"{env_name}.json"
        if bpath.exists():
            base = json.loads(bpath.read_text())
        else:
            base = compute_baselines(env_name, cfg["length"], cfg["baseline_n"])
            bpath.write_text(json.dumps(base, indent=2))
        print(f"\n=== {env_name}: random={base['random']:.2f} optimum={base['heuristic']:.2f} ===")

        # --- cells (resumable: one JSON each, skipped if present) ---
        for method in cfg["methods"]:
            for seed in range(cfg["seeds"]):
                cpath = cells_dir / f"{env_name}__{method}__s{seed}.json"
                if cpath.exists():
                    skipped += 1
                    continue
                print(f"  >>> {env_name} | {method} | seed {seed}  "
                      f"({done + skipped + 1}/{total}, {(time.time()-t0)/60:.1f} min)")
                m = train_cell(env_name, method, seed, cfg)
                m.update({"env": env_name, "method": method, "seed": seed})
                cpath.write_text(json.dumps(m, indent=2))
                done += 1

    print(f"\n[done] {done} trained, {skipped} skipped, {(time.time()-t0)/60:.1f} min")
    _summary(cfg, out, base_dir, cells_dir)


def _summary(cfg, out, base_dir, cells_dir):
    """Aggregate per-cell JSONs into a table + a results.json blob."""
    def norm(x, b):
        denom = (b["heuristic"] - b["random"])
        return (x - b["random"]) / denom if denom else float("nan")

    results = {}
    print("\n" + "=" * 84)
    print(f"{'env':<26}{'method':<16}{'final':>8}{'peak':>8}{'drift':>8}{'norm_final':>12}")
    print("-" * 84)
    for env_name in cfg["envs"]:
        bpath = base_dir / f"{env_name}.json"
        if not bpath.exists():
            continue
        b = json.loads(bpath.read_text())
        results[env_name] = {"baselines": b, "cells": {}}
        for method in cfg["methods"]:
            cells = []
            for seed in range(cfg["seeds"]):
                cpath = cells_dir / f"{env_name}__{method}__s{seed}.json"
                if cpath.exists():
                    cells.append(json.loads(cpath.read_text()))
            if not cells:
                continue
            results[env_name]["cells"][method] = cells
            fin = float(np.mean([c["greedy_final"] for c in cells]))
            peak = float(np.mean([c["greedy_best"] for c in cells]))
            drift = float(np.mean([c["greedy_drift"] for c in cells]))
            print(f"{env_name:<26}{method:<16}{fin:>8.2f}{peak:>8.2f}{drift:>8.2f}{norm(fin,b):>12.2f}")
    print("=" * 84)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")