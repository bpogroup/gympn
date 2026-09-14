"""X15: DCL (model-based planner) on s1 — first run of the planner, and the
first test of the lineage used STRUCTURALLY rather than as a scalar signal.

Motivation (CAUSAL_LINEAGE_RETHINK.md §7, and the model-free-bottleneck
discussion): every scalar-advantage method funnels the lineage graph through
a one-number interface and its value evaporates. DCL replaces the interface:
it plans by CRN-averaged rollouts over every enabled action and distils the
result into the actor. The lineage then does jobs a scalar cannot express —
sharing one rollout across independent candidates, pruning candidates that
cannot earn in-horizon, coupling truncation.

Three arms (small budget; this is an exploratory first run, not a tuned one):
  dcl_plain   = planner, no lineage machinery (compute_target_pi path)
  dcl_lin     = structural lineage: sharing + pruning + truncation, RAW tally
                (unbiased; sharing only fires with a restricted tally, so its
                benefit here is pruning + truncation cost savings)
  dcl_lin_t   = + lineage-restricted tally (enables sharing; biased on
                foreclosure envs like s1 -- included precisely to see the
                bias the planner is supposed to make irrelevant)

Registered expectations (exploratory, so predictions are soft, 2026-07-21):
  E1: dcl_plain should be COMPETITIVE with cfpk's 0.849 on s1 finals -- the
      planner evaluates the opportunity cost directly (it is in the rollouts),
      which is exactly what the scalar methods could not do. If the planner
      is NOT competitive, the model-based pivot does not clear its own bar and
      that is the headline finding.
  E2: dcl_lin ~ dcl_plain on finals (raw tally is unbiased) but fewer
      rollouts / faster (pruning + truncation) -- the free structural win.
  E3: dcl_lin_t may LOSE on s1 finals (restricted tally reintroduces the
      foreclosure bias X13 measured) even though it harvests more samples --
      the planned negative control that confirms the mechanism.

Cost note: DCL is rollouts x actions x horizon per step; even at these small
budgets it is far slower than PPO. Small epoch/episode counts on purpose.

Run: python run_dcl_s1.py [workers]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import (_set_seed, _env_length, _extract_metrics,  # noqa: E402
                       compute_baselines, _threads_per_worker)
from stoch_envs import STOCH_BUILDERS  # noqa: E402

ENV = "s1_stoch_sequence"

ARMS = {
    "dcl_plain":  dict(dcl_lineage=False),
    "dcl_lin":    dict(dcl_lineage=True,  dcl_lineage_tally=False),
    "dcl_lin_t":  dict(dcl_lineage=True,  dcl_lineage_tally=True),
}


def _make_dcl_args(arm: str, seed: int, cfg, logdir_base: str) -> dict:
    lin = ARMS[arm].get("dcl_lineage", False)
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
        "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        # DCL needs env tracing whenever any lineage machinery is on.
        "causal_rl": False,
        "causal_beta": cfg.causal_beta,
        "algorithm": "dcl",
        "dcl_horizon": getattr(cfg, "dcl_horizon", 8),
        "dcl_rollouts": getattr(cfg, "dcl_rollouts", 12),
        "dcl_temp": getattr(cfg, "dcl_temp", 0.5),
        "dcl_lineage": lin,
        "dcl_lineage_tally": ARMS[arm].get("dcl_lineage_tally", False),
        "verbose": 1,
        "use_gpu": False,
        "agent_seed": int(seed),
        "use_wandb": False,
        "open_tensorboard": False,
        "test_in_train": True,
        "test_freq": cfg.test_freq,
        "test_episodes": getattr(cfg, "test_episodes", 10),
        "save_freq": 1_000_000,
        "name": f"{ENV}__{arm}__s{seed}",
        "datetag": False,
        "logdir": logdir_base,
    }
    return args


def train_dcl_cell(arm: str, seed: int, cfg, logdir_base: str, baselines: dict) -> dict:
    from gympn.environment import AEPN_Env  # noqa
    _set_seed(seed)
    lin = ARMS[arm].get("dcl_lineage", False)
    # env trace needed only when the planner uses lineage
    builder = STOCH_BUILDERS[ENV]
    env = builder(causal_rl=lin, allow_postpone=True,
                  causal_postpone_tokenflow=lin)
    args = _make_dcl_args(arm, seed, cfg, logdir_base)
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=_env_length(cfg, ENV), args_dict=args)
    finally:
        sys.argv = saved_argv
    history = getattr(env, "training_history", {}) or {}
    metrics = _extract_metrics(history, cfg)
    metrics.update({"env": ENV, "method": arm, "seed": seed,
                    "baselines": baselines, "minutes": (time.time() - t0) / 60.0})
    return metrics


def _worker(payload):
    import torch
    arm, seed, cfg, logdir_base, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (arm, seed, train_dcl_cell(arm, seed, cfg, logdir_base, baselines), None)
    except Exception:
        import traceback
        return (arm, seed, None, traceback.format_exc())


def main(workers=3):
    cfg = stoch_config()
    # small, exploratory budget — DCL is expensive (~1.3s/planner call,
    # ~29 steps/episode). This first-look budget targets ~8 min/cell.
    cfg.epochs = 8
    cfg.episodes_per_epoch = 5
    cfg.test_freq = 2
    cfg.seeds = 3
    cfg.dcl_horizon = 5
    cfg.dcl_rollouts = 6
    cfg.dcl_temp = 0.5

    out = Path("suite_results_dcl")
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    print(f"[dcl] baselines for {ENV} ...", flush=True)
    bpath = out / "baselines.json"
    if bpath.exists():
        baselines = json.loads(bpath.read_text())
    else:
        baselines = compute_baselines(ENV, cfg)
        bpath.write_text(json.dumps(baselines))
    print(f"[dcl] {ENV}: random={baselines['random_mean']:.2f} "
          f"heuristic={baselines['heuristic_mean']:.2f}", flush=True)

    pending = []
    for arm in ARMS:
        for seed in range(cfg.seeds):
            cell = out / "cells" / f"{ENV}__{arm}__s{seed}.json"
            if not cell.exists():
                pending.append((arm, seed))
    print(f"[dcl] {len(pending)} cells to train, {workers} workers "
          f"(horizon={cfg.dcl_horizon} rollouts={cfg.dcl_rollouts})", flush=True)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    tpw = _threads_per_worker(workers)
    payloads = [(arm, seed, cfg, logdir, baselines, tpw) for (arm, seed) in pending]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for fut in as_completed(futs):
            arm, seed, metrics, err = fut.result()
            if err is not None:
                print(f"[dcl] !!! {arm}_s{seed} FAILED:\n{err}", flush=True)
                continue
            (out / "cells" / f"{ENV}__{arm}__s{seed}.json").write_text(
                json.dumps(metrics))
            gf = metrics.get("greedy_final")
            print(f"[dcl] <<< {arm}_s{seed} greedy_final={gf} "
                  f"({metrics['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)",
                  flush=True)

    # summary
    def norm(v):
        r = baselines["random_mean"]
        return (v - r) / (baselines["heuristic_mean"] - r)
    print(f"\n[dcl] === s1 finals (anchor={baselines['heuristic_mean']:.2f}, "
          f"cfpk ref 0.849) ===", flush=True)
    for arm in ARMS:
        finals = []
        for seed in range(cfg.seeds):
            cell = out / "cells" / f"{ENV}__{arm}__s{seed}.json"
            if cell.exists():
                m = json.loads(cell.read_text())
                if m.get("greedy_final") is not None:
                    finals.append(m["greedy_final"])
        if finals:
            nf = [norm(f) for f in finals]
            print(f"[dcl] {arm:<10} raw={np.mean(finals):.2f} "
                  f"norm={np.mean(nf):.3f}±{np.std(nf):.2f} n={len(finals)}",
                  flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)