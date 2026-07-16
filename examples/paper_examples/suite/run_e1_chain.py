"""E1 experiment: chain-vs-shortcut — LRQ vs REC (legacy) vs plain PPO.

Prediction (PAPER_PLAN_LRQ.md, C1): the equal-split redistribution (rec) ranks
the shortcut (7) above completing the chain (6 = 12/2 diluted) at the shared
resource and converges toward the all-B trap (return ~70, BELOW random ~85);
LRQ sees the full 12 and converges toward the optimum (~115). PPO learns from
temporal returns and should land near the optimum, more slowly.

Baselines (deterministic, from e1_chain_env smoke): optimum 115, trap 70,
random ~85.5.

Resumable per-cell like the other suite runners.
Run: python run_e1_chain.py [n_seeds] [n_workers]
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
from run_suite import _set_seed, _threads_per_worker  # noqa: E402
from e1_chain_env import make_e1_chain, make_e1b_oneshot  # noqa: E402

METHODS = ("ppo", "lrq", "rec")

# Deterministic anchors from the e1_chain_env smoke runs.
#  - "full": repeated, non-foreclosing choice — TD's bootstrap can heal the
#    split (rec reached the optimum 5/5 here; kept as the honest contrast).
#  - "oneshot": terminal (foreclosing) choice — no bootstrap rescue; the
#    dilution bias must bite.
VARIANTS = {
    "full": dict(builder=make_e1_chain, optimum=115.0, trap=70.0,
                 random=85.5, length=10, out="e1_chain"),
    "oneshot": dict(builder=make_e1b_oneshot, optimum=12.0, trap=7.0,
                    random=9.0, length=2, out="e1b_oneshot"),
    # 3-stage chain: the diluted share drops to 12/3 = 4 vs the shortcut's 7,
    # tripling the trap's advantage gap while the true values stay 12 vs 7 —
    # separates the bias signal from optimization noise (at 2 stages the
    # 6-vs-7 gap was within seed noise: rec trapped 3/5, lrq 2/5, ppo 1/5).
    "oneshot3": dict(builder=lambda **kw: make_e1b_oneshot(stages=3, **kw),
                     optimum=12.0, trap=7.0, random=9.5, length=2,
                     out="e1c_oneshot3"),
}


def run_one(method: str, seed: int, cfg: SuiteConfig, variant: str = "full") -> dict:
    causal = method in ("lrq", "rec")
    if method == "rec":
        # Resurrected legacy scheme + its credits-as-rewards consumption path.
        import legacy_schemes
        legacy_schemes.install(gamma=0.9, self_credit=0.5, legacy_lam=0.0)
    _set_seed(seed)
    spec = VARIANTS[variant]
    env = spec["builder"](causal_rl=causal, allow_postpone=True,
                          causal_postpone_tokenflow=causal)
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
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_rl": causal,
        "causal_scheme": "lrq" if method == "lrq" else ("rec" if method == "rec" else "lrq"),
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
        "test_episodes": getattr(cfg, "test_episodes", 10),
        "save_freq": 1_000_000,
        "name": f"{spec['out']}__{method}__s{seed}",
        "datetag": False,
        "logdir": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               spec["out"], "train"),
    }
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=spec["length"], args_dict=args)
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
    import torch
    method, seed, cfg, threads, variant = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (method, seed, run_one(method, seed, cfg, variant), None)
    except Exception:
        import traceback
        return (method, seed, None, traceback.format_exc())


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    variant = sys.argv[3] if len(sys.argv) > 3 else "full"
    spec = VARIANTS[variant]
    OPTIMUM, TRAP, RANDOM = spec["optimum"], spec["trap"], spec["random"]
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), spec["out"])
    CELLS = os.path.join(OUT, "cells")
    os.makedirs(CELLS, exist_ok=True)
    cfg = SuiteConfig()
    if variant.startswith("oneshot"):
        # Episodes are ~2 sim steps; buy gradient signal instead of wall time.
        cfg.episodes_per_epoch = 60

    pending, cells = [], {}
    for method in METHODS:
        for seed in range(n_seeds):
            path = os.path.join(CELLS, f"{method}_s{seed}.json")
            if os.path.exists(path):
                cells[(method, seed)] = json.loads(open(path).read())
            else:
                pending.append((method, seed))
    print(f"[e1] {len(cells)} cached, {len(pending)} to train, {n_workers} workers",
          flush=True)

    t0 = time.time()
    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(n_workers)
        payloads = [(m, s, cfg, tpw, variant) for (m, s) in pending]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_cell_worker, p) for p in payloads]
            for fut in as_completed(futs):
                method, seed, res, err = fut.result()
                if err is not None:
                    print(f"[e1] !!! {method}_s{seed} FAILED:\n{err}", flush=True)
                    continue
                cells[(method, seed)] = res
                with open(os.path.join(CELLS, f"{method}_s{seed}.json"), "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[e1] <<< {method}_s{seed} ({res['minutes']:.1f} min; "
                      f"{(time.time()-t0)/60:.1f} total)", flush=True)

    # --- summary against the closed-form anchors -------------------------
    def norm(x):
        return (x - RANDOM) / (OPTIMUM - RANDOM)

    n_test = cfg.epochs // cfg.test_freq
    print(f"\n[e1] anchors: optimum={OPTIMUM} trap={TRAP} random={RANDOM} "
          f"(trap is BELOW random)", flush=True)
    summary = {}
    for method in METHODS:
        runs = [cells[(m, s)] for (m, s) in sorted(cells) if m == method]
        if not runs:
            continue
        finals = [r["greedy"][:n_test][-1] for r in runs if r["greedy"]]
        sampled_tail = [float(np.mean(r["sampled"][-5:])) for r in runs if r["sampled"]]
        trapped = sum(1 for f in finals if abs(f - TRAP) < (OPTIMUM - TRAP) * 0.15)
        summary[method] = {
            "greedy_final_per_seed": finals,
            "greedy_final_mean": float(np.mean(finals)) if finals else None,
            "greedy_final_norm": float(np.mean([norm(f) for f in finals])) if finals else None,
            "sampled_tail_mean": float(np.mean(sampled_tail)) if sampled_tail else None,
            "seeds_near_trap": trapped,
            "n": len(runs),
        }
        print(f"[e1] {method}: greedy_final={finals} "
              f"(mean {summary[method]['greedy_final_mean']}), "
              f"sampled_tail={summary[method]['sampled_tail_mean']:.1f}, "
              f"near-trap seeds: {trapped}/{len(runs)}", flush=True)

    with open(os.path.join(OUT, f"e1_summary_{n_seeds}seeds.json"), "w") as f:
        json.dump({"anchors": {"optimum": OPTIMUM, "trap": TRAP, "random": RANDOM},
                   "summary": summary}, f, indent=2)
    print(f"[e1] wrote {os.path.join(OUT, f'e1_summary_{n_seeds}seeds.json')}", flush=True)
