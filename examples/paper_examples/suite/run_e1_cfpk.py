"""X12a: G1 generality — cfpk on the E1 chain-vs-shortcut family.

The acid test from CAUSAL_LINEAGE_RETHINK.md §7: E1's variants are
one-shot / repeated foreclosing CHOICES with NO resource pools — the
topology class where the retired R5 (shadow prices) provably had nothing
to average over. cfpk must run here with zero env-specific lines (it does:
same method key, same code path as the s1 probe).

Writes cfpk cells into the SAME per-variant cell stores as run_e1_chain.py
(e1_chain / e1b_oneshot / e1c_oneshot3), so the cached ppo/lrq/rec
baselines remain the comparison set.

Registered predictions (written before the run, 2026-07-19):
  P1 (do-no-harm, oneshot3): PPO is already 10/10 optimal at k=3; cfpk
      must stay 10/10 (constant preference pressure must not break a
      solved one-shot choice).
  P2 (win, oneshot k=2): the noisy discriminator — baselines trapped
      rec 3/5, lrq 2/5, ppo 1/5. The true gap (12 vs 7) is huge relative
      to noise, so the fork gate should pass it trivially: cfpk 0/10
      near-trap seeds.
  P3 (do-no-harm, full): repeated choice, everyone converges; cfpk mean
      greedy_final within noise of ppo's.

Run: python run_e1_cfpk.py [n_seeds] [n_workers] [variant|all]
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import SuiteConfig  # noqa: E402
from run_suite import _set_seed, _threads_per_worker  # noqa: E402
from run_e1_chain import VARIANTS  # noqa: E402


def run_one_cfpk(seed: int, cfg: SuiteConfig, variant: str) -> dict:
    _set_seed(seed)
    spec = VARIANTS[variant]
    env = spec["builder"](causal_rl=False, allow_postpone=True,
                          causal_postpone_tokenflow=False)
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
        "causal_rl": False,
        "causal_beta": cfg.causal_beta,
        "smdp_discount": True,
        "cf_fork_prob": 0.25,
        "cf_reps": 3,
        "cf_gate": 2.0,
        "cf_lookahead": 6.0,
        "cf_max_forks": 2,
        "cf_coef": 1.0,
        "cf_updates": 2,
        "cf_anneal": False,
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
        "name": f"{spec['out']}__cfpk__s{seed}",
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
        "method": "cfpk",
        "seed": seed,
        "sampled": [float(x) for x in history.get("mean_returns", [])],
        "greedy": [float(x) for x in history.get("test_mean_returns", [])],
        "entropy": [float(x) for x in history.get("policy_ent", [])],
        "minutes": (time.time() - t0) / 60.0,
    }


def _cell_worker(payload):
    import torch
    seed, cfg, threads, variant = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (variant, seed, run_one_cfpk(seed, cfg, variant), None)
    except Exception:
        import traceback
        return (variant, seed, None, traceback.format_exc())


def main(n_seeds, n_workers, variants):
    here = os.path.dirname(os.path.abspath(__file__))
    pending = []
    for variant in variants:
        cells_dir = os.path.join(here, VARIANTS[variant]["out"], "cells")
        os.makedirs(cells_dir, exist_ok=True)
        for seed in range(n_seeds):
            if not os.path.exists(os.path.join(cells_dir, f"cfpk_s{seed}.json")):
                pending.append((variant, seed))
    print(f"[e1-cfpk] {len(pending)} cells to train across {variants}, "
          f"{n_workers} workers", flush=True)

    t0 = time.time()
    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(n_workers)

        def _cfg_for(variant):
            cfg = SuiteConfig()
            if variant.startswith("oneshot"):
                cfg.episodes_per_epoch = 60   # same override as run_e1_chain
            return cfg

        payloads = [(s, _cfg_for(v), tpw, v) for (v, s) in pending]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_cell_worker, p) for p in payloads]
            for fut in as_completed(futs):
                variant, seed, res, err = fut.result()
                if err is not None:
                    print(f"[e1-cfpk] !!! {variant}/cfpk_s{seed} FAILED:\n{err}",
                          flush=True)
                    continue
                cells_dir = os.path.join(here, VARIANTS[variant]["out"], "cells")
                with open(os.path.join(cells_dir, f"cfpk_s{seed}.json"), "w") as f:
                    json.dump(res, f, indent=2)
                gf = res["greedy"][-1] if res["greedy"] else None
                print(f"[e1-cfpk] <<< {variant}/cfpk_s{seed} greedy_final={gf} "
                      f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)",
                      flush=True)

    # ---- summary vs cached baselines ------------------------------------
    cfg = SuiteConfig()
    n_test = cfg.epochs // cfg.test_freq
    for variant in variants:
        spec = VARIANTS[variant]
        cells_dir = os.path.join(here, spec["out"], "cells")
        opt, trap, rnd = spec["optimum"], spec["trap"], spec["random"]
        print(f"\n[e1-cfpk] === {variant} (optimum={opt} trap={trap} "
              f"random={rnd}) ===", flush=True)
        for method in ("cfpk", "ppo", "lrq", "rec"):
            finals = []
            for fn in sorted(os.listdir(cells_dir)):
                if fn.startswith(f"{method}_s"):
                    r = json.loads(open(os.path.join(cells_dir, fn)).read())
                    if r.get("greedy"):
                        finals.append(r["greedy"][:n_test][-1])
            if not finals:
                continue
            trapped = sum(1 for f in finals if abs(f - trap) < (opt - trap) * 0.15)
            at_opt = sum(1 for f in finals if abs(f - opt) < (opt - trap) * 0.15)
            print(f"[e1-cfpk] {method:<5} finals={finals} "
                  f"mean={np.mean(finals):.2f} at-optimum {at_opt}/{len(finals)} "
                  f"near-trap {trapped}/{len(finals)}", flush=True)


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    variants = list(VARIANTS) if which == "all" else [which]
    main(n_seeds, n_workers, variants)