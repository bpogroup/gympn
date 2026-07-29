"""N-copies scaling sweep: does causal-component-factored credit (ccf) beat PPO
by a margin that GROWS with the number of independent components N?

Prediction (LINEAGE_SPARSE_CORRECTION follow-up / multi-agent credit story):
at a fixed training budget, PPO's advantage variance ~ N (contaminated by the
other copies' independent reward noise) so its final DEGRADES as N grows, while
ccf factors that noise away and holds. The ccf-PPO gap should rise ~sqrt(N).

Run: python run_ncopies_sweep.py [workers]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import _set_seed, _extract_metrics, _threads_per_worker  # noqa: E402
from ncopies_env import (make_n_copies, ncopies_heuristic,  # noqa: E402
                         make_n_copies_hard, ncopies_hard_heuristic)

NS = [1, 2, 4, 8]
SEEDS = 3
LENGTH = 20
METHODS = ("ppo", "ccf")

# selected by mode (set in main): base match/cross vs hard 3-type/2-employee
HARD = False
def MAKE(n, **kw): return (make_n_copies_hard if HARD else make_n_copies)(n, **kw)
def HEUR(): return ncopies_hard_heuristic if HARD else ncopies_heuristic
def OUTDIR(): return "suite_results_ncopies_hard" if HARD else "suite_results_ncopies"


def _baselines(n):
    from gympn.environment import AEPN_Env  # noqa
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    rnd, heu = [], []
    for s in range(15):
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = MAKE(n, causal_rl=False, allow_postpone=False)
        rnd.append(float(env.testing_run(solver=RandomSolver(), length=LENGTH)))
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = MAKE(n, causal_rl=False, allow_postpone=False)
        heu.append(float(env.testing_run(solver=HeuristicSolver(HEUR()), length=LENGTH)))
    return {"random_mean": float(np.mean(rnd)), "heuristic_mean": float(np.mean(heu))}


def _args(method, seed, cfg, logdir):
    a = {
        "algorithm": "ppo-clip",
        "episodes": cfg.episodes_per_epoch, "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam,
        "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_beta": cfg.causal_beta,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": cfg.test_freq,
        "test_episodes": getattr(cfg, "test_episodes", 10),
        "save_freq": 1_000_000, "name": f"{method}__s{seed}",
        "datetag": False, "logdir": logdir,
    }
    if method == "ccf":
        a.update({"causal_rl": True, "causal_scheme": "ccf"})
    else:  # ppo: discount-matched baseline (SMDP-GAE at the same beta)
        a.update({"causal_rl": False, "smdp_discount": True})
    return a


def train_cell(n, method, seed, cfg, logdir, baselines):
    _set_seed(seed)
    env = MAKE(n, causal_rl=(method == "ccf"), allow_postpone=False)
    args = _args(method, seed, cfg, logdir)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LENGTH, args_dict=args)
    finally:
        sys.argv = saved
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfg)
    m.update({"N": n, "method": method, "seed": seed,
              "baselines": baselines, "minutes": (time.time() - t0) / 60.0})
    return m


def _worker(payload):
    import torch
    global HARD
    n, method, seed, cfg, logdir, baselines, threads, hard = payload
    HARD = bool(hard)
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (n, method, seed, train_cell(n, method, seed, cfg, logdir, baselines), None)
    except Exception:
        import traceback
        return (n, method, seed, None, traceback.format_exc())


def main(workers=3):
    cfg = stoch_config()
    cfg.epochs = 15
    cfg.episodes_per_epoch = 8
    cfg.test_freq = 3

    out = Path(OUTDIR())
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    baselines = {}
    for n in NS:
        bp = out / f"baselines_N{n}.json"
        if bp.exists():
            baselines[n] = json.loads(bp.read_text())
        else:
            baselines[n] = _baselines(n)
            bp.write_text(json.dumps(baselines[n]))
        b = baselines[n]
        print(f"[ncopies] N={n}: random={b['random_mean']:.2f} heuristic={b['heuristic_mean']:.2f}",
              flush=True)

    pending = [(n, m, s) for n in NS for m in METHODS for s in range(SEEDS)
               if not (out / "cells" / f"N{n}__{m}__s{s}.json").exists()]
    print(f"[ncopies] {len(pending)} cells, {workers} workers", flush=True)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    tpw = _threads_per_worker(workers)
    payloads = [(n, m, s, cfg, logdir, baselines[n], tpw, HARD) for (n, m, s) in pending]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for fut in as_completed(futs):
            n, m, s, res, err = fut.result()
            if err:
                print(f"[ncopies] !!! N{n}/{m}_s{s} FAILED:\n{err}", flush=True)
                continue
            (out / "cells" / f"N{n}__{m}__s{s}.json").write_text(json.dumps(res))
            print(f"[ncopies] <<< N{n}/{m}_s{s} greedy_final={res.get('greedy_final')} "
                  f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    # summary: normalized final vs N, and the ccf-ppo gap
    print("\n[ncopies] === scaling: normalized final vs N (gap should GROW with N) ===", flush=True)
    print(f"  {'N':>3} | {'ppo':>14} | {'ccf':>14} | {'ccf-ppo':>8}", flush=True)
    for n in NS:
        b = baselines[n]; r, h = b["random_mean"], b["heuristic_mean"]
        def norm(v): return (v - r) / (h - r) if h != r else 0.0
        vals = {}
        for m in METHODS:
            fs = []
            for s in range(SEEDS):
                c = out / "cells" / f"N{n}__{m}__s{s}.json"
                if c.exists():
                    mm = json.loads(c.read_text())
                    if mm.get("greedy_final") is not None:
                        fs.append(norm(mm["greedy_final"]))
            vals[m] = (np.mean(fs), np.std(fs)) if fs else (float('nan'), 0)
        gap = vals["ccf"][0] - vals["ppo"][0]
        print(f"  {n:>3} | {vals['ppo'][0]:>6.3f}+-{vals['ppo'][1]:<5.2f} | "
              f"{vals['ccf'][0]:>6.3f}+-{vals['ccf'][1]:<5.2f} | {gap:>+8.3f}", flush=True)


if __name__ == "__main__":
    if "hard" in sys.argv[1:]:
        HARD = True
    workers = next((int(a) for a in sys.argv[1:] if a.isdigit()), 3)
    print(f"[ncopies] mode={'HARD' if HARD else 'base'}", flush=True)
    main(workers)