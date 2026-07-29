"""Multi-site (flex=0, independent) diagnostic: does ccf beat PPO once both are
adequately trained? 30 epochs, 6 seeds each, learning curves + final CIs.

Resolves the 15-epoch/2-seed tie (one collapsed ccf seed). If ccf's curve rises
faster/steadier and its final CI clears PPO's, the realistic env showcases ccf;
if the means coincide with tight CIs, it's a genuine wash and we rethink.

Run: python run_multisite_validate.py [workers]
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
from multisite_env import make_multisite, multisite_heuristic  # noqa: E402

N_SITES, N_LOCAL, N_FLEX = 4, 1, 0     # independent config, strong learnable signal
LEN = 20
EPOCHS = 40
EPISODES = 8
SEEDS = 12
METHODS = ("ppo", "ccf")
OUT_DIR = "suite_results_multisite_bf"   # brute-force run (30ep/6seed kept separately)


def _baselines():
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    rnd, heu = [], []
    for s in range(20):
        random.seed(1000 + s); np.random.seed(1000 + s)
        rnd.append(float(make_multisite(N_SITES, N_LOCAL, N_FLEX, allow_postpone=False)
                         .testing_run(solver=RandomSolver(), length=LEN)))
        random.seed(1000 + s); np.random.seed(1000 + s)
        heu.append(float(make_multisite(N_SITES, N_LOCAL, N_FLEX, allow_postpone=False)
                         .testing_run(solver=HeuristicSolver(multisite_heuristic), length=LEN)))
    return {"random_mean": float(np.mean(rnd)), "heuristic_mean": float(np.mean(heu))}


def _args(method, seed, cfg, logdir):
    a = {
        "algorithm": "ppo-clip",
        "episodes": cfg.episodes_per_epoch, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam, "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None), "causal_beta": cfg.causal_beta,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": cfg.test_freq, "test_episodes": 12,
        "save_freq": 10**9, "name": f"{method}__s{seed}", "datetag": False, "logdir": logdir,
    }
    if method == "ccf":
        a.update({"causal_rl": True, "causal_scheme": "ccf"})
    else:
        a.update({"causal_rl": False, "smdp_discount": True})
    return a


def train_cell(method, seed, cfg, logdir, baselines):
    _set_seed(seed)
    env = make_multisite(N_SITES, N_LOCAL, N_FLEX,
                         causal_rl=(method == "ccf"), allow_postpone=False)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LEN, args_dict=_args(method, seed, cfg, logdir))
    finally:
        sys.argv = saved
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfg)
    m.update({"method": method, "seed": seed, "baselines": baselines,
              "minutes": (time.time() - t0) / 60.0})
    return m


def _worker(payload):
    import torch
    method, seed, cfg, logdir, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (method, seed, train_cell(method, seed, cfg, logdir, baselines), None)
    except Exception:
        import traceback
        return (method, seed, None, traceback.format_exc())


def _summary(out, baselines):
    r, h = baselines["random_mean"], baselines["heuristic_mean"]
    def norm(v): return (v - r) / (h - r) if h != r else 0.0
    print(f"\n[multisite] baselines: random={r:.1f} heuristic={h:.1f}", flush=True)
    stats = {}
    for m in METHODS:
        finals, curves = [], []
        for s in range(SEEDS):
            c = out / "cells" / f"{m}__s{s}.json"
            if c.exists():
                mm = json.loads(c.read_text())
                if mm.get("greedy_final") is not None:
                    finals.append(norm(mm["greedy_final"]))
                gc = mm.get("greedy_curve")
                if gc:
                    curves.append([norm(x) for x in gc])
        finals = np.array(finals)
        n = len(finals)
        mean = float(finals.mean()) if n else float("nan")
        ci = float(1.96 * finals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        stats[m] = {"mean": mean, "ci": ci, "n": n, "finals": finals.tolist(), "curves": curves}
        print(f"[multisite] {m:>4}: norm_final = {mean:.3f} +/- {ci:.3f} (95% CI, n={n})  "
              f"seeds={np.round(finals,3).tolist()}", flush=True)
    gap = stats["ccf"]["mean"] - stats["ppo"]["mean"]
    # non-overlap of 95% CIs => significant at ~0.05
    lo_ccf = stats["ccf"]["mean"] - stats["ccf"]["ci"]
    hi_ppo = stats["ppo"]["mean"] + stats["ppo"]["ci"]
    verdict = "ccf > ppo (95% CIs disjoint)" if lo_ccf > hi_ppo else "inconclusive (CIs overlap)"
    print(f"[multisite] ccf-ppo gap = {gap:+.3f}  -> {verdict}", flush=True)
    # compact learning-curve comparison (mean over seeds at each recorded epoch)
    print("[multisite] learning curves (normalized, mean over seeds):", flush=True)
    for m in METHODS:
        cs = stats[m]["curves"]
        if cs:
            L = min(len(c) for c in cs)
            arr = np.array([c[:L] for c in cs])
            mc = arr.mean(0)
            idx = np.linspace(0, L - 1, min(L, 8)).astype(int)
            print(f"    {m:>4}: " + "  ".join(f"{mc[i]:.2f}" for i in idx), flush=True)
    (out / "summary.json").write_text(json.dumps(
        {"baselines": baselines, "stats": {m: {k: v for k, v in stats[m].items() if k != "curves"}
                                           for m in METHODS}, "gap": gap, "verdict": verdict}, indent=2))


def main(workers=3):
    cfg = stoch_config()
    cfg.epochs = EPOCHS
    cfg.episodes_per_epoch = EPISODES
    cfg.test_freq = 2

    out = Path(OUT_DIR)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    bp = out / "baselines.json"
    baselines = json.loads(bp.read_text()) if bp.exists() else _baselines()
    if not bp.exists():
        bp.write_text(json.dumps(baselines))
    print(f"[multisite] baselines: random={baselines['random_mean']:.1f} "
          f"heuristic={baselines['heuristic_mean']:.1f}", flush=True)

    pending = [(m, s) for m in METHODS for s in range(SEEDS)
               if not (out / "cells" / f"{m}__s{s}.json").exists()]
    print(f"[multisite] {len(pending)} cells ({EPOCHS} epochs, {SEEDS} seeds), {workers} workers",
          flush=True)

    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(workers)
        payloads = [(m, s, cfg, logdir, baselines, tpw) for (m, s) in pending]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                m, s, res, err = fut.result()
                if err:
                    print(f"[multisite] !!! {m}_s{s} FAILED:\n{err}", flush=True)
                    continue
                (out / "cells" / f"{m}__s{s}.json").write_text(json.dumps(res))
                print(f"[multisite] <<< {m}_s{s} final={res.get('greedy_final')} "
                      f"({res['minutes']:.1f}min; {(time.time()-t0)/60:.1f} total)", flush=True)

    _summary(out, baselines)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3)