r"""Multi-site routing benchmark: PPO vs NF-GAE.

PROTOCOL: 40 epochs x 20 episodes, greedy eval every 2 epochs on 20 episodes,
eval_seed 555000 (scenario i = eval_seed + i for every arm/seed/epoch), seeds
0-19, shared hyperparameters (common.Hyper), no per-arm tuning,
allow_postpone=False. Anchors are the stored ones
(suite_results_multisite_bf/baselines.json: random 62.90, heuristic 80.25),
computed on the same no-postpone net.

Run: python run_multisite_protocol.py [workers] [seeds=20] [methods=ppo,nfgae] [epochs=40]
                                      [net=aepn|hgt] [flat=1] [threads=T] [tag=X]
(epochs != 40 routes to a scratch dir: smoke only). Resumable by cell file.
net: actor+critic encoder, default aepn. The cells in
suite_results_multisite_protocol are HGT (PPO only), so net=hgt keeps that
directory and any other net routes to suite_results_multisite_protocol_<net>[_flat]:
the two networks never share a results directory. flat=1: flat graph
observations (gympn/flat_graph.py; non-HGT nets). threads=T overrides the
per-worker torch threads (default: physical cores / workers).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from common import Hyper, set_seed, extract_metrics, threads_per_worker  # noqa: E402
from multisite_env import make_multisite  # noqa: E402

N_SITES, N_LOCAL, N_FLEX = 4, 1, 0
LENGTH = 20
EPOCHS = 40
EPISODES = 20
TEST_FREQ = 2
TEST_EPISODES = 20
EVAL_SEED = 555_000
SEEDS = 20
OUTDIR = Path("suite_results_multisite_protocol")
SRC_BASELINES = Path("suite_results_multisite_bf") / "baselines.json"

METHODS = ["ppo", "nfgae"]

NET = "aepn"
FLAT = False
THREADS = None
TAG = None      # tag=X: separate results directory *_X (e.g. a rerun next to existing cells)

for _a in sys.argv[1:]:
    if _a.startswith("net="):
        NET = _a.split("=", 1)[1]
    elif _a.startswith("flat="):
        FLAT = _a.split("=", 1)[1] not in ("0", "false", "False")
    elif _a.startswith("tag="):
        TAG = _a.split("=", 1)[1]
    elif _a.startswith("threads="):
        THREADS = int(_a.split("=", 1)[1])
    elif _a.startswith("seeds="):
        SEEDS = int(_a.split("=", 1)[1])
    elif _a.startswith("methods="):
        METHODS = _a.split("=", 1)[1].split(",")
    elif _a.startswith("epochs="):
        EPOCHS = int(_a.split("=", 1)[1])
        OUTDIR = Path(f"suite_results_multisite_protocol_smoke{EPOCHS}")
if NET != "hgt" or FLAT:
    OUTDIR = Path(f"{OUTDIR}_{NET}" + ("_flat" if FLAT else ""))
if TAG:
    OUTDIR = Path(f"{OUTDIR}_{TAG}")
if set(METHODS) - {"ppo", "nfgae"}:
    raise SystemExit(f"unknown methods {sorted(set(METHODS) - {'ppo', 'nfgae'})}; choose from ppo, nfgae")


def _args(method, seed, cfg, logdir):
    a = {
        "algorithm": "ppo-clip",
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam, "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "beta": cfg.beta,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": TEST_FREQ, "test_episodes": TEST_EPISODES,
        "eval_seed": EVAL_SEED,
        "save_freq": 1_000_000, "name": f"{method}__s{seed}", "datetag": False,
        "logdir": logdir,
        "policy_kwargs": {"encoder": NET}, "value_kwargs": {"encoder": NET},
    }
    if FLAT:
        a["flat_obs"] = True
    a.update({"smdp_discount": True, "nfgae": method == "nfgae"})
    return a


def train_cell(method, seed, cfg, logdir, baselines):
    set_seed(seed)
    env = make_multisite(N_SITES, N_LOCAL, N_FLEX, allow_postpone=False)
    args = _args(method, seed, cfg, logdir)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LENGTH, args_dict=args)
    finally:
        sys.argv = saved
    m = extract_metrics(getattr(env, "training_history", {}) or {}, EPOCHS, TEST_FREQ)
    m.update({"env": "multisite", "n_sites": N_SITES, "n_local": N_LOCAL, "n_flex": N_FLEX,
              "allow_postpone": False, "method": method, "seed": seed,
              "net": NET, "flat_obs": FLAT,
              "epochs": EPOCHS, "episodes": EPISODES, "test_freq": TEST_FREQ,
              "test_episodes": TEST_EPISODES, "eval_seed": EVAL_SEED,
              "baselines": baselines, "minutes": (time.time() - t0) / 60.0})
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


def summary(out, baselines):
    r, h = baselines["random_mean"], baselines["heuristic_mean"]
    norm = lambda v: (v - r) / (h - r)
    print(f"\n[multisite-protocol] anchors random={r:.2f} heuristic={h:.2f} (gap {h-r:.2f})",
          flush=True)
    print(f"  {'method':<11} {'n':>3} {'final':>7} {'(SD)':>7} {'best':>7} {'collapsed':>9} {'ent':>6}", flush=True)
    rows = {}
    for m in METHODS:
        fin, best, ent = [], [], []
        for s in range(SEEDS):
            f = out / "cells" / f"{m}__s{s}.json"
            if f.exists():
                j = json.loads(f.read_text())
                if j.get("greedy_final") is not None:
                    fin.append(norm(j["greedy_final"])); best.append(norm(j["greedy_best"]))
                    ent.append(j.get("entropy_final") or float("nan"))
        if fin:
            fin = np.array(fin)
            rows[m] = {"n": len(fin), "final_mean": float(fin.mean()),
                       "final_sd": float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
                       "best_mean": float(np.mean(best)), "collapsed": int((fin <= 0.25).sum()),
                       "entropy_final": float(np.nanmean(ent)), "finals": fin.tolist()}
            print(f"  {m:<11} {len(fin):>3} {fin.mean():>7.3f} {rows[m]['final_sd']:>7.3f} "
                  f"{np.mean(best):>7.3f} {rows[m]['collapsed']:>9d} {np.nanmean(ent):>6.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps({"baselines": baselines, "rows": rows}, indent=2))


def main(workers):
    cfg = Hyper()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "cells").mkdir(exist_ok=True)
    logdir = str(OUTDIR / "train")

    bp = OUTDIR / "baselines.json"
    if not bp.exists():
        shutil.copy(SRC_BASELINES, bp)
    baselines = json.loads(bp.read_text())

    print(f"[multisite-protocol] anchors: random={baselines['random_mean']:.2f} "
          f"heuristic={baselines['heuristic_mean']:.2f}", flush=True)

    # seeds outer, arms inner: partial results stay balanced across arms
    pending = [(m, s) for s in range(SEEDS) for m in METHODS
               if not (OUTDIR / "cells" / f"{m}__s{s}.json").exists()]
    tpw = THREADS if THREADS is not None else threads_per_worker(workers)
    print(f"[multisite-protocol] {len(pending)} cells ({len(METHODS)} arms x {SEEDS} seeds, "
          f"{EPOCHS} epochs x {EPISODES} episodes, eval every {TEST_FREQ} on {TEST_EPISODES}), "
          f"{workers} workers x {tpw} threads, eval_seed={EVAL_SEED}, net={NET}"
          + (", flat" if FLAT else "") + f" -> {OUTDIR}", flush=True)

    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        payloads = [(m, s, cfg, logdir, baselines, tpw) for (m, s) in pending]
        t0 = time.time()
        r, h = baselines["random_mean"], baselines["heuristic_mean"]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                m, s, res, err = fut.result()
                if err:
                    print(f"[multisite-protocol] !!! {m}_s{s} FAILED:\n{err}", flush=True)
                    continue
                (OUTDIR / "cells" / f"{m}__s{s}.json").write_text(json.dumps(res))
                gf = res.get("greedy_final")
                nrm = (gf - r) / (h - r) if gf is not None else float("nan")
                print(f"[multisite-protocol] <<< {m:<11} s{s:<2} final={nrm:+.3f} "
                      f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    summary(OUTDIR, baselines)


if __name__ == "__main__":
    w = next((int(a) for a in sys.argv[1:] if a.isdigit()), 6)
    main(w)
