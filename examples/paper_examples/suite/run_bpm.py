r"""BPM experiment runner: PPO vs NF-GAE on the insurer and the BPM nets.

Protocol: 40 epochs x 8 episodes, greedy evaluation every 3 epochs on 20
episodes under common random numbers (eval_seed 555000), seeds 0-19, shared
hyperparameters (common.Hyper), no per-arm tuning. Anchors: random and
heuristic policies over 40 episodes (insurer_slow: the waiting heuristic).

Arms: ppo (SMDP-GAE PPO) and nfgae (net-factored GAE, paper/NFGAE_THEORY.md).

Run:  python run_bpm.py env=insurer N=2 [workers] [seeds=20] [methods=ppo,nfgae]
                        [epochs=40] [threads=k] [episodes=8] [plr=x]
                        [net=aepn|hgt|temb|tembf] [bases=K] [flat=1] [local=1]
                        [beta=0.5]   (SMDP discount; routes to *_beta<val>)
                        [postpone=0|1|component]  (no / global / per-component
                                     postpone; *_nopp, unsuffixed, *_ppc)
The insurer envs default to postpone=0. nfgae needs postpone=0 or component.
Output: suite_results_bpm_<env>_n<N>_ep<epochs>.../cells/N<N>__<method>__s<seed>.json
(epochs != 40 routes to a *_smoke<epochs> dir). Resumable by cell file.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from common import Hyper, set_seed, extract_metrics, threads_per_worker  # noqa: E402
from bpm_envs import BPM_BUILDERS, BPM_HEURISTICS  # noqa: E402
from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS  # noqa: E402

BPM_BUILDERS = {**BPM_BUILDERS, **INSURER_BUILDERS}
BPM_HEURISTICS = {**BPM_HEURISTICS, **INSURER_HEURISTICS}

ENV = "insurer"
N = 8
LENGTH = 20
EPOCHS = 40
EPISODES = 8
TEST_FREQ = 3
TEST_EPISODES = 20
EVAL_SEED = 555_000
SEEDS = 20
THREADS = None
BETA = None                    # SMDP discount override (Hyper.beta if None)
METHODS = ["ppo", "nfgae"]
POSTPONE = True
_POSTPONE_SET = False
POSTPONE_SCOPE = "global"
LOCAL = False                  # local=1: nfgae component turns, local observations (routes to *_local)
PLR = None                     # plr=x: policy learning rate override (routes to *_plr<x>)
FLAT = False                   # flat=1: flat graph observations (needs net=temb|tembf|aepn; routes to *_flat)
NET = "aepn"                   # net=hgt|temb|tembf|aepn: actor+critic encoder (default aepn; *_aepn dirs, hgt = the old unsuffixed dirs)
BASES = None                   # bases=K: AEPNStack basis count (default 8; routes to *_b<K>)

for _a in sys.argv[1:]:
    if _a.startswith("env="):
        ENV = _a.split("=", 1)[1]
    elif _a.startswith("N="):
        N = int(_a.split("=", 1)[1])
    elif _a.startswith("seeds="):
        SEEDS = int(_a.split("=", 1)[1])
    elif _a.startswith("methods="):
        METHODS = _a.split("=", 1)[1].split(",")
    elif _a.startswith("epochs="):
        EPOCHS = int(_a.split("=", 1)[1])
    elif _a.startswith("threads="):
        THREADS = int(_a.split("=", 1)[1])
    elif _a.startswith("beta="):
        BETA = float(_a.split("=", 1)[1])
    elif _a.startswith("episodes="):
        EPISODES = int(_a.split("=", 1)[1])
    elif _a.startswith("local="):
        LOCAL = _a.split("=", 1)[1] not in ("0", "false", "False")
    elif _a.startswith("plr="):
        PLR = float(_a.split("=", 1)[1])
    elif _a.startswith("flat="):
        FLAT = _a.split("=", 1)[1] not in ("0", "false", "False")
    elif _a.startswith("bases="):
        BASES = int(_a.split("=", 1)[1])
    elif _a.startswith("net="):
        NET = {"temb": "type_embed", "tembf": "type_embed_film"}.get(_a.split("=", 1)[1], _a.split("=", 1)[1])
    elif _a.startswith("postpone="):
        _POSTPONE_SET = True
        _v = _a.split("=", 1)[1]
        POSTPONE = _v not in ("0", "false", "False")
        if _v == "component":
            POSTPONE_SCOPE = "component"

# The insurer is work-conserving by design (insurer_env.py): no postpone unless asked.
if ENV.startswith("insurer") and not _POSTPONE_SET:
    POSTPONE = False
if set(METHODS) - {"ppo", "nfgae"}:
    raise SystemExit(f"unknown methods {sorted(set(METHODS) - {'ppo', 'nfgae'})}; choose from ppo, nfgae")
if ENV not in BPM_BUILDERS:
    raise SystemExit(f"unknown env {ENV!r}; choose from {sorted(BPM_BUILDERS)}")
BUILD = BPM_BUILDERS[ENV]
HEUR = BPM_HEURISTICS[ENV]
OUTDIR = Path(f"suite_results_bpm_{ENV}_n{N}_ep{EPOCHS}" + ("" if EPOCHS == 40 else f"_smoke{EPOCHS}")
              + ("" if BETA is None else f"_beta{BETA:g}")
              + ("" if POSTPONE else "_nopp")
              + ("_ppc" if POSTPONE and POSTPONE_SCOPE == "component" else "")
              + {"hgt": "", "type_embed": "_temb", "type_embed_film": "_tembf", "aepn": "_aepn"}[NET]
              + ("" if BASES is None else f"_b{BASES}")
              + ("" if EPISODES == 8 else f"_eps{EPISODES}")
              + ("" if PLR is None else f"_plr{PLR:g}")
              + ("_flat" if FLAT else "")
              + ("_local" if LOCAL else ""))
TAG = f"bpm-{ENV}-n{N}"


def _baselines():
    if getattr(BUILD, "wait_anchor", False):        # insurer slow-mismatch variant
        from insurer_env import wait_anchors
        return wait_anchors(BUILD, N, LENGTH)
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    rnd, heu = [], []
    for s in range(40):
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = BUILD(N, allow_postpone=False)
        rnd.append(float(env.testing_run(solver=RandomSolver(), length=LENGTH)))
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = BUILD(N, allow_postpone=False)
        heu.append(float(env.testing_run(solver=HeuristicSolver(HEUR), length=LENGTH)))
    return {"random_mean": float(np.mean(rnd)), "heuristic_mean": float(np.mean(heu)),
            "random_std": float(np.std(rnd)), "heuristic_std": float(np.std(heu)),
            "episodes": 40}


def _args(method, seed, cfg, logdir):
    a = {
        "algorithm": "ppo-clip",
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr if PLR is None else PLR, "policy_updates": cfg.policy_updates,
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
    }
    # Always explicit: the library default is aepn, so net=hgt must say so.
    net_kw = {"encoder": NET}
    if BASES is not None:
        net_kw["encoder_kwargs"] = {"num_bases": BASES}
    a["policy_kwargs"] = dict(net_kw)
    a["value_kwargs"] = dict(net_kw)
    if FLAT:
        a["flat_obs"] = True
    if LOCAL and method == "nfgae":
        a["local_obs"] = True
    a.update({"smdp_discount": True, "nfgae": method == "nfgae"})
    return a


def train_cell(method, seed, cfg, logdir, baselines):
    set_seed(seed)
    env = BUILD(N, allow_postpone=POSTPONE)
    env.postpone_scope = POSTPONE_SCOPE
    args = _args(method, seed, cfg, logdir)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LENGTH, args_dict=args)
    finally:
        sys.argv = saved
    m = extract_metrics(getattr(env, "training_history", {}) or {}, EPOCHS, TEST_FREQ)
    m.update({"env": ENV, "N": N, "method": method, "seed": seed,
              "epochs": EPOCHS, "episodes": EPISODES, "eval_seed": EVAL_SEED,
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
    print(f"\n[{TAG}] anchors random={r:.2f} heuristic={h:.2f} (gap {h-r:.2f})", flush=True)
    print(f"  {'method':<11} {'n':>3} {'final':>7} {'(SD)':>7} {'best':>7} {'collapsed':>9} {'ent':>6}", flush=True)
    rows = {}
    for m in METHODS:
        fin, best, ent = [], [], []
        for s in range(SEEDS):
            f = out / "cells" / f"N{N}__{m}__s{s}.json"
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
    (out / "summary.json").write_text(json.dumps({"env": ENV, "N": N, "baselines": baselines,
                                                  "rows": rows}, indent=2))


def _keep_awake():
    """Windows: stop the machine from sleeping while cells run (the N=8 pilot
    lost a night to modern standby on battery). ES_CONTINUOUS|ES_SYSTEM_REQUIRED;
    the display may still turn off. No-op elsewhere."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def main(workers):
    _keep_awake()
    cfg = Hyper()
    if BETA is not None:
        cfg.beta = BETA
        print(f"[{TAG}] beta override: {BETA}", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "cells").mkdir(exist_ok=True)
    logdir = str(OUTDIR / "train")

    bp = OUTDIR / f"baselines_N{N}.json"
    baselines = json.loads(bp.read_text()) if bp.exists() else _baselines()
    if not bp.exists():
        bp.write_text(json.dumps(baselines, indent=2))
    print(f"[{TAG}] anchors: random={baselines['random_mean']:.2f} "
          f"heuristic={baselines['heuristic_mean']:.2f}", flush=True)

    pending = [(m, s) for s in range(SEEDS) for m in METHODS
               if not (OUTDIR / "cells" / f"N{N}__{m}__s{s}.json").exists()]
    print(f"[{TAG}] {len(pending)} cells ({len(METHODS)} arms x {SEEDS} seeds, {EPOCHS} epochs), "
          f"{workers} workers", flush=True)

    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = THREADS if THREADS else threads_per_worker(workers)
        print(f"[{TAG}] torch threads per worker: {tpw}", flush=True)
        r, h = baselines["random_mean"], baselines["heuristic_mean"]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, (m, s, cfg, logdir, baselines, tpw)) for (m, s) in pending]
            for fut in as_completed(futs):
                m, s, res, err = fut.result()
                if err:
                    print(f"[{TAG}] !!! {m}_s{s} FAILED:\n{err}", flush=True)
                    continue
                (OUTDIR / "cells" / f"N{N}__{m}__s{s}.json").write_text(json.dumps(res))
                gf = res.get("greedy_final")
                nrm = (gf - r) / (h - r) if gf is not None else float("nan")
                print(f"[{TAG}] <<< {m:<11} s{s:<2} final={nrm:+.3f} "
                      f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    summary(OUTDIR, baselines)


if __name__ == "__main__":
    w = next((int(a) for a in sys.argv[1:] if a.isdigit()), 4)
    main(w)
