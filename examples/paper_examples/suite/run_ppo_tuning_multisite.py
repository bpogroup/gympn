r"""PPO hyperparameter grid on multi-site routing (the realistic environment).

WHY. In the converged 20-seed comparison PPO scores 0.161 normalized with 14 of
20 seeds collapsed, against cgae_cflow's 0.673 with none. A referee will read
that gap as an untuned baseline. Every arm in the paper shares ONE
hyperparameter set with no per-arm tuning (the conservative design for a
credit-assignment comparison), so the question this grid answers is narrow:
does ANY reasonable PPO setting close the gap? Only the baseline is tuned; the
method keeps the shared setting. If the best PPO cell is still far below 0.673
the headline gain is not a tuning artifact; if it is not, the paper must say so.

WHAT VARIES. One knob per config around the shared setting (lr 3e-4, lambda
0.95, entropy bonus 0.01, which anneals linearly to 0 over training):

  base    : the shared setting, re-run under common random numbers (eval_seed)
  lr_lo   : policy and value lr 1e-4
  lr_hi   : policy and value lr 1e-3
  lam_lo  : GAE lambda 0.90
  ent_hi  : entropy bonus 0.03  (collapse = low-entropy poor policy; this is the
            classic anti-collapse knob)

PROTOCOL. Identical to run_multisite_validate.py, which produced the paper's
multi-site cells: 4 sites x 2 dedicated specialists, n_flex = 0, horizon 20,
40 epochs x 8 episodes, greedy eval every 2 epochs on 12 episodes, seeds 0-19,
allow_postpone = False. Normalization anchors are COPIED from
suite_results_multisite_bf/baselines.json so the scale is the paper's. One
difference is deliberate: eval_seed is set, so evaluation uses common random
numbers. The paper's multi-site cells were produced without it, which is why
`base` is re-run here rather than borrowed.

Run: python run_ppo_tuning_multisite.py [workers] [seeds=20] [configs=base,lr_lo,...]
Resumable by cell file.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import _set_seed, _extract_metrics, _threads_per_worker  # noqa: E402
from multisite_env import make_multisite  # noqa: E402

N_SITES, N_LOCAL, N_FLEX = 4, 1, 0
LEN = 20
EPOCHS = 40
EPISODES = 8
TEST_EPISODES = 12
EVAL_SEED = 555_000
SEEDS = 20
OUT_DIR = Path("suite_results_ppo_tuning_multisite")
SRC_BASELINES = Path("suite_results_multisite_bf") / "baselines.json"

CONFIGS = {
    "base":   {},
    "lr_lo":  {"policy_lr": 1e-4, "value_lr": 1e-4},
    "lr_hi":  {"policy_lr": 1e-3, "value_lr": 1e-3},
    "lam_lo": {"lam": 0.90},
    "ent_hi": {"ent_bonus": 0.03},
}
ACTIVE = list(CONFIGS)

for _a in sys.argv[1:]:
    if _a.startswith("seeds="):
        SEEDS = int(_a.split("=", 1)[1])
    elif _a.startswith("configs="):
        ACTIVE = _a.split("=", 1)[1].split(",")
    elif _a.startswith("epochs="):          # smoke only: routes to a scratch dir
        EPOCHS = int(_a.split("=", 1)[1])
        OUT_DIR = Path(f"suite_results_ppo_tuning_multisite_smoke{EPOCHS}")


def _args(cfgname, seed, cfg, logdir):
    a = {
        "algorithm": "ppo-clip",
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam, "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_beta": cfg.causal_beta,
        "causal_rl": False, "smdp_discount": True,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": 2, "test_episodes": TEST_EPISODES,
        "eval_seed": EVAL_SEED,
        "save_freq": 10**9, "name": f"ppo_{cfgname}__s{seed}", "datetag": False,
        "logdir": logdir,
    }
    a.update(CONFIGS[cfgname])
    return a


def train_cell(cfgname, seed, cfg, logdir, baselines):
    _set_seed(seed)
    env = make_multisite(N_SITES, N_LOCAL, N_FLEX, causal_rl=False, allow_postpone=False)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    args = _args(cfgname, seed, cfg, logdir)
    try:
        env.training_run(length=LEN, args_dict=args)
    finally:
        sys.argv = saved
    cfg_for_extract = stoch_config()
    cfg_for_extract.epochs = EPOCHS
    cfg_for_extract.test_freq = 2
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfg_for_extract)
    m.update({"method": "ppo", "config": cfgname, "overrides": CONFIGS[cfgname],
              "hyper": {k: args[k] for k in ("policy_lr", "value_lr", "lam", "ent_bonus",
                                             "eps", "gam", "policy_kld_limit", "batch_size",
                                             "episodes", "epochs")},
              "eval_seed": EVAL_SEED, "seed": seed, "baselines": baselines,
              "minutes": (time.time() - t0) / 60.0})
    return m


def _worker(payload):
    import torch
    cfgname, seed, cfg, logdir, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (cfgname, seed, train_cell(cfgname, seed, cfg, logdir, baselines), None)
    except Exception:
        import traceback
        return (cfgname, seed, None, traceback.format_exc())


def summary(out, baselines):
    r, h = baselines["random_mean"], baselines["heuristic_mean"]
    norm = lambda v: (v - r) / (h - r)
    print(f"\n[ppo-tune] anchors random={r:.2f} heuristic={h:.2f}; paper: PPO 0.161 [14 collapsed], "
          f"cgae_cflow 0.673 [0]", flush=True)
    print(f"  {'config':<8} {'n':>3} {'final':>7} {'(SD)':>7} {'best':>7} {'collapsed':>9} {'ent':>6}",
          flush=True)
    rows = {}
    for c in ACTIVE:
        fin, best, ent = [], [], []
        for s in range(SEEDS):
            f = out / "cells" / f"ppo_{c}__s{s}.json"
            if f.exists():
                j = json.loads(f.read_text())
                if j.get("greedy_final") is not None:
                    fin.append(norm(j["greedy_final"])); best.append(norm(j["greedy_best"]))
                    ent.append(j.get("entropy_final") or float("nan"))
        if fin:
            fin = np.array(fin)
            rows[c] = {"n": len(fin), "final_mean": float(fin.mean()), "final_sd": float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
                       "best_mean": float(np.mean(best)), "collapsed": int((fin <= 0.25).sum()),
                       "entropy_final": float(np.nanmean(ent)), "finals": fin.tolist()}
            print(f"  {c:<8} {len(fin):>3} {fin.mean():>7.3f} {rows[c]['final_sd']:>7.3f} "
                  f"{np.mean(best):>7.3f} {rows[c]['collapsed']:>9d} {np.nanmean(ent):>6.2f}", flush=True)
    (out / "summary.json").write_text(json.dumps({"baselines": baselines, "rows": rows}, indent=2))


def main(workers):
    cfg = stoch_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cells").mkdir(exist_ok=True)
    logdir = str(OUT_DIR / "train")

    bp = OUT_DIR / "baselines.json"
    if not bp.exists():
        shutil.copy(SRC_BASELINES, bp)
    baselines = json.loads(bp.read_text())

    # seeds outer, configs inner: partial results stay balanced across configs
    pending = [(c, s) for s in range(SEEDS) for c in ACTIVE
               if not (OUT_DIR / "cells" / f"ppo_{c}__s{s}.json").exists()]
    print(f"[ppo-tune] {len(pending)} cells ({len(ACTIVE)} configs x {SEEDS} seeds, {EPOCHS} epochs), "
          f"{workers} workers, eval_seed={EVAL_SEED}", flush=True)
    print(f"[ppo-tune] configs: {json.dumps({c: CONFIGS[c] for c in ACTIVE})}", flush=True)

    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(workers)
        payloads = [(c, s, cfg, logdir, baselines, tpw) for (c, s) in pending]
        t0 = time.time()
        r, h = baselines["random_mean"], baselines["heuristic_mean"]
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                c, s, res, err = fut.result()
                if err:
                    print(f"[ppo-tune] !!! {c}_s{s} FAILED:\n{err}", flush=True)
                    continue
                (OUT_DIR / "cells" / f"ppo_{c}__s{s}.json").write_text(json.dumps(res))
                gf = res.get("greedy_final")
                nrm = (gf - r) / (h - r) if gf is not None else float("nan")
                print(f"[ppo-tune] <<< {c:<7} s{s:<2} final={nrm:+.3f} "
                      f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    summary(OUT_DIR, baselines)


if __name__ == "__main__":
    w = next((int(a) for a in sys.argv[1:] if a.isdigit()), 6)
    main(w)
