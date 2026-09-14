"""H1+H2 gate for lrq2c (LINEAGE_SPARSE_CORRECTION.md).

lrq2c = PPO + lrq2 lineage advantage + sparse EXACT indirect correction (CRN
forks at foreclosure-gated / structurally-contested decisions only). Compared
head-to-head with lrq2 (no correction), same seeds.

  H1 (do-no-harm): on a_sequence_joint (direct-dominated, lrq perfect), lrq2c
     must ~= lrq2 (the forks fire but Î self-cancels since there is no
     foreclosure). Falsifier: lrq2c < lrq2 => the correction injects bias.
  H2 (the fix, decisive): on s1_stoch_sequence, lrq2c must LIFT lrq2 off its
     ~0.28 toward >=0.76 (PPO) / 0.849 (cfpk). Falsifier: no movement => the
     structural router misses the foreclosing decisions or the fork budget is
     too noisy to correct the ranking.

Run: python run_lrq2c_gate.py [workers]
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import (_set_seed, _extract_metrics, compute_baselines,  # noqa: E402
                       _threads_per_worker)
from stoch_envs import STOCH_BUILDERS  # noqa: E402
from envs import ENV_BUILDERS  # noqa: E402

# env -> (builder, length, seeds, hypothesis)
CELLS = {
    "s1_stoch_sequence":  (STOCH_BUILDERS["s1_stoch_sequence"], 20, 5, "H2"),
    "a_sequence_joint":   (ENV_BUILDERS["a_sequence_joint"],    10, 3, "H1"),
}
METHODS = ("lrq2", "lrq2c")


def _make_args(method, seed, cfg, length, logdir_base):
    a = {
        "algorithm": "ppo-clip",
        "episodes": cfg.episodes_per_epoch,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam,
        "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_rl": True,
        "causal_scheme": method,          # 'lrq2' or 'lrq2c'
        "causal_beta": cfg.causal_beta,
        # forks only for lrq2c; lrq2 is fork-free (cf_fork_prob=0 => no cf_config)
        "cf_fork_prob": (0.5 if method == "lrq2c" else 0.0),
        "cf_reps": 3, "cf_gate": 2.0, "cf_lookahead": 6.0, "cf_max_forks": 8,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": cfg.test_freq,
        "test_episodes": getattr(cfg, "test_episodes", 10),
        "save_freq": 1_000_000, "name": f"{method}__s{seed}",
        "datetag": False, "logdir": logdir_base,
    }
    return a


def train_cell(env_name, method, seed, cfg, length, logdir_base, baselines):
    _set_seed(seed)
    builder = CELLS[env_name][0]
    env = builder(causal_rl=True, allow_postpone=True, causal_postpone_tokenflow=False)
    args = _make_args(method, seed, cfg, length, logdir_base)
    saved = sys.argv
    sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=length, args_dict=args)
    finally:
        sys.argv = saved
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfg)
    m.update({"env": env_name, "method": method, "seed": seed,
              "baselines": baselines, "minutes": (time.time() - t0) / 60.0})
    return m


def _worker(payload):
    import torch
    env_name, method, seed, cfg, length, logdir, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (env_name, method, seed,
                train_cell(env_name, method, seed, cfg, length, logdir, baselines), None)
    except Exception:
        import traceback
        return (env_name, method, seed, None, traceback.format_exc())


def main(workers=3):
    cfg = stoch_config()
    cfg.epochs = 8
    cfg.episodes_per_epoch = 8
    cfg.test_freq = 2

    out = Path("suite_results_lrq2c")
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    # baselines per env (at that env's length)
    baselines = {}
    for env_name, (_, length, _, _) in CELLS.items():
        bp = out / f"baselines_{env_name}.json"
        if bp.exists():
            baselines[env_name] = json.loads(bp.read_text())
        else:
            bcfg = stoch_config(); bcfg.env_length = {env_name: length}
            baselines[env_name] = compute_baselines(env_name, bcfg)
            bp.write_text(json.dumps(baselines[env_name]))
        b = baselines[env_name]
        print(f"[lrq2c] {env_name}: random={b['random_mean']:.2f} "
              f"heuristic={b['heuristic_mean']:.2f}", flush=True)

    pending = []
    for env_name, (_, length, seeds, _) in CELLS.items():
        for method in METHODS:
            for seed in range(seeds):
                cell = out / "cells" / f"{env_name}__{method}__s{seed}.json"
                if not cell.exists():
                    pending.append((env_name, method, seed, length))
    print(f"[lrq2c] {len(pending)} cells, {workers} workers", flush=True)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    tpw = _threads_per_worker(workers)
    payloads = [(e, m, s, cfg, L, logdir, baselines[e], tpw) for (e, m, s, L) in pending]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for fut in as_completed(futs):
            env_name, method, seed, m, err = fut.result()
            if err:
                print(f"[lrq2c] !!! {env_name}/{method}_s{seed} FAILED:\n{err}", flush=True)
                continue
            (out / "cells" / f"{env_name}__{method}__s{seed}.json").write_text(json.dumps(m))
            print(f"[lrq2c] <<< {env_name}/{method}_s{seed} "
                  f"greedy_final={m.get('greedy_final')} ({m['minutes']:.1f} min; "
                  f"{(time.time()-t0)/60:.1f} total)", flush=True)

    # summary
    print("\n[lrq2c] === H1(a_joint, do-no-harm) + H2(s1, the fix) ===", flush=True)
    for env_name, (_, _, seeds, hyp) in CELLS.items():
        b = baselines[env_name]; r, h = b["random_mean"], b["heuristic_mean"]
        def norm(v): return (v - r) / (h - r) if h != r else 0.0
        print(f"  [{hyp}] {env_name} (anchor={h:.2f}):", flush=True)
        per = {}
        for method in METHODS:
            fs = []
            for seed in range(seeds):
                c = out / "cells" / f"{env_name}__{method}__s{seed}.json"
                if c.exists():
                    mm = json.loads(c.read_text())
                    if mm.get("greedy_final") is not None:
                        fs.append((seed, norm(mm["greedy_final"]), mm["greedy_final"]))
            per[method] = {s: nv for s, nv, _ in fs}
            if fs:
                nv = [x[1] for x in fs]
                print(f"      {method:<6} norm_final={np.mean(nv):.3f}+-{np.std(nv):.2f} "
                      f"raw={[round(x[2],1) for x in fs]} n={len(fs)}", flush=True)
        common = sorted(set(per.get("lrq2c", {})) & set(per.get("lrq2", {})))
        if common:
            d = [per["lrq2c"][s] - per["lrq2"][s] for s in common]
            w = sum(x > 0 for x in d)
            print(f"      PAIRED lrq2c-lrq2: mean={np.mean(d):+.3f} {w}W/{len(d)-w}L", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)