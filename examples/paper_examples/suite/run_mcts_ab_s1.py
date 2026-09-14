"""The decisive s1 A/B: lineage-attributed MCTS backup vs whole-return backup
(AEPN_NATIVE_LEARNING.md §3.3).

Standard MCTS backs up the WHOLE-trajectory return to every decision edge
(= mc_q inside the tree). Lineage backup credits each edge only by its
CAUSAL-DESCENDANT rewards (= lrq inside the tree). Everything else is identical
(same sims, horizon, conflict gate, coupling truncation, env). The only
difference is the backup restriction, so this isolates the lineage contribution
exactly — the tree-search analog of the lrq-vs-mc_q comparison (28W/0L factual).

Both arms need the causal trace, so the ENV is built causal_rl=True for both;
the AGENT stays non-causal (it consumes the search via distillation, not a
causal advantage path).

Registered predictions (§3.3):
  P1 (variance/speed): mcts_lineage converges in fewer epochs than mcts_whole
     (measured off-line: ~24% lower per-decision Q SD).
  P2 (the decisive one — finals): mcts_lineage HOLDS OR BEATS mcts_whole on s1
     finals. If it does, the tree's sibling comparison covers the foreclosure
     bias that sank cfpl, and lineage-as-headline is supported.
  FALSIFIER: mcts_lineage LOSES finals on s1 => the sibling comparison does not
     cover the bias => lineage is a variance/speed co-mechanism, not the
     headline. cfpk ref on s1 = 0.849; ppo/lcv0 band 0.72-0.76.

Cost: MCTS-with-rollout is expensive (per decision: n_sims * a rollout to the
clock horizon). Small budget on purpose; extend seeds once the sign is clear.

Run: python run_mcts_ab_s1.py [workers]
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
    "mcts_whole":   dict(lineage=False),   # rollout backup, whole return-to-go
    "mcts_lineage": dict(lineage=True),    # rollout backup, causal-descendant only
}


def _make_args(arm: str, seed: int, cfg, logdir_base: str) -> dict:
    lin = ARMS[arm]["lineage"]
    return {
        "episodes": cfg.episodes_per_epoch,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "max_episode_length": getattr(cfg, "max_episode_length", None),
        "policy_lr": cfg.policy_lr,
        "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr,
        "value_updates": cfg.value_updates,
        "gam": cfg.gam,
        "lam": cfg.lam,
        "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_rl": False,                 # AGENT non-causal (env carries the trace)
        "causal_beta": cfg.causal_beta,     # SMDP discount (project default 0.5)
        "algorithm": "mcts",
        "mcts_sims": getattr(cfg, "mcts_sims", 24),
        "mcts_lookahead": getattr(cfg, "mcts_lookahead", 8.0),
        "mcts_temp": getattr(cfg, "mcts_temp", 1.0),
        "mcts_c_puct": 1.5,
        "mcts_conflict_gate": True,
        "mcts_coupling_truncate": True,
        "mcts_rollout_backup": True,        # both arms rollout-based (fair mc_q baseline)
        "mcts_lineage_backup": lin,         # THE difference
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


def train_cell(arm: str, seed: int, cfg, logdir_base: str, baselines: dict) -> dict:
    from gympn.environment import AEPN_Env  # noqa (parity with the DCL runner)
    _set_seed(seed)
    # BOTH arms need the trace -> causal env. postpone tokenflow OFF: with it ON,
    # postpone re-emits every eligible token and becomes an ancestor of the whole
    # backlog, contaminating lineage attribution and creating a postpone
    # attractor. Off => postpone is a credit sink; the planner already credits
    # postpone edges by whole return-to-go, so it is still valued correctly.
    builder = STOCH_BUILDERS[ENV]
    env = builder(causal_rl=True, allow_postpone=True, causal_postpone_tokenflow=False)
    args = _make_args(arm, seed, cfg, logdir_base)
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
        return (arm, seed, train_cell(arm, seed, cfg, logdir_base, baselines), None)
    except Exception:
        import traceback
        return (arm, seed, None, traceback.format_exc())


def _auc(curve, r, h):
    """Mean normalized greedy return over the run = a convergence-speed proxy."""
    if not curve or h == r:
        return None
    return float(np.mean([(v - r) / (h - r) for v in curve]))


def main(workers=3):
    cfg = stoch_config()
    # Stability-contrast pass: more seeds, fewer epochs/sims, episode cap. The
    # question now is whether lineage is more STABLE than whole across seeds
    # (whole collapsed 1/3 seeds even with the postpone control), which more
    # seeds answers better than more epochs. The episode cap bounds the
    # backlog blowup that makes a collapsed (always-postpone) seed crawl.
    cfg.epochs = 4
    cfg.episodes_per_epoch = 4
    cfg.test_freq = 1
    cfg.seeds = 5
    cfg.mcts_sims = 16
    cfg.mcts_lookahead = 8.0
    cfg.max_episode_length = 30
    # Sharpened distillation target: with ~10 transient real-action nodes vs one
    # stable postpone node, temp=1 lets the GNN's argmax collapse to postpone
    # (postpone target 0.107 > per-real-action 0.09). temp=0.5 drives postpone's
    # target toward 0 so the GNN's argmax lands on a real action (measured: greedy
    # 0 -> ~11 on s1, postpone-argmax 100% -> 2%).
    cfg.mcts_temp = 0.5

    out = Path("suite_results_mcts_ab")
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    print(f"[mcts_ab] baselines for {ENV} ...", flush=True)
    bpath = out / "baselines.json"
    if bpath.exists():
        baselines = json.loads(bpath.read_text())
    else:
        baselines = compute_baselines(ENV, cfg)
        bpath.write_text(json.dumps(baselines))
    r, h = baselines["random_mean"], baselines["heuristic_mean"]
    print(f"[mcts_ab] {ENV}: random={r:.2f} heuristic={h:.2f} (cfpk ref 0.849)",
          flush=True)

    pending = [(arm, seed) for arm in ARMS for seed in range(cfg.seeds)
               if not (out / "cells" / f"{ENV}__{arm}__s{seed}.json").exists()]
    print(f"[mcts_ab] {len(pending)} cells, {workers} workers "
          f"(sims={cfg.mcts_sims} lookahead={cfg.mcts_lookahead} "
          f"epochs={cfg.epochs} eps={cfg.episodes_per_epoch} seeds={cfg.seeds})",
          flush=True)

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
                print(f"[mcts_ab] !!! {arm}_s{seed} FAILED:\n{err}", flush=True)
                continue
            (out / "cells" / f"{ENV}__{arm}__s{seed}.json").write_text(json.dumps(metrics))
            gf = metrics.get("greedy_final")
            print(f"[mcts_ab] <<< {arm}_s{seed} greedy_final={gf} "
                  f"({metrics['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)",
                  flush=True)

    # ---- summary + paired comparison -------------------------------------- #
    def load(arm):
        rows = []
        for seed in range(cfg.seeds):
            c = out / "cells" / f"{ENV}__{arm}__s{seed}.json"
            if c.exists():
                rows.append(json.loads(c.read_text()))
        return rows

    def norm(v):
        return (v - r) / (h - r)

    print(f"\n[mcts_ab] === s1 A/B (anchor={h:.2f}, cfpk ref 0.849) ===", flush=True)
    per_seed = {}
    for arm in ARMS:
        rows = load(arm)
        finals = [norm(m["greedy_final"]) for m in rows if m.get("greedy_final") is not None]
        bests = [norm(m["greedy_best"]) for m in rows if m.get("greedy_best") is not None]
        drifts = [m["greedy_drift"] for m in rows if m.get("greedy_drift") is not None]
        aucs = [a for m in rows if (a := _auc(m.get("greedy_curve"), r, h)) is not None]
        per_seed[arm] = {m["seed"]: norm(m["greedy_final"]) for m in rows
                         if m.get("greedy_final") is not None}
        if finals:
            print(f"[mcts_ab] {arm:<13} final={np.mean(finals):.3f}±{np.std(finals):.2f} "
                  f"best={np.mean(bests):.3f} drift={np.mean(drifts):.2f} "
                  f"auc={np.mean(aucs):.3f} n={len(finals)}", flush=True)

    # paired lineage - whole on finals (same seeds)
    common = sorted(set(per_seed.get("mcts_lineage", {})) & set(per_seed.get("mcts_whole", {})))
    if common:
        diffs = [per_seed["mcts_lineage"][s] - per_seed["mcts_whole"][s] for s in common]
        wins = sum(d > 0 for d in diffs)
        print(f"\n[mcts_ab] PAIRED lineage - whole (finals, n={len(common)}): "
              f"mean Δ={np.mean(diffs):+.3f}  {wins}W/{len(diffs)-wins}L", flush=True)
        print("[mcts_ab] P2 verdict: lineage >= whole on finals => tree covers the "
              "bias (headline supported); lineage < whole => falsifier "
              "(variance/speed co-mechanism only).", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)