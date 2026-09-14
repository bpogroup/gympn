"""Credit-spectrum experiment (Experiments 1 & 2 of PAPER_PLAN.md).

Runs the four schemes {ppo, mc_q, lrq, ccf} on two contrasting environments and
turns Table 1 (the asserted bias/variance spectrum) into data:

  * s1_stoch_sequence -- SINGLE causal component (shared 3-employee pool) with
    foreclosure/opportunity-cost. Predicted: ppo == mc_q == ccf (Corollary 1
    equality; the null), while lrq is BIASED (drops sibling-lineage opportunity
    cost) and underperforms -> validates unbiasedness (Prop 1).
  * ncopies (N=4) -- MULTI component (independent copies), no foreclosure.
    Predicted: ccf == lrq > ppo == mc_q -> the variance win (Prop 2); lrq is fine
    here because there is no opportunity cost to drop.

Run: python run_spectrum.py [workers]
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
from envs import make_env, perfect_heuristic, HEURISTICS  # noqa: E402
from ncopies_env import make_n_copies, ncopies_heuristic  # noqa: E402
from foreclosure_env import make_foreclosure, perfect_heuristic as foreclosure_heuristic  # noqa: E402
from join_env import make_join, make_join_pairs, optimal_heuristic as join_heuristic  # noqa: E402

SEEDS = 8
EPOCHS = 25
EPISODES = 8
METHODS = ("ppo", "mc_q", "lrq", "ccf", "s_ccf")

# env registry: name -> (make(causal_rl, allow_postpone), heuristic, length, tag)
ENVS = {
    "s1": dict(
        make=lambda causal_rl: make_env("s1_stoch_sequence", causal_rl=causal_rl,
                                         allow_postpone=False),
        heuristic=HEURISTICS.get("s1_stoch_sequence", perfect_heuristic),
        length=20, tag="single component (foreclosure)"),
    "ncopies4": dict(
        make=lambda causal_rl: make_n_copies(4, causal_rl=causal_rl,
                                             allow_postpone=False),
        heuristic=ncopies_heuristic,
        length=20, tag="multi component (independent)"),
    "foreclosure": dict(
        # Optimum REQUIRES postpone (hold the one-shot specialist for the high
        # task); lineage credit is documented to reinforce the +3 trap. This is
        # where lineage's foreclosure bias should hurt while the value baseline
        # (ppo, and ccf via its SMDP-TD postpone credit) can price the cost.
        # lrq v1 refuses postpone envs by design (degenerate postpone credit);
        # we use lrq2 (postpone-safe; identical to lrq where there is no postpone).
        # mc_q dropped here: a network-shape incompatibility with the postpone
        # representation, unrelated to credit assignment (mc_q ~= ppo anyway).
        make=lambda causal_rl: make_foreclosure(causal_rl=causal_rl, allow_postpone=True),
        heuristic=foreclosure_heuristic,
        length=8, tag="foreclosure (opportunity cost; needs postpone)",
        methods=("ppo", "lrq2", "ccf"), epochs=40),
    "join": dict(
        # Pure AND-join: only ccf should FLIP to the suboptimal join (its realized
        # component includes r2 only when it joins); lrq/mc_q/ppo/s_ccf stay correct.
        make=lambda causal_rl: make_join(causal_rl=causal_rl, allow_postpone=False),
        heuristic=join_heuristic,
        length=20, tag="AND-join (ccf-bias capstone)"),
    "joinbal": dict(
        # Streaming AND-join with BALANCED small rewards (r_join=1,r1=2,r2=3): the
        # decision is now ~20% of the return (learnable), while still r2 > r1-r_join
        # so ccf flips. Streaming staggers decisions in time (avoids the all-zero-
        # sojourn SMDP issue of the fixed-pairs variant). The clean capstone: ccf
        # should learn the trap (join), s_ccf/lrq/mc_q/ppo the optimum (standalone).
        make=lambda causal_rl: make_join(causal_rl=causal_rl, allow_postpone=False,
                                         r_join=1.0, r1=2.0, r2=3.0),
        heuristic=join_heuristic,
        length=20, tag="AND-join balanced (learnable ccf-bias)"),
}


def _methods_for(env_key):
    return ENVS[env_key].get("methods", METHODS)


def _baselines(env_key):
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    spec = ENVS[env_key]; L = spec["length"]
    rnd, heu = [], []
    for s in range(20):
        random.seed(5000 + s); np.random.seed(5000 + s)
        rnd.append(float(spec["make"](False).testing_run(solver=RandomSolver(), length=L)))
        random.seed(5000 + s); np.random.seed(5000 + s)
        heu.append(float(spec["make"](False).testing_run(
            solver=HeuristicSolver(spec["heuristic"]), length=L)))
    return {"random_mean": float(np.mean(rnd)), "heuristic_mean": float(np.mean(heu))}


def _args(env_key, method, seed, cfg, logdir):
    epochs = ENVS[env_key].get("epochs", cfg.epochs)
    a = {
        "algorithm": "ppo-clip",
        "episodes": cfg.episodes_per_epoch, "epochs": epochs, "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam, "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None), "causal_beta": cfg.causal_beta,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": cfg.test_freq, "test_episodes": 12,
        "save_freq": 10**9, "name": f"{env_key}__{method}__s{seed}",
        "datetag": False, "logdir": logdir,
    }
    if method == "ppo":
        a.update({"causal_rl": False, "smdp_discount": True})
    else:  # mc_q / lrq / ccf all use the causal_rl redistribution path
        a.update({"causal_rl": True, "causal_scheme": method})
    return a


def train_cell(env_key, method, seed, cfg, logdir, baselines):
    _set_seed(seed)
    env = ENVS[env_key]["make"](method != "ppo")
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=ENVS[env_key]["length"], args_dict=_args(env_key, method, seed, cfg, logdir))
    finally:
        sys.argv = saved
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfg)
    m.update({"env": env_key, "method": method, "seed": seed, "baselines": baselines,
              "minutes": (time.time() - t0) / 60.0})
    return m


def _worker(payload):
    import torch
    env_key, method, seed, cfg, logdir, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (env_key, method, seed, train_cell(env_key, method, seed, cfg, logdir, baselines), None)
    except Exception:
        import traceback
        return (env_key, method, seed, None, traceback.format_exc())


def _summary(out, baselines):
    from scipy import stats
    print("\n[spectrum] === normalized final return (0=random, 1=heuristic) ===", flush=True)
    for env_key in ENVS:
        b = baselines[env_key]; r, h = b["random_mean"], b["heuristic_mean"]
        def norm(v): return (v - r) / (h - r) if h != r else 0.0
        print(f"\n  {env_key}  [{ENVS[env_key]['tag']}]  random={r:.2f} heuristic={h:.2f}", flush=True)
        finals = {}
        for m in _methods_for(env_key):
            vals = []
            for s in range(SEEDS):
                c = out / "cells" / f"{env_key}__{m}__s{s}.json"
                if c.exists():
                    mm = json.loads(c.read_text())
                    if mm.get("greedy_final") is not None:
                        vals.append(norm(mm["greedy_final"]))
            vals = np.array(vals); finals[m] = vals
            n = len(vals)
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            mean = vals.mean() if n else float("nan")
            print(f"    {m:>5}: {mean:.3f} +/- {ci:.3f}  (n={n})", flush=True)
        # key contrasts: ccf vs every other scheme present
        for other in [m for m in _methods_for(env_key) if m != "ccf"]:
            if len(finals.get("ccf", [])) > 1 and len(finals.get(other, [])) > 1:
                t, p = stats.ttest_ind(finals["ccf"], finals[other], equal_var=False)
                print(f"    -> ccf vs {other}: gap={finals['ccf'].mean()-finals[other].mean():+.3f} (p={p:.3f})", flush=True)
    (out / "summary.json").write_text(json.dumps({"baselines": baselines}, indent=2))


def main(workers=6):
    cfg = stoch_config()
    cfg.epochs = EPOCHS; cfg.episodes_per_epoch = EPISODES; cfg.test_freq = 2

    out = Path("suite_results_spectrum")
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    baselines = {}
    for env_key in ENVS:
        bp = out / f"baselines_{env_key}.json"
        baselines[env_key] = json.loads(bp.read_text()) if bp.exists() else _baselines(env_key)
        if not bp.exists():
            bp.write_text(json.dumps(baselines[env_key]))
        b = baselines[env_key]
        print(f"[spectrum] {env_key}: random={b['random_mean']:.2f} heuristic={b['heuristic_mean']:.2f}", flush=True)

    pending = [(e, m, s) for e in ENVS for m in _methods_for(e) for s in range(SEEDS)
               if not (out / "cells" / f"{e}__{m}__s{s}.json").exists()]
    print(f"[spectrum] {len(pending)} cells, {workers} workers", flush=True)

    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(workers)
        payloads = [(e, m, s, cfg, logdir, baselines[e], tpw) for (e, m, s) in pending]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                e, m, s, res, err = fut.result()
                if err:
                    print(f"[spectrum] !!! {e}/{m}_s{s} FAILED:\n{err}", flush=True)
                    continue
                (out / "cells" / f"{e}__{m}__s{s}.json").write_text(json.dumps(res))
                print(f"[spectrum] <<< {e}/{m}_s{s} final={res.get('greedy_final')} "
                      f"({res['minutes']:.1f}min; {(time.time()-t0)/60:.1f} total)", flush=True)

    _summary(out, baselines)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6)