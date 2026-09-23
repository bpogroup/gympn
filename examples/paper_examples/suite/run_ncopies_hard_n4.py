r"""Hard N-copies at N=4: the scaling benchmark with resolving power.

WHY. On the paper's N-copies family the myopic heuristic is provably the
non-idling optimum, so the normalized ceiling is exactly 1 and at N=4/N=8 every
bounded causal arm sits within 0.03 of it. The nulls between cgae_cflow and the
other bounded credit rules there are SATURATION, not equivalence (Section 6.6
says so and says the harder per-copy variant "has not been run"). It has, but
only at 15 epochs x 3 seeds, ccf vs ppo: normalized ccf 0.24 vs ppo 0.03 at N=4,
i.e. everything far below the heuristic, so the arms have room to separate.

THE ENVIRONMENT. `make_n_copies_hard`: per copy 3 task types x 2 heterogeneous
employees (employee k fast on type k, type 2 matches neither and must be
deprioritized), own arrival/queue/busy/pool, so copies stay causally
independent and K ~ N still holds. A genuine per-copy assignment rather than a
binary match/cross rule.

PROTOCOL. Identical to the runner that produced suite_results_n4_ep40 (the
paper's N=4 cells): 40 epochs x 8 episodes, greedy eval every 3 epochs on 20
episodes under common random numbers (eval_seed 555000), allow_postpone=True
with causal_postpone_tokenflow on the causal arms, seeds 0-19, shared
hyperparameters, no per-arm tuning. Anchors on the canonical no-postpone env.

ARMS, in two tiers so a complete core table lands first:
  tier 1: ppo, cgae_cflow, ccf, cgae, cgae_flow
  tier 2: cgae_cap, mc_q, cfgae, cgae_dag

WHAT CAN HAPPEN. cgae_cflow may LOSE to ccf or cgae here; the paper then has to
say so. That is the point of running it.

Run: python run_ncopies_hard_n4.py [workers] [seeds=20] [methods=a,b,...] [epochs=40]
(epochs != 40 routes to a scratch dir: smoke only). Resumable by cell file.
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
from ncopies_env import make_n_copies_hard, ncopies_hard_heuristic  # noqa: E402

N = 4
LENGTH = 20
EPOCHS = 40
EPISODES = 8
TEST_FREQ = 3
TEST_EPISODES = 20
EVAL_SEED = 555_000
SEEDS = 20
THREADS = None
OUTDIR = Path("suite_results_ncopies_hard_n4_ep40")

TIER1 = ["ppo", "cgae_cflow", "ccf", "cgae", "cgae_flow"]
TIER2 = ["cgae_cap", "mc_q", "cfgae", "cgae_dag"]
METHODS = TIER1 + TIER2
CAUSAL = {m: (m != "ppo") for m in METHODS}

for _a in sys.argv[1:]:
    if _a.startswith("seeds="):
        SEEDS = int(_a.split("=", 1)[1])
    elif _a.startswith("methods="):
        METHODS = _a.split("=", 1)[1].split(",")
    elif _a.startswith("epochs="):
        EPOCHS = int(_a.split("=", 1)[1])
        OUTDIR = Path(f"suite_results_ncopies_hard_n4_smoke{EPOCHS}")
    elif _a.startswith("threads="):        # torch intra-op threads per worker
        THREADS = int(_a.split("=", 1)[1])


def _baselines():
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    rnd, heu = [], []
    for s in range(40):
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = make_n_copies_hard(N, causal_rl=False, allow_postpone=False)
        rnd.append(float(env.testing_run(solver=RandomSolver(), length=LENGTH)))
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = make_n_copies_hard(N, causal_rl=False, allow_postpone=False)
        heu.append(float(env.testing_run(solver=HeuristicSolver(ncopies_hard_heuristic),
                                         length=LENGTH)))
    return {"random_mean": float(np.mean(rnd)), "heuristic_mean": float(np.mean(heu)),
            "random_std": float(np.std(rnd)), "heuristic_std": float(np.std(heu)),
            "episodes": 40}


def _args(method, seed, cfg, logdir):
    a = {
        "algorithm": "ppo-clip",
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": cfg.batch_size,
        "policy_lr": cfg.policy_lr, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam, "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "causal_beta": cfg.causal_beta,
        "verbose": 0, "use_gpu": False, "agent_seed": int(seed),
        "use_wandb": False, "open_tensorboard": False,
        "test_in_train": True, "test_freq": TEST_FREQ, "test_episodes": TEST_EPISODES,
        "eval_seed": EVAL_SEED,
        "save_freq": 1_000_000, "name": f"{method}__s{seed}", "datetag": False,
        "logdir": logdir,
    }
    if CAUSAL[method]:
        a.update({"causal_rl": True, "causal_scheme": method})
    else:
        a.update({"causal_rl": False, "smdp_discount": True})
    return a


def train_cell(method, seed, cfg, logdir, baselines):
    _set_seed(seed)
    env = make_n_copies_hard(N, causal_rl=CAUSAL[method], allow_postpone=True,
                             causal_postpone_tokenflow=CAUSAL[method])
    args = _args(method, seed, cfg, logdir)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LENGTH, args_dict=args)
    finally:
        sys.argv = saved
    cfx = stoch_config(); cfx.epochs = EPOCHS; cfx.test_freq = TEST_FREQ
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfx)
    m.update({"N": N, "variant": "hard", "method": method, "seed": seed,
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


def crn_precheck(episodes=20):
    """Causal and non-causal envs must consume the global random stream
    identically, or scenario alignment between arms breaks. Measured with a
    scripted policy, not assumed."""
    from gympn.agents import Agent
    from gympn.environment import AEPN_Env

    class _NoOp:
        def train(self):
            pass

    class Scripted(Agent):
        def __init__(self):
            self.policy_model = _NoOp(); self.value_model = _NoOp()
            self.best_test_metric = float('inf')

        def act(self, state, deterministic=True, return_logprob=False):
            return 0

    def build(causal):
        pn = make_n_copies_hard(N, causal_rl=causal, allow_postpone=True,
                                causal_postpone_tokenflow=causal)
        pn.length = LENGTH
        if causal:
            import types, uuid
            for place in pn.places:
                for tok in place.marking:
                    setattr(tok, '_id', str(uuid.uuid4()))
            pn.causal_trace._pn = pn
            pn.causal_trace._static_comp_cache = None
            pn.causal_trace._ls_hca_classify_cache = None
            try:
                pn.causal_trace.flush()
            except Exception:
                pass
            sent = types.SimpleNamespace(_id="__initial__")
            for place in pn.places:
                for tok in place.marking:
                    pn.causal_trace.register_token(tok, sent, parent_tokens=[], time=0)
            pn.causal_trace.register_transition(
                transition=sent, input_tokens=[],
                output_tokens=[t for p in pn.places for t in p.marking],
                is_action=False, reward=0.0, time=0)
        return AEPN_Env(pn)

    a = Scripted().test_in_train(build(False), episodes=episodes, eval_seed=EVAL_SEED)['mean_returns']
    b = Scripted().test_in_train(build(True), episodes=episodes, eval_seed=EVAL_SEED)['mean_returns']
    return float(a), float(b)


def summary(out, baselines):
    r, h = baselines["random_mean"], baselines["heuristic_mean"]
    norm = lambda v: (v - r) / (h - r)
    print(f"\n[hard-n4] anchors random={r:.2f} heuristic={h:.2f} (gap {h-r:.2f})", flush=True)
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
    (out / "summary.json").write_text(json.dumps({"baselines": baselines, "rows": rows}, indent=2))


def main(workers):
    cfg = stoch_config()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "cells").mkdir(exist_ok=True)
    logdir = str(OUTDIR / "train")

    pc = OUTDIR / "crn_precheck.json"
    if not pc.exists():
        a, b = crn_precheck()
        pc.write_text(json.dumps({"non_causal": a, "causal": b, "match": a == b}, indent=2))
        print(f"[hard-n4] CRN pre-check: non-causal {a:.4f} | causal {b:.4f} -> "
              f"{'MATCH' if a == b else 'MISMATCH (cross-arm CRN broken)'}", flush=True)

    bp = OUTDIR / f"baselines_N{N}.json"
    baselines = json.loads(bp.read_text()) if bp.exists() else _baselines()
    if not bp.exists():
        bp.write_text(json.dumps(baselines, indent=2))
    print(f"[hard-n4] anchors: random={baselines['random_mean']:.2f} "
          f"heuristic={baselines['heuristic_mean']:.2f}", flush=True)

    # tier 1 over all seeds first (seeds outer, arms inner), then tier 2
    order = [m for m in TIER1 if m in METHODS] + [m for m in TIER2 if m in METHODS] \
        + [m for m in METHODS if m not in TIER1 + TIER2]
    t1 = [m for m in order if m in TIER1]; t2 = [m for m in order if m not in TIER1]
    pending = [(m, s) for tier in (t1, t2) for s in range(SEEDS) for m in tier
               if not (OUTDIR / "cells" / f"N{N}__{m}__s{s}.json").exists()]
    print(f"[hard-n4] {len(pending)} cells ({len(METHODS)} arms x {SEEDS} seeds, {EPOCHS} epochs), "
          f"{workers} workers", flush=True)

    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = THREADS if THREADS else _threads_per_worker(workers)
        print(f"[hard-n4] torch threads per worker: {tpw}", flush=True)
        r, h = baselines["random_mean"], baselines["heuristic_mean"]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(_worker, (m, s, cfg, logdir, baselines, tpw)) for (m, s) in pending]
            for fut in as_completed(futs):
                m, s, res, err = fut.result()
                if err:
                    print(f"[hard-n4] !!! {m}_s{s} FAILED:\n{err}", flush=True)
                    continue
                (OUTDIR / "cells" / f"N{N}__{m}__s{s}.json").write_text(json.dumps(res))
                gf = res.get("greedy_final")
                nrm = (gf - r) / (h - r) if gf is not None else float("nan")
                print(f"[hard-n4] <<< {m:<11} s{s:<2} final={nrm:+.3f} "
                      f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    summary(OUTDIR, baselines)


if __name__ == "__main__":
    w = next((int(a) for a in sys.argv[1:] if a.isdigit()), 4)
    main(w)
