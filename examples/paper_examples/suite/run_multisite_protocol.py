r"""Multi-site rerun under the protocol the paper states.

WHY. The paper's multi-site cells (suite_results_multisite_bf) were produced
with 8 episodes per epoch, 12 greedy-evaluation episodes and NO evaluation seed,
while Table 8 / Appendix E state 20 episodes per epoch, greedy evaluation every
second epoch on 20 episodes, and common random numbers for every evaluation
point. The PPO tuning grid showed the evaluation draw alone moves PPO's
multi-site number from 0.161 to 0.279, so the headline environment must be
re-run under the stated protocol rather than described around.

PROTOCOL (matches Table 8 and the other converged cells):
  40 epochs x 20 episodes, greedy eval every 2 epochs on 20 episodes,
  eval_seed 555000 (scenario i = eval_seed + i for every arm/seed/epoch),
  seeds 0-19, shared hyperparameters, no per-arm tuning.
  allow_postpone=False, as in every previous multi-site cell and as the paper's
  optimum argument for multi-site requires (Section 6.6): the multi-site
  ceiling 1.029 is the NON-idling optimum. Anchors are the stored ones
  (suite_results_multisite_bf/baselines.json: random 62.90, heuristic 80.25),
  computed on the same no-postpone net.

ARMS, two tiers. Tier 1 are the distinct estimators; tier 2 (cgae, cgae_flow)
are provably bit-identical to cgae_cflow on multi-site (fan-out exactly 1) and
are run last, only so the identity can be re-verified on the new cells.
  tier 1: ppo, cgae_cflow, ccf, cgae_cap, mc_q
  tier 2: cgae, cgae_flow

Run: python run_multisite_protocol.py [workers] [seeds=20] [methods=a,b,...] [epochs=40]
(epochs != 40 routes to a scratch dir: smoke only). Resumable by cell file.
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
LENGTH = 20
EPOCHS = 40
EPISODES = 20
TEST_FREQ = 2
TEST_EPISODES = 20
EVAL_SEED = 555_000
SEEDS = 20
OUTDIR = Path("suite_results_multisite_protocol")
SRC_BASELINES = Path("suite_results_multisite_bf") / "baselines.json"

TIER1 = ["ppo", "cgae_cflow", "ccf", "cgae_cap", "mc_q"]
TIER2 = ["cgae", "cgae_flow"]
METHODS = TIER1 + TIER2

for _a in sys.argv[1:]:
    if _a.startswith("seeds="):
        SEEDS = int(_a.split("=", 1)[1])
    elif _a.startswith("methods="):
        METHODS = _a.split("=", 1)[1].split(",")
    elif _a.startswith("epochs="):
        EPOCHS = int(_a.split("=", 1)[1])
        OUTDIR = Path(f"suite_results_multisite_protocol_smoke{EPOCHS}")
CAUSAL = {m: (m != "ppo") for m in METHODS}   # after argv: methods= may add arms


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
    env = make_multisite(N_SITES, N_LOCAL, N_FLEX, causal_rl=CAUSAL[method],
                         allow_postpone=False)
    args = _args(method, seed, cfg, logdir)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LENGTH, args_dict=args)
    finally:
        sys.argv = saved
    cfx = stoch_config(); cfx.epochs = EPOCHS; cfx.test_freq = TEST_FREQ
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfx)
    m.update({"env": "multisite", "n_sites": N_SITES, "n_local": N_LOCAL, "n_flex": N_FLEX,
              "allow_postpone": False, "method": method, "seed": seed,
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


def crn_precheck(episodes=20):
    """Causal and non-causal envs must consume the global random stream
    identically, or scenario alignment between arms breaks."""
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
        pn = make_multisite(N_SITES, N_LOCAL, N_FLEX, causal_rl=causal, allow_postpone=False)
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
    print(f"\n[multisite-protocol] anchors random={r:.2f} heuristic={h:.2f} (gap {h-r:.2f}); "
          f"paper (old protocol): cgae_cflow 0.673 [0], ccf 0.511 [3], mc_q 0.030 [17], PPO 0.161 [14]",
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
    cfg = stoch_config()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "cells").mkdir(exist_ok=True)
    logdir = str(OUTDIR / "train")

    bp = OUTDIR / "baselines.json"
    if not bp.exists():
        shutil.copy(SRC_BASELINES, bp)
    baselines = json.loads(bp.read_text())

    pc = OUTDIR / "crn_precheck.json"
    if not pc.exists():
        a, b = crn_precheck()
        verdict = "MATCH" if abs(a - b) < 1e-9 else "MISMATCH"
        print(f"[multisite-protocol] CRN pre-check: non-causal {a:.4f} | causal {b:.4f} -> {verdict}", flush=True)
        pc.write_text(json.dumps({"non_causal": a, "causal": b, "verdict": verdict}))
        if verdict != "MATCH":
            print("[multisite-protocol] !!! scenario alignment between arms is broken; aborting", flush=True)
            return
    print(f"[multisite-protocol] anchors: random={baselines['random_mean']:.2f} "
          f"heuristic={baselines['heuristic_mean']:.2f}", flush=True)

    order = [m for m in TIER1 if m in METHODS] + [m for m in TIER2 if m in METHODS] \
        + [m for m in METHODS if m not in TIER1 + TIER2]
    t1 = [m for m in order if m in TIER1]; t2 = [m for m in order if m not in TIER1]
    # seeds outer, arms inner within a tier: partial results stay balanced across arms
    pending = [(m, s) for tier in (t1, t2) for s in range(SEEDS) for m in tier
               if not (OUTDIR / "cells" / f"{m}__s{s}.json").exists()]
    tpw = _threads_per_worker(workers)
    print(f"[multisite-protocol] {len(pending)} cells ({len(METHODS)} arms x {SEEDS} seeds, "
          f"{EPOCHS} epochs x {EPISODES} episodes, eval every {TEST_FREQ} on {TEST_EPISODES}), "
          f"{workers} workers x {tpw} threads, eval_seed={EVAL_SEED}", flush=True)

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
