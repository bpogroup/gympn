r"""ppo vs ccf vs cgae vs cfgae on n-copies, N in {1,2,4}, 5 seeds, CRN eval.

s1 was the wrong env to price these schemes on: ppo already sits at 14.0-14.6
against an optimum of 14.78, and under common random numbers all three arms
came out indistinguishable (every p >= 0.45). N-copies is the env where
component-factored credit is SUPPOSED to bite -- N independent copies means
each decision's reward is contaminated by N-1 other copies' noise, and the
stored 12-seed sweep shows ccf's normalized-final gap over ppo growing with N:

    N       1       2       4       8
    gap  +0.029  +0.335  +0.274  +0.569

ccf is included here as a POSITIVE CONTROL, not for its own sake. It is the
known winner on this env; if it also comes out null under this protocol then
the protocol is eating real effects and a cgae/cfgae null says nothing.

N=1 is the other control, and it is the sharp one for cfgae: with a single
component the reward mask is identically one and cfgae's recursion is
term-for-term PPO's, so cfgae@N=1 must land on ppo@N=1. If it does not, the
component partition is doing something it should not.

N=8 is omitted deliberately: those cells cost 33-64 min EACH (stored sweep
timings), i.e. ~5-6 h for this design, versus ~2.5 h for N<=4. The gap is
already decisive at N=2/4.

POSTPONE. Trained with allow_postpone=True, matching s1 and run_suite -- the
stored ncopies sweep trains postpone-OFF, so these cells are NOT comparable to
it cell-for-cell; ccf is re-run here to re-anchor the control under this
setting. Postpone-off was rejected because run_evolutions' single-binding
branch keys on causal_rl when postpone is off, which would let the causal arms
face a different decision sequence from the ppo baseline. That turned out to be
harmless on this env (measured: zero single-binding action states at N=1,2,4
under fixed and random policies) but only by luck; with postpone on the branch
is bypassed entirely and the structures are identical by construction.

Baselines stay on the canonical no-postpone env, as in run_suite.compute_
baselines -- they are only the normalization anchors (random .. heuristic).

CRN pre-check: the causal and non-causal envs must consume the global `random`
stream identically or scenario alignment BETWEEN arms breaks. Measured with a
scripted policy rather than assumed, and recorded in crn_precheck.json.

Run: python run_ncopies_three_way_crn.py [workers]
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
from ncopies_env import make_n_copies, ncopies_heuristic  # noqa: E402

NS = [1, 2, 4]
SEEDS = 5
LENGTH = 20
METHODS = ("ppo", "ccf", "cgae", "cfgae", "cgae_flow")
CAUSAL = {"ppo": False, "ccf": True, "cgae": True, "cfgae": True,
          "cgae_flow": True}
EVAL_SEED = 555_000
OUTDIR = "suite_results_ncopies_3way_crn"


def _baselines(n):
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    rnd, heu = [], []
    for s in range(15):
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = make_n_copies(n, causal_rl=False, allow_postpone=False)
        rnd.append(float(env.testing_run(solver=RandomSolver(), length=LENGTH)))
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = make_n_copies(n, causal_rl=False, allow_postpone=False)
        heu.append(float(env.testing_run(solver=HeuristicSolver(ncopies_heuristic), length=LENGTH)))
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
        "eval_seed": EVAL_SEED,          # common random numbers
        "save_freq": 1_000_000, "name": f"{method}__s{seed}",
        "datetag": False, "logdir": logdir,
    }
    if CAUSAL[method]:
        a.update({"causal_rl": True, "causal_scheme": method})
    else:  # ppo: discount-matched baseline (SMDP-GAE at the same beta)
        a.update({"causal_rl": False, "smdp_discount": True})
    return a


def train_cell(n, method, seed, cfg, logdir, baselines):
    _set_seed(seed)
    # allow_postpone=True, matching s1 / run_suite.train_cell -- NOT the stored
    # ncopies sweep, which trains postpone-off. With postpone on, the agent
    # gets a genuine defer action and run_evolutions returns to it
    # unconditionally, so the decision structure cannot depend on causal_rl
    # (measured on s1: identical action-set histograms, 360 calls, in both
    # modes). Postpone-off relies on there happening to be no single-binding
    # action states -- true on this env, but by luck rather than construction.
    # causal_postpone_tokenflow mirrors run_suite: on for the causal arms, so
    # postpone is a lineage member rather than a credit sink.
    env = make_n_copies(n, causal_rl=CAUSAL[method], allow_postpone=True,
                        causal_postpone_tokenflow=CAUSAL[method])
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
    n, method, seed, cfg, logdir, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (n, method, seed, train_cell(n, method, seed, cfg, logdir, baselines), None)
    except Exception:
        import traceback
        return (n, method, seed, None, traceback.format_exc())


# --------------------------------------------------------------------------
# CRN pre-check: do the causal and non-causal envs see the SAME scenarios?
# --------------------------------------------------------------------------
def crn_precheck(n, episodes=20):
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
            n_a = len(state['actions_dict']) if isinstance(state, dict) and 'actions_dict' in state else 1
            return 0 if n_a else 0

    def build(causal):
        pn = make_n_copies(n, causal_rl=causal, allow_postpone=True,
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


def main(workers=3):
    global SEEDS, NS
    cfg = stoch_config()
    cfg.epochs = 15
    cfg.episodes_per_epoch = 8
    cfg.test_freq = 3

    out = Path(OUTDIR)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    print(f"[3way-nc] CRN pre-check (allow_postpone=True, as s1), "
          f"eval_seed={EVAL_SEED}", flush=True)
    precheck = {}
    for n in NS:
        a, b = crn_precheck(n)
        ok = (a == b)
        precheck[n] = {"non_causal": a, "causal": b, "match": ok}
        print(f"  N={n}: non-causal {a:.4f} | causal {b:.4f} -> "
              f"{'MATCH' if ok else 'MISMATCH (cross-arm CRN broken)'}", flush=True)
    (out / "crn_precheck.json").write_text(json.dumps(precheck, indent=2))

    baselines = {}
    for n in NS:
        bp = out / f"baselines_N{n}.json"
        if bp.exists():
            baselines[n] = json.loads(bp.read_text())
        else:
            baselines[n] = _baselines(n)
            bp.write_text(json.dumps(baselines[n]))
        b = baselines[n]
        print(f"[3way-nc] N={n}: random={b['random_mean']:.2f} "
              f"heuristic={b['heuristic_mean']:.2f}", flush=True)

    pending = [(n, m, s) for n in NS for m in METHODS for s in range(SEEDS)
               if not (out / "cells" / f"N{n}__{m}__s{s}.json").exists()]
    print(f"[3way-nc] {len(pending)} cells, {workers} workers", flush=True)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    tpw = _threads_per_worker(workers)
    payloads = [(n, m, s, cfg, logdir, baselines[n], tpw) for (n, m, s) in pending]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(_worker, p) for p in payloads]
        for fut in as_completed(futs):
            n, m, s, res, err = fut.result()
            if err:
                print(f"[3way-nc] !!! N{n}/{m}_s{s} FAILED:\n{err}", flush=True)
                continue
            (out / "cells" / f"N{n}__{m}__s{s}.json").write_text(json.dumps(res))
            print(f"[3way-nc] <<< N{n}/{m}_s{s} greedy_final={res.get('greedy_final')} "
                  f"({res['minutes']:.1f} min; {(time.time()-t0)/60:.1f} total)", flush=True)

    print("\n[3way-nc] === normalized final vs N ===", flush=True)
    hdr = "  N |" + "".join(f" {m:>15} |" for m in METHODS)
    print(hdr, flush=True)
    for n in NS:
        b = baselines[n]; r, h = b["random_mean"], b["heuristic_mean"]
        def norm(v): return (v - r) / (h - r) if h != r else 0.0
        line = f"  {n} |"
        for m in METHODS:
            fs = []
            for s in range(SEEDS):
                c = out / "cells" / f"N{n}__{m}__s{s}.json"
                if c.exists():
                    mm = json.loads(c.read_text())
                    if mm.get("greedy_final") is not None:
                        fs.append(norm(mm["greedy_final"]))
            line += f" {np.mean(fs):>6.3f}+-{np.std(fs):<5.2f} |" if fs else "      n/a       |"
        print(line, flush=True)


if __name__ == "__main__":
    # CLI: [workers] [seeds=N] [ns=2,4]
    #   python run_ncopies_three_way_crn.py 3 seeds=20 ns=2,4
    # Resumable by cell file, so raising `seeds` re-runs only the new ones.
    for _a in sys.argv[1:]:
        if _a.startswith("seeds="):
            SEEDS = int(_a.split("=", 1)[1])
        elif _a.startswith("ns="):
            NS = [int(x) for x in _a.split("=", 1)[1].split(",") if x]
    main(next((int(a) for a in sys.argv[1:] if a.isdigit()), 3))
