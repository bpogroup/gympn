r"""BPM decision-type benchmark runner (conference version of the paper).

Trains the arms on one of the two new BPM environments in `bpm_envs.py`
(next-activity selection, rework / quality gate) at a chosen number of
independent copies N, under the SAME protocol as the paper's 40-epoch cells
(run_ncopies_hard_n4.py / run_n2_epochs.py): 40 epochs x 8 episodes, greedy
eval every 3 epochs on 20 episodes under common random numbers (eval_seed
555000), allow_postpone=True with causal_postpone_tokenflow on the causal
arms, seeds 0-19, shared hyperparameters, no per-arm tuning. Anchors on the
canonical no-postpone env over 40 episodes.

Arms (the conference paper's set): ppo, cgae_cflow, mc_q. Any other suite
arm name is accepted for ad-hoc checks.

Run:  python run_bpm.py env=next_activity N=8 [workers] [seeds=20]
                        [methods=ppo,cgae_cflow,mc_q] [epochs=40] [threads=k]
                        [beta=0.5]   (causal wall-clock discount; routes to *_beta<val>)
                        [postpone=0] (no postpone; routes to *_nopp)
                        [postpone=component] (one postpone per net component, the
                                     scope nfgae's Theorem 1 allows; routes to *_ppc)
Arm `nfgae` (net-factored GAE, paper/NFGAE_THEORY.md) runs on the plain SMDP-GAE
path with no causal trace; it needs postpone=0.
Output: suite_results_bpm_<env>_n<N>_ep<epochs>/cells/N<N>__<method>__s<seed>.json
(epochs != 40 routes to a *_smoke<epochs> dir). Resumable by cell file.
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
from bpm_envs import BPM_BUILDERS, BPM_HEURISTICS  # noqa: E402
from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS  # noqa: E402

BPM_BUILDERS = {**BPM_BUILDERS, **INSURER_BUILDERS}
BPM_HEURISTICS = {**BPM_HEURISTICS, **INSURER_HEURISTICS}

ENV = "next_activity"
N = 8
LENGTH = 20
EPOCHS = 40
EPISODES = 8
TEST_FREQ = 3
TEST_EPISODES = 20
EVAL_SEED = 555_000
SEEDS = 20
THREADS = None
BETA = None                    # causal wall-clock discount override (cfg.causal_beta if None)
METHODS = ["ppo", "cgae_cflow", "mc_q"]
POSTPONE = True
_POSTPONE_SET = False
POSTPONE_SCOPE = "global"
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
              + ("_flat" if FLAT else ""))
# nfgae needs no causal trace: it runs on the plain SMDP-GAE path.
CAUSAL = {m: (m not in ("ppo", "nfgae")) for m in METHODS}
TAG = f"bpm-{ENV}-n{N}"


def _baselines():
    from gympn.solvers import RandomSolver, HeuristicSolver
    import random
    rnd, heu = [], []
    for s in range(40):
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = BUILD(N, causal_rl=False, allow_postpone=False)
        rnd.append(float(env.testing_run(solver=RandomSolver(), length=LENGTH)))
        random.seed(1000 + s); np.random.seed(1000 + s)
        env = BUILD(N, causal_rl=False, allow_postpone=False)
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
        "causal_beta": cfg.causal_beta,
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
    if CAUSAL[method]:
        a.update({"causal_rl": True, "causal_scheme": method})
    elif method == "nfgae":
        a.update({"causal_rl": False, "causal_scheme": "nfgae", "smdp_discount": True})
    else:
        a.update({"causal_rl": False, "smdp_discount": True})
    return a


def train_cell(method, seed, cfg, logdir, baselines):
    _set_seed(seed)
    env = BUILD(N, causal_rl=CAUSAL[method], allow_postpone=POSTPONE,
                causal_postpone_tokenflow=CAUSAL[method] and POSTPONE)
    env.postpone_scope = POSTPONE_SCOPE
    args = _args(method, seed, cfg, logdir)
    saved = sys.argv; sys.argv = sys.argv[:1]
    t0 = time.time()
    try:
        env.training_run(length=LENGTH, args_dict=args)
    finally:
        sys.argv = saved
    cfx = stoch_config(); cfx.epochs = EPOCHS; cfx.test_freq = TEST_FREQ
    m = _extract_metrics(getattr(env, "training_history", {}) or {}, cfx)
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
        pn = BUILD(N, causal_rl=causal, allow_postpone=POSTPONE,
                   causal_postpone_tokenflow=causal and POSTPONE)
        pn.postpone_scope = POSTPONE_SCOPE
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
    cfg = stoch_config()
    if BETA is not None:
        cfg.causal_beta = BETA
        print(f"[{TAG}] causal_beta override: {BETA}", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "cells").mkdir(exist_ok=True)
    logdir = str(OUTDIR / "train")

    pc = OUTDIR / "crn_precheck.json"
    if not pc.exists():
        a, b = crn_precheck()
        pc.write_text(json.dumps({"non_causal": a, "causal": b, "match": a == b}, indent=2))
        print(f"[{TAG}] CRN pre-check: non-causal {a:.4f} | causal {b:.4f} -> "
              f"{'MATCH' if a == b else 'MISMATCH (cross-arm CRN broken)'}", flush=True)

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
        tpw = THREADS if THREADS else _threads_per_worker(workers)
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
