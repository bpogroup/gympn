r"""BPM experiment runner: PPO vs NF-GAE on the insurer and the BPM nets.

Protocol: 40 epochs x 8 episodes, greedy evaluation every 3 epochs on 20
episodes under common random numbers (eval_seed 555000), seeds 0-19, shared
hyperparameters (common.Hyper), no per-arm tuning. Anchors: random and
heuristic policies over 40 episodes (insurer_slow: the waiting heuristic).

Arms: ppo (SMDP-GAE PPO, one agent), nfgae (net-factored GAE, one agent,
../paper/NFGAE_THEORY.md) and agents (one PPO agent per net component, in
component turns, credited with its component's return; gympn.component_agents).

Run:  python scripts/run_bpm.py env=insurer N=2 [workers] [seeds=20] [methods=ppo,nfgae]
                        [epochs=40] [threads=k] [episodes=8] [plr=x] [lam=x] [bs=k] [init=<policy.pth>] [cwarm=k] [rb=k] [cert=static|marking]
                        [net=aepn|hgt|temb|tembf] [bases=K] [flat=1] [local=1]
                        [beta=0.5]   (SMDP discount, default 0.05; any value but the legacy 0.5 routes to *_beta<val>)
                        [view=full]  (default for aepn: reverse edges + per-layer global attention +
                                      count-aware aggregation, the lossless whole-net view; *_fv)
                        [view=local]  (forward edges only, the pre-2026-10-17 default and the
                                       unsuffixed result dirs; default for non-aepn nets)
                        [gpu=1]      (PPO updates on the CUDA GPU, rollouts on the CPU; same
                                      results up to float rounding, so no directory suffix)
                        [critic_mask=1]  (view=full: the critic's attention also stays inside
                                          each component, as the actor's does; *_cm)
                        [postpone=0|1|component]  (no / global / per-component
                                     postpone; *_nopp, unsuffixed, *_ppc)
The insurer and bpic15 envs default to postpone=0. nfgae needs postpone=0 or component.
Output: results/bpm_<env>_n<N>_ep<epochs>.../cells/N<N>__<method>__s<seed>.json
(epochs != 40 routes to a *_smoke<epochs> dir), relative to the suite directory
whatever the working directory is. Resumable by cell file. See ../REPRODUCE.md
for the exact commands behind the paper's result folders.
"""
import json
import os
import sys
import time
from pathlib import Path

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SUITE, "envs"))

import numpy as np  # noqa: E402

from common import Hyper, set_seed, extract_metrics, threads_per_worker, beta_tag  # noqa: E402
from bpm_envs import BPM_BUILDERS, BPM_HEURISTICS  # noqa: E402
from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS  # noqa: E402
from bpic15_env import BPIC15_BUILDERS, BPIC15_HEURISTICS  # noqa: E402

BPM_BUILDERS = {**BPM_BUILDERS, **INSURER_BUILDERS, **BPIC15_BUILDERS}
BPM_HEURISTICS = {**BPM_HEURISTICS, **INSURER_HEURISTICS, **BPIC15_HEURISTICS}

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
LAM = None                     # lam=x: GAE lambda override (default Hyper.lam = 0.95; routes to *_lam<x>)
INIT = None                    # init=<path>: start the policy from a saved (torch.save) policy (routes to *_init<stem>)
CWARM = 0                      # cwarm=k: first k epochs train only the critic (routes to *_cw<k>)
RB = 1                         # rb=k: collect k episodes side by side, batched forward passes (routes to *_rb<k>)
FLAT = False                   # flat=1: flat graph observations (needs net=temb|tembf|aepn; routes to *_flat)
NET = "aepn"                   # net=hgt|temb|tembf|aepn: actor+critic encoder (default aepn; *_aepn dirs, hgt = the old unsuffixed dirs)
BASES = None                   # bases=K: AEPNStack basis count (default 8; routes to *_b<K>)
BS = None                      # bs=k: minibatch size override (default Hyper.batch_size; routes to *_bs<k>)
MARKING = False                # cert=marking: nfgae recomputes the certificate at the current marking (routes to *_mc)
VIEW = None                    # view=local|full (*'' / *_fv); default full for aepn (the library default), local otherwise
AGENT_BATCH = "share"          # abatch=full: component agents each get the full batch size (*_abfull)
GPU = False                    # gpu=1: PPO updates on the CUDA GPU (Agent.update_device)
CRITIC_MASK = False            # critic_mask=1: view=full critic attention also stays inside components (*_cm)

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
    elif _a.startswith("lam="):
        LAM = float(_a.split("=", 1)[1])
    elif _a.startswith("init="):
        INIT = os.path.abspath(_a.split("=", 1)[1])
    elif _a.startswith("cwarm="):
        CWARM = int(_a.split("=", 1)[1])
    elif _a.startswith("rb="):
        RB = int(_a.split("=", 1)[1])
    elif _a.startswith("flat="):
        FLAT = _a.split("=", 1)[1] not in ("0", "false", "False")
    elif _a.startswith("abatch="):
        AGENT_BATCH = _a.split("=", 1)[1]
        if AGENT_BATCH not in ("share", "full"):
            raise SystemExit(f"abatch must be share or full, got {AGENT_BATCH!r}")
    elif _a.startswith("cert="):
        _v = _a.split("=", 1)[1]
        if _v not in ("static", "marking"):
            raise SystemExit(f"cert must be static or marking, got {_v!r}")
        MARKING = _v == "marking"
    elif _a.startswith("view="):
        _v = _a.split("=", 1)[1]
        if _v not in ("local", "full"):
            raise SystemExit(f"view must be local or full, got {_v!r}")
        VIEW = _v
    elif _a.startswith("gpu="):
        GPU = _a.split("=", 1)[1] in ("1", "true", "True")
    elif _a.startswith("critic_mask="):
        CRITIC_MASK = _a.split("=", 1)[1] in ("1", "true", "True")
    elif _a.startswith("bs="):
        BS = int(_a.split("=", 1)[1])
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

# The insurer and BPIC 2015 envs are work-conserving by design: no postpone unless asked.
if ENV.startswith(("insurer", "bpic15")) and not _POSTPONE_SET:
    POSTPONE = False
if VIEW is None:
    VIEW = "full" if NET == "aepn" else "local"
if set(METHODS) - {"ppo", "nfgae", "agents"}:
    raise SystemExit(f"unknown methods {sorted(set(METHODS) - {'ppo', 'nfgae', 'agents'})}; "
                     "choose from ppo, nfgae, agents")
if ENV not in BPM_BUILDERS:
    raise SystemExit(f"unknown env {ENV!r}; choose from {sorted(BPM_BUILDERS)}")
BUILD = BPM_BUILDERS[ENV]
HEUR = BPM_HEURISTICS[ENV]
RESULTS = Path(SUITE) / "results"
OUTDIR = RESULTS / (f"bpm_{ENV}_n{N}_ep{EPOCHS}" + ("" if EPOCHS == 40 else f"_smoke{EPOCHS}")
              + beta_tag(Hyper.beta if BETA is None else BETA)
              + ("" if POSTPONE else "_nopp")
              + ("_ppc" if POSTPONE and POSTPONE_SCOPE == "component" else "")
              + {"hgt": "", "type_embed": "_temb", "type_embed_film": "_tembf", "aepn": "_aepn"}[NET]
              + ("" if BASES is None else f"_b{BASES}")
              + ("" if EPISODES == 8 else f"_eps{EPISODES}")
              + ("" if PLR is None else f"_plr{PLR:g}")
              + ("" if LAM is None else f"_lam{LAM:g}")
              + ("" if INIT is None else f"_init{os.path.splitext(os.path.basename(INIT))[0]}")
              + ("" if not CWARM else f"_cw{CWARM}")
              + ("" if RB == 1 else f"_rb{RB}")
              + ("" if BS is None else f"_bs{BS}")
              + ("_flat" if FLAT else "")
              + ("_local" if LOCAL else "")
              + ("_mc" if MARKING else "")
              + {"local": "", "full": "_fv"}[VIEW]
              + ("_cm" if CRITIC_MASK else "")
              + ("_abfull" if AGENT_BATCH == "full" else ""))
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
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": cfg.batch_size if BS is None else BS,
        "policy_lr": cfg.policy_lr if PLR is None else PLR, "policy_updates": cfg.policy_updates,
        "value_lr": cfg.value_lr, "value_updates": cfg.value_updates,
        "eps": cfg.ppo_eps, "gam": cfg.gam, "lam": cfg.lam if LAM is None else LAM, "ent_bonus": cfg.ent_bonus,
        "policy_kld_limit": getattr(cfg, "policy_kld_limit", None),
        "beta": cfg.beta,
        "verbose": 0, "use_gpu": GPU, "agent_seed": int(seed),
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
    if VIEW == "full":
        if NET != "aepn":
            raise SystemExit("view=full needs net=aepn (global attention is built into AEPNStack)")
        net_kw["encoder_kwargs"] = {**net_kw.get("encoder_kwargs", {}), "global_attn": True, "count_aggr": True}
    a["policy_kwargs"] = dict(net_kw)
    a["value_kwargs"] = dict(net_kw)
    if CRITIC_MASK:
        if VIEW != "full":
            raise SystemExit("critic_mask=1 needs view=full")
        a["value_kwargs"]["encoder_kwargs"] = {**net_kw.get("encoder_kwargs", {}), "component_mask": True}
    if FLAT:
        a["flat_obs"] = True
    if INIT is not None:
        a["load_policy_network"] = True
        a["policy_network"] = INIT
    if CWARM:
        a["critic_warmup"] = CWARM
    if RB != 1:
        a["rollout_batch"] = RB
    if LOCAL and method == "nfgae":
        a["local_obs"] = True
    if MARKING and method in ("nfgae", "agents"):
        a["marking_certificate"] = True
    a["global_view"] = {"local": False, "full": "edges"}[VIEW]
    if method == "agents":
        a["agent_batch"] = AGENT_BATCH
    a.update({"smdp_discount": True, "nfgae": method in ("nfgae", "agents"),
              "component_agents": method == "agents"})
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
    m["heldout"] = _heldout_scores(method, logdir, f"{method}__s{seed}")
    m.update({"env": ENV, "N": N, "method": method, "seed": seed,
              "epochs": EPOCHS, "episodes": EPISODES, "eval_seed": EVAL_SEED,
              "baselines": baselines, "minutes": (time.time() - t0) / 60.0,
              "update_device": "cuda" if GPU else "cpu"})
    return m


def _heldout_scores(method, logdir, name):
    """Greedy returns of the cell's final and best checkpoints on the held-out
    days (envs/heldout.py), or None for component agents / local turns."""
    import torch
    from heldout import make_env, score_policy, HELDOUT_SEED, HELDOUT_DAYS
    if method == "agents" or (LOCAL and method == "nfgae"):
        return None
    env = make_env(BUILD, N, LENGTH, POSTPONE, POSTPONE_SCOPE, FLAT,
                   {"local": False, "full": "edges"}[VIEW], MARKING and method in ("nfgae", "agents"))
    out = {}
    for which in ("final", "best"):
        path = os.path.join(logdir, name, f"{which}_policy.pth")
        if os.path.exists(path):
            pol = torch.load(path, weights_only=False).eval()
            out[which] = score_policy(env, pol, HELDOUT_SEED, HELDOUT_DAYS)
    return out


def _paired_anchors():
    """Random and heuristic on the evaluation days and on the held-out days
    (envs/heldout.py), cached in <OUTDIR>/eval_anchors.json."""
    from heldout import make_env, anchors, EVAL_SEED, EVAL_DAYS, HELDOUT_SEED, HELDOUT_DAYS
    f = OUTDIR / "eval_anchors.json"
    if f.exists():
        return json.loads(f.read_text())
    env = make_env(BUILD, N, LENGTH, POSTPONE, POSTPONE_SCOPE)
    a = {"eval": anchors(env, HEUR, EVAL_SEED, EVAL_DAYS),
         "heldout": anchors(env, HEUR, HELDOUT_SEED, HELDOUT_DAYS)}
    f.write_text(json.dumps(a, indent=1))
    return a


def _worker(payload):
    import torch
    method, seed, cfg, logdir, baselines, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (method, seed, train_cell(method, seed, cfg, logdir, baselines), None)
    except Exception:
        import traceback
        return (method, seed, None, traceback.format_exc())


def summary(out, baselines, paired):
    """final: greedy_final on the 20 evaluation days, normalised with random
    and heuristic on THOSE days. heldout final/best: the final and best
    checkpoints on 100 held-out days, normalised on the same days. (The best
    checkpoint is selected on the evaluation days; never quote its score
    there.)"""
    ev, ho = paired["eval"], paired["heldout"]
    nrm = lambda v, a: (v - a["random_mean"]) / (a["heuristic_mean"] - a["random_mean"])
    print(f"\n[{TAG}] anchors random/heuristic: eval days {ev['random_mean']:.2f}/{ev['heuristic_mean']:.2f}, "
          f"held-out days {ho['random_mean']:.2f}/{ho['heuristic_mean']:.2f}", flush=True)
    print(f"  {'method':<11} {'n':>3} {'final':>7} {'(SD)':>7} {'heldout final':>13} {'heldout best':>12} "
          f"{'collapsed':>9} {'ent':>6}", flush=True)
    rows = {}
    for m in METHODS:
        fin, hf, hb, ent = [], [], [], []
        for s in range(SEEDS):
            f = out / "cells" / f"N{N}__{m}__s{s}.json"
            if f.exists():
                j = json.loads(f.read_text())
                if j.get("greedy_final") is not None:
                    fin.append(nrm(j["greedy_final"], ev))
                    ent.append(j.get("entropy_final") or float("nan"))
                    h = j.get("heldout") or {}
                    if "final" in h:
                        hf.append(nrm(np.mean(h["final"]), ho))
                    if "best" in h:
                        hb.append(nrm(np.mean(h["best"]), ho))
        if fin:
            fin = np.array(fin)
            fmt = lambda xs: f"{np.mean(xs):.3f}" if xs else "-"
            rows[m] = {"n": len(fin), "final_mean": float(fin.mean()),
                       "final_sd": float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
                       "heldout_final_mean": float(np.mean(hf)) if hf else None,
                       "heldout_best_mean": float(np.mean(hb)) if hb else None,
                       "collapsed": int((fin <= 0.25).sum()),
                       "entropy_final": float(np.nanmean(ent)), "finals": fin.tolist(),
                       "heldout_finals": hf, "heldout_bests": hb}
            print(f"  {m:<11} {len(fin):>3} {fin.mean():>7.3f} {rows[m]['final_sd']:>7.3f} {fmt(hf):>13} "
                  f"{fmt(hb):>12} {rows[m]['collapsed']:>9d} {np.nanmean(ent):>6.2f}", flush=True)
    means = {k: {kk: v for kk, v in a.items() if kk.endswith("mean") or kk in ("seed", "days")}
             for k, a in paired.items()}
    (out / "summary.json").write_text(json.dumps({"env": ENV, "N": N, "baselines": baselines,
                                                  "paired_anchors": means, "rows": rows}, indent=2))


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
    paired = _paired_anchors()
    print(f"[{TAG}] anchors on the evaluation days: random={paired['eval']['random_mean']:.2f} "
          f"heuristic={paired['eval']['heuristic_mean']:.2f}", flush=True)

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
        r, h = paired["eval"]["random_mean"], paired["eval"]["heuristic_mean"]
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

    summary(OUTDIR, baselines, paired)


if __name__ == "__main__":
    w = next((int(a) for a in sys.argv[1:] if a.isdigit()), 4)
    main(w)
