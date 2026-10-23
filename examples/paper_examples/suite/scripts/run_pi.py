"""Rollout policy iteration, first round: distil the rollout planner over the
heuristic into the policy network (gympn.policy_iteration).

1. The planner (RolloutPlanner over HeuristicBase, R rollouts per option,
   options and returns restricted to the acting component, marking
   certificate) plays `days` training days (seeds 2_000_000+d: neither the
   evaluation, the held-out nor the anchor days). Every decision is recorded
   with the observation PPO's actor sees (flat, whole-net view, components).
2. PPO's actor (gympn.train.make_policy_network, same settings as run_bpm's
   AEPN runs) is trained on the planner's choices: soft targets
   softmax(option scores / tau) over the compared options (tau=0: the
   planner's choice only).
3. The network is scored greedily on the 20 evaluation days after every
   epoch; the epoch with the best score is selected (so its score there is
   optimistic), and the selected and the final networks are scored on the
   100 held-out days. All normalised with random and heuristic on the same
   days (envs/heldout.py).

Output: results/pi_<env>_n<N>_R<R>_d<days>_mc/ with decisions.pt (the
recorded decisions and option scores, reused if present), and per tau
policy_tau<tau>.pth (the selected network, loadable as run_bpm init=),
policy_tau<tau>_final.pth and summary_tau<tau>.json; eval_anchors.json.

usage: python scripts/run_pi.py env=bpic15 [N=1] [days=40] [R=8] [workers=8]
                                [epochs=15] [tau=0.25] [gpu=1] [seed=0]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE / "envs"))

from heldout import (make_env as _make_env, anchors, score_policy, EVAL_SEED, EVAL_DAYS,  # noqa: E402
                     HELDOUT_SEED, HELDOUT_DAYS)

opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ENV, N = opts.get("env", "bpic15"), int(opts.get("N", 1))
DAYS, R, WORKERS = int(opts.get("days", 40)), int(opts.get("R", 8)), int(opts.get("workers", 8))
EPOCHS, GPU, SEED = int(opts.get("epochs", 15)), opts.get("gpu", "0") == "1", int(opts.get("seed", 0))
TAU = float(opts.get("tau", 0.25))
LENGTH, TRAIN_SEED = 20, 2_000_000
OUTDIR = SUITE / "results" / f"pi_{ENV}_n{N}_R{R}_d{DAYS}_mc"


def registries():
    from bpm_envs import BPM_BUILDERS, BPM_HEURISTICS
    from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS
    from bpic15_env import BPIC15_BUILDERS, BPIC15_HEURISTICS
    return ({**BPM_BUILDERS, **INSURER_BUILDERS, **BPIC15_BUILDERS},
            {**BPM_HEURISTICS, **INSURER_HEURISTICS, **BPIC15_HEURISTICS})


def make_env():
    """The training/evaluation env of run_bpm's AEPN marking-certificate runs."""
    builders, _ = registries()
    return _make_env(builders[ENV], N, LENGTH, flat_obs=True, global_view='edges', marking_certificate=True)


def make_planner(env):
    from gympn.policy_iteration import RolloutPlanner, HeuristicBase
    _, heur = registries()
    return RolloutPlanner(HeuristicBase(heur[ENV]), rollouts=R, component=True, seed=SEED + 1)


def make_policy(env):
    """PPO's actor, as gympn.train builds it for run_bpm's AEPN runs."""
    from gympn.train import make_parser, make_policy_network
    args = make_parser().parse_args([])
    args.policy_kwargs = {"encoder": "aepn", "encoder_kwargs": {"global_attn": True, "count_aggr": True}}
    args.flat_obs, args.global_view = True, 'edges'
    torch.manual_seed(SEED)
    return make_policy_network(args, metadata=env.pn.make_metadata())


def log(msg):
    print(f"[pi-{ENV}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def main():
    from gympn.policy_iteration import collect_decisions, distill
    OUTDIR.mkdir(parents=True, exist_ok=True)
    env = make_env()
    _, heur = registries()
    fa = OUTDIR / "eval_anchors.json"
    if fa.exists():
        anc = json.loads(fa.read_text())
    else:
        anc = {"eval": anchors(env, heur[ENV], EVAL_SEED, EVAL_DAYS),
               "heldout": anchors(env, heur[ENV], HELDOUT_SEED, HELDOUT_DAYS)}
        fa.write_text(json.dumps(anc, indent=1))
    nrm = lambda v, a: (v - a["random_mean"]) / (a["heuristic_mean"] - a["random_mean"])
    log(f"anchors random/heuristic: eval {anc['eval']['random_mean']:.2f}/{anc['eval']['heuristic_mean']:.2f}, "
        f"held-out {anc['heldout']['random_mean']:.2f}/{anc['heldout']['heuristic_mean']:.2f}")

    fd = OUTDIR / "decisions.pt"
    t0 = time.time()
    graphs = torch.load(fd, weights_only=False) if fd.exists() else None
    if graphs is not None and not hasattr(graphs[0], "t_scores"):
        log("recorded decisions have no option scores: collecting again")
        graphs = None
    if graphs is not None:
        log(f"loaded {len(graphs)} recorded decisions")
    else:
        log(f"planner (R={R}) on {DAYS} training days, {WORKERS} workers")
        graphs = collect_decisions(make_env, make_planner, [TRAIN_SEED + d for d in range(DAYS)],
                                   workers=WORKERS, log=log)
        torch.save(graphs, fd)
        log(f"{len(graphs)} decisions recorded ({(time.time() - t0) / 60:.0f} min)")

    policy = make_policy(env)

    def evaluate(pol):
        r = score_policy(make_env(), pol, EVAL_SEED, EVAL_DAYS)
        return {"eval_norm": float(nrm(np.mean(r), anc["eval"]))}

    hist = distill(policy, graphs, epochs=EPOCHS, device="cuda" if GPU and torch.cuda.is_available() else None,
                   seed=SEED, log=log, evaluate=evaluate, eval_every=1, tau=TAU, select="eval_norm")
    ckpts = hist.pop("checkpoints")
    tag = f"tau{TAU:g}"
    final = make_policy(env)
    with torch.no_grad():
        final({'graph': graphs[0]})
    final.load_state_dict(ckpts[max(ckpts)])
    final.eval()
    torch.save(policy, OUTDIR / f"policy_{tag}.pth")
    torch.save(final, OUTDIR / f"policy_{tag}_final.pth")
    ho_sel = score_policy(make_env(), policy, HELDOUT_SEED, HELDOUT_DAYS)
    ho_fin = score_policy(make_env(), final, HELDOUT_SEED, HELDOUT_DAYS)
    sel = hist["selected_epoch"]
    res = {"env": ENV, "N": N, "days": DAYS, "R": R, "epochs": EPOCHS, "tau": TAU, "seed": SEED,
           "decisions": len(graphs), "history": hist, "selected_epoch": sel,
           "eval_norm_selected": next(e["eval_norm"] for e in hist["evals"] if e["epoch"] == sel),
           "eval_norm_final": hist["evals"][-1]["eval_norm"],
           "heldout_norm_selected": float(nrm(np.mean(ho_sel), anc["heldout"])),
           "heldout_norm_final": float(nrm(np.mean(ho_fin), anc["heldout"])),
           "heldout_selected": ho_sel, "heldout_final": ho_fin, "minutes": (time.time() - t0) / 60}
    (OUTDIR / f"summary_{tag}.json").write_text(json.dumps(res, indent=1))
    log(f"tau={TAU:g}: selected epoch {sel} (eval days {res['eval_norm_selected']:.3f}, selected on them) "
        f"-> held-out {res['heldout_norm_selected']:.3f}; final epoch: eval {res['eval_norm_final']:.3f}, "
        f"held-out {res['heldout_norm_final']:.3f} (random 0, heuristic 1 on the same days)")


if __name__ == "__main__":
    main()
