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

Later rounds of policy iteration: base=<policy.pth> makes the planner play
out with that network (greedy) instead of the heuristic, init=<policy.pth>
starts the distillation from a network, and round=k (k > 1) uses new training
days (seeds 2_000_000 + 100_000 (k-1) + d) and writes to ..._mc_r<k>/.

usage: python scripts/run_pi.py env=bpic15 [N=1] [days=40] [R=8] [workers=8]
                                [epochs=15] [tau=0.25] [gpu=1] [seed=0]
                                [base=<policy.pth>] [init=<policy.pth>] [round=1]
                                [lr=1e-3] [ema=0.995] [extra=<dir>,<dir>] [name=<tag>]
extra: the decisions recorded by earlier rounds (their results directories)
are added to this round's (aggregation); name: a tag for this distillation's
output files, so several distillations of the same decisions can coexist.
margin=m: the planner departs from its base only when the best option's mean
rollout return beats the base's by more than m (distil only its confident
improvements); fair=1: every rollout redraws what a policy cannot observe
(bpic15: the next arrival times); confirm=k: a departure from the base is kept
only if an independent scoring with k rollouts per option repeats it; age=1:
every observed token gets its age, clock - token.time (GymProblem.obs_token_age;
with env=bpic15_wip that includes the time spent on each task in progress).
clock=1: every observed token gets the time of day (GymProblem.obs_clock);
merge=1: one action node per option, the FIFO binding fired (GymProblem.merge_options);
clocknode=1: a dedicated clock node (GymProblem.clock_node);
race=eps [rz=2] [rstep=2] [rmin=4] [rcap=64]: racing instead (RolloutPlanner
race_eps): rounds of rstep paired rollouts of the surviving options, an option
dropped once rz standard errors behind the leader, stop when no survivor can
beat the leader by more than eps (or at rcap rollouts per option); setlab=1
trains on the survivor set (set_class_nll) instead of the single choice;
renorm=1: the loss renormalizes the class probabilities within the deciding
component (distill(renorm=True)); cset=k: an independent collection, training
days offset by 10 000 k (suffix _cset<k>);
cf=n0,n1,...: component-factored collection (policy_iteration
collect_component_decisions): component k of the episode (episode_components
order, logged) is simulated alone on n_k training days, instead of DAYS whole
days; allocate simulation where labels are worth most;
sh=M: Deep Controlled Learning's action selection, Sequential Halving with
common random numbers and M rollouts per option in total (RolloutPlanner
sh_budget; use with margin=0, no test). DCL's rounds: round=k base=<the
previous round's policy_tau0.pth> init=<the same> (scripts/run_dcl.sh);
test=k [min_t=2]: a departure is kept only if k fresh rollouts of it and
of the base's choice, paired, give a mean advantage above the margin with a t
statistic of at least min_t (RolloutPlanner test/min_t).
All of these add a suffix to the results directory.
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
BASE = os.path.abspath(opts["base"]) if "base" in opts else None
INIT = os.path.abspath(opts["init"]) if "init" in opts else None
ROUND = int(opts.get("round", 1))
LR = float(opts.get("lr", 1e-3))
EMA = float(opts["ema"]) if "ema" in opts else None
EXTRA = [d for d in opts.get("extra", "").split(",") if d]
NAME = opts.get("name", "")
MARGIN = float(opts.get("margin", 0.0))
FAIR = opts.get("fair", "0") == "1"
CONFIRM = int(opts.get("confirm", 0))
AGE = opts.get("age", "0") == "1"
CLOCK = opts.get("clock", "0") == "1"
MERGE = opts.get("merge", "0") == "1"
CLOCKNODE = opts.get("clocknode", "0") == "1"
SH = int(opts.get("sh", 0))
RACE = float(opts["race"]) if "race" in opts else None
RZ, RSTEP = float(opts.get("rz", 2.0)), int(opts.get("rstep", 2))
RMIN, RCAP = int(opts.get("rmin", 4)), int(opts.get("rcap", 64))
SETLAB = opts.get("setlab", "0") == "1"
CF = [int(x) for x in opts["cf"].split(",")] if "cf" in opts else None
RENORM = opts.get("renorm", "0") == "1"
CSET = int(opts.get("cset", 0))
TEST, MIN_T = int(opts.get("test", 0)), float(opts.get("min_t", 2.0))
LENGTH, TRAIN_SEED = 20, 2_000_000 + 100_000 * (ROUND - 1) + 10_000 * CSET
OUTDIR = SUITE / "results" / (f"pi_{ENV}_n{N}_R{R}_d{DAYS}_mc" + (f"_r{ROUND}" if ROUND > 1 else "")
                         + (f"_m{MARGIN:g}" if MARGIN else "") + ("_fair" if FAIR else "") + (f"_c{CONFIRM}" if CONFIRM else "")
                         + ("_age" if AGE else "") + ("_clk" if CLOCK else "") + ("_mrg" if MERGE else "") + ("_clkn" if CLOCKNODE else "") + (f"_sh{SH}" if SH else "")
                         + (f"_race{RACE:g}z{RZ:g}c{RCAP}" if RACE is not None else "")
                         + ("_cf" + "-".join(map(str, CF)) if CF else "")
                         + (f"_cset{CSET}" if CSET else "") + (f"_t{TEST}z{MIN_T:g}" if TEST else ""))


def registries():
    from bpm_envs import BPM_BUILDERS, BPM_HEURISTICS
    from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS
    from bpic15_env import BPIC15_BUILDERS, BPIC15_HEURISTICS
    return ({**BPM_BUILDERS, **INSURER_BUILDERS, **BPIC15_BUILDERS},
            {**BPM_HEURISTICS, **INSURER_HEURISTICS, **BPIC15_HEURISTICS})


def make_env():
    """The training/evaluation env of run_bpm's AEPN marking-certificate runs."""
    builders, _ = registries()
    env = _make_env(builders[ENV], N, LENGTH, flat_obs=True, global_view='edges', marking_certificate=True)
    env.pn.obs_token_age = env.frozen_pn.obs_token_age = AGE      # (reset starts from frozen_pn)
    env.pn.obs_clock = env.frozen_pn.obs_clock = CLOCK
    env.pn.merge_options = env.frozen_pn.merge_options = MERGE
    env.pn.clock_node = env.frozen_pn.clock_node = CLOCKNODE
    return env


def make_planner(env):
    from gympn.policy_iteration import RolloutPlanner, HeuristicBase, NetworkBase
    if BASE is not None:
        torch.set_num_threads(1)
        base = NetworkBase(torch.load(BASE, weights_only=False).eval())
    else:
        _, heur = registries()
        base = HeuristicBase(heur[ENV])
    resample = None
    if FAIR:
        from bpic15_env import bpic15_resample_hidden
        assert ENV.startswith("bpic15"), "fair=1: resampling defined for bpic15 only"
        resample = bpic15_resample_hidden
    return RolloutPlanner(base, rollouts=R, component=True, seed=SEED + ROUND, margin=MARGIN,
                          resample=resample, confirm=CONFIRM,
                          test=TEST, min_t=MIN_T, sh_budget=SH, race_eps=RACE, race_z=RZ,
                          race_step=RSTEP, race_min=RMIN, race_cap=RCAP)


def make_policy(env, init=None):
    """PPO's actor, as gympn.train builds it for run_bpm's AEPN runs (or the
    saved network `init`)."""
    if init is not None:
        return torch.load(init, weights_only=False)
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
        log(f"planner (R={R}, base {'network ' + os.path.basename(BASE) if BASE else 'heuristic'}) "
            f"on {DAYS} training days (round {ROUND}), {WORKERS} workers")
        if CF:
            from gympn.policy_iteration import episode_components, collect_component_decisions
            comps = episode_components(make_env, seed=TRAIN_SEED)
            assert len(comps) == len(CF), f"cf= needs {len(comps)} day counts, one per component"
            for k, (m, n) in enumerate(zip(comps, CF)):
                log(f"component {k}: {n} days, {len(m)} transitions, e.g. {sorted(m)[:3]}")
            tasks = [(TRAIN_SEED + d, m) for m, n in zip(comps, CF) for d in range(n)]
            graphs = collect_component_decisions(make_env, make_planner, tasks, workers=WORKERS, log=log)
        else:
            graphs = collect_decisions(make_env, make_planner, [TRAIN_SEED + d for d in range(DAYS)],
                                       workers=WORKERS, log=log)
        torch.save(graphs, fd)
        log(f"{len(graphs)} decisions recorded ({(time.time() - t0) / 60:.0f} min)")

    for d in EXTRA:
        more = torch.load(SUITE / "results" / d / "decisions.pt", weights_only=False)
        log(f"+ {len(more)} decisions from {d}")
        graphs = graphs + more
    policy = make_policy(env, INIT)

    def evaluate(pol):
        r = score_policy(make_env(), pol, EVAL_SEED, EVAL_DAYS)
        return {"eval_norm": float(nrm(np.mean(r), anc["eval"]))}

    hist = distill(policy, graphs, epochs=EPOCHS, device="cuda" if GPU and torch.cuda.is_available() else None,
                   seed=SEED, log=log, evaluate=evaluate, eval_every=1, tau=TAU, select="eval_norm",
                   lr=LR, ema=EMA, set_labels=SETLAB, renorm=RENORM)
    ckpts = hist.pop("checkpoints")
    tag = f"tau{TAU:g}" + ("_rn" if RENORM else "") + (f"_{NAME}" if NAME else "")
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
           "lr": LR, "ema": EMA, "extra": EXTRA, "init": INIT, "base": BASE, "round": ROUND,
           "margin": MARGIN, "fair": FAIR, "confirm": CONFIRM, "age": AGE, "clock": CLOCK, "merge": MERGE, "clocknode": CLOCKNODE, "sh": SH, "race": RACE, "rz": RZ, "rstep": RSTEP,
           "rmin": RMIN, "rcap": RCAP, "setlab": SETLAB, "cf": CF, "renorm": RENORM, "cset": CSET, "test": TEST, "min_t": MIN_T,
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
