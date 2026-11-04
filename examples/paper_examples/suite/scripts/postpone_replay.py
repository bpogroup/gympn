"""Where do the policies of the paper protocol wait? Replay of the final policies
of the slow-mismatch insurer (DCL observation: options and clock node) on 20
scenarios (seeds 777000+i): the model-free arms of Table 2 (PPO none / global /
component, NF-PPO none / component), the component agents, and DCL and NF-DCL
with and without component-scoped postponement, next to the waiting heuristic.
Per episode and process: decisions to wait, "wasted" waits (a matched
underwriting assignment, or any claims or complaints assignment, was available
in the waiting process), and the matched and mismatched underwriting
assignments. Means over the seeds of an arm are printed and written to
results/postpone_replay.json.

usage: python scripts/postpone_replay.py [workers=4] [episodes=20]
"""
import glob
import json
import os
import random
import sys
from collections import Counter

import numpy as np

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SUITE, "envs"))
R = os.path.join(SUITE, "results")
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
WORKERS, EPISODES, LENGTH, SEED0 = int(opts.get("workers", 4)), int(opts.get("episodes", 20)), 20, 777_000
MF = "bpm_insurer_slow_n1_ep150_smoke150_beta0.05{pp}aepn_flat_mc_fv_clkn_mrg"
PI = "pi_insurer_slow_n1_R8_d40_mc{pp}_fair_mrg_clkn_sh32"
ARMS = {  # label: (policy glob, postpone mode, marking certificate)
    "PPO / none": (f"{MF.format(pp='_nopp_')}/train/ppo__s*/final_policy.pth", None, False),
    "NF-PPO / none": (f"{MF.format(pp='_nopp_')}/train/nfgae__s*/final_policy.pth", None, True),
    "PPO / global": (f"{MF.format(pp='_')}/train/ppo__s*/final_policy.pth", "global", False),
    "PPO / component": (f"{MF.format(pp='_ppc_')}/train/ppo__s*/final_policy.pth", "component", False),
    "NF-PPO / component": (f"{MF.format(pp='_ppc_')}/train/nfgae__s*/final_policy.pth", "component", True),
    "Agents / component": (f"{MF.format(pp='_ppc_')}_abfull/train/agents__s*/final_policy.pth", "component", True),
    "DCL / none": (f"{PI.format(pp='')}/policy_tau0.25*_final.pth", None, False),
    "NF-DCL / none": (f"{PI.format(pp='')}_cf40-40-40_dyn/policy_tau0.25*_final.pth", None, True),
    "DCL / component": (f"{PI.format(pp='_ppc')}/policy_tau0.25*_final.pth", "component", False),
    "NF-DCL / component": (f"{PI.format(pp='_ppc')}_cf40-40-40_dyn/policy_tau0.25*_final.pth", "component", True),
}


def _process(tid):
    return {"cl": "claims", "uw": "underwriting", "co": "complaints"}[tid[:2]]


def _env(mode, marking):
    from heldout import make_env
    from insurer_env import INSURER_BUILDERS
    env = make_env(INSURER_BUILDERS["insurer_slow"], 1, LENGTH, mode is not None,
                   mode or 'global', True, 'edges', marking)
    for pn in (env.pn, env.frozen_pn):          # the DCL observation, as run_bpm's _obs_options
        pn.clock_node, pn.obs_clock, pn.merge_options = True, False, True
    return env


def episode(env, seed, choose):
    import torch
    from insurer_env import _is_uw_mismatch, _is_postpone
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    obs, done = env.reset(), False
    part = env.pn.net_partition()
    comp_proc = {part[t._id]: _process(t._id) for t in env.pn.actions}
    c, total = Counter(), 0.0
    while not done and env.pn.pn_actions:
        B = env.pn.pn_actions
        a = choose(env, obs)
        b = B[a]
        if _is_postpone(b):
            if b[2] is None:                         # global postpone: every process waits
                procs = {comp_proc[part[x[2]._id]] for x in B if not _is_postpone(x)}
            else:
                procs = {comp_proc[b[2].comp]}
            for p in procs:
                c[f"wait_{p}"] += 1
                useful = [x for x in B if not _is_postpone(x) and comp_proc[part[x[2]._id]] == p
                          and not _is_uw_mismatch(x)]
                if useful:
                    c[f"wasted_wait_{p}"] += 1
        else:
            p = _process(b[2]._id)
            c[f"assign_{p}"] += 1
            if p == "underwriting":
                c["uw_mismatch" if _is_uw_mismatch(b) else "uw_matched"] += 1
        obs, r, done, _, _ = env.step(a)
        total += r
    c["reward"] = total
    return c


def policy_chooser(path, env):
    import torch
    pol = torch.load(path, weights_only=False).eval()
    env.pn.match_policy_view(pol)
    env.frozen_pn.match_policy_view(pol)

    def choose(env, obs):
        with torch.no_grad():
            return int(torch.argmax(pol(obs).reshape(-1)).item())
    return choose


def run_arm(label):
    import torch
    torch.set_num_threads(1)
    if label == "waiting heuristic":
        from insurer_env import insurer_wait_choice
        env = _env("component", False)

        def choose(env, obs):
            return insurer_wait_choice(env.pn.pn_actions, env.pn.net_partition(), 7)
        return label, [[episode(env, SEED0 + i, choose) for i in range(EPISODES)]]
    pattern, mode, marking = ARMS[label]
    runs = []
    for path in sorted(glob.glob(os.path.join(R, pattern))):
        if "_rn_" in os.path.basename(path):      # the component-local-loss ablation of DCL
            continue
        env = _env(mode, marking)
        ch = policy_chooser(path, env)
        runs.append([episode(env, SEED0 + i, ch) for i in range(EPISODES)])
    return label, runs


KEYS = ["wait_underwriting", "wasted_wait_underwriting", "uw_matched", "uw_mismatch",
        "wait_claims", "wasted_wait_claims", "wait_complaints", "wasted_wait_complaints", "reward"]
SHORT = ["wait UW", "wasted", "UW match", "UW mism.", "wait CL", "wasted", "wait CO", "wasted", "reward"]


def main():
    import multiprocessing as mp
    labels = ["waiting heuristic"] + list(ARMS)
    with mp.get_context("spawn").Pool(min(WORKERS, len(labels))) as pool:
        results = pool.map(run_arm, labels)
    print(f"per-episode means over the seeds of each arm (final policies, {EPISODES} scenarios)")
    print(f"{'arm':<22}{'seeds':>6} " + "".join(f"{s:>10}" for s in SHORT))
    out = {}
    for label, runs in results:
        per_run = {k: [float(np.mean([e[k] for e in eps])) for eps in runs] for k in KEYS}
        out[label] = per_run
        if runs:
            print(f"{label:<22}{len(runs):>6} " + "".join(f"{np.mean(per_run[k]):10.2f}" for k in KEYS))
        else:
            print(f"{label:<22}{0:>6}  (no policies found)")
    json.dump(out, open(os.path.join(R, "postpone_replay.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
