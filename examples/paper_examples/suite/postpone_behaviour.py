"""Where do the trained policies wait? Behaviour check for the postponement
experiment (insurer_slow, 1 region; run_postpone_queue.sh).

Each run's best checkpoint is played greedily on 20 scenarios (seeds 777000+i)
in the environment it was trained in (no / global / component postpone). Per
episode we count, per process: decisions to wait, and in underwriting the
matched and mismatched assignments, plus "wasted" waits (the process waited
although a matched underwriting assignment or any claims/complaints assignment
was available). The waiting heuristic (the anchor) is reported for reference.

Run: python postpone_behaviour.py [workers=5]
"""
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LENGTH, EPISODES, SEED0 = 20, 20, 777_000
ARMS = {  # (label, results dir suffix, method, postpone mode)
    "PPO / none": ("_nopp_aepn_flat", "ppo", None),
    "NF-GAE / none": ("_nopp_aepn_flat", "nfgae", None),
    "PPO / global": ("_aepn_flat", "ppo", "global"),
    "PPO / component": ("_ppc_aepn_flat", "ppo", "component"),
    "NF-GAE / component": ("_ppc_aepn_flat", "nfgae", "component"),
}


def _process(tid):
    return {"cl": "claims", "uw": "underwriting", "co": "complaints"}[tid[:2]]


def episode(seed, mode, choose):
    import torch
    from gympn.environment import AEPN_Env
    from insurer_env import INSURER_BUILDERS, _is_uw_mismatch, _is_postpone
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = INSURER_BUILDERS["insurer_slow"](1, allow_postpone=mode is not None)
    pn.length = LENGTH
    if mode == "component":
        pn.postpone_scope = 'component'
    env = AEPN_Env(pn)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    part = env.pn.net_partition()
    comp_proc = {part[t._id]: _process(t._id) for t in env.pn.actions}
    c, total = Counter(), 0.0
    while env.pn.pn_actions:
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
        if done:
            break
    c["reward"] = total
    return c


def policy_chooser(path):
    import torch
    pol = torch.load(path, weights_only=False)
    pol.eval()

    def choose(env, obs):
        with torch.no_grad():
            return int(torch.argmax(pol({'graph': obs['graph']}).flatten()).item())
    return choose



def run_arm(label):
    if label == "waiting heuristic":
        def choose(env, obs):
            from insurer_env import insurer_wait_choice
            return insurer_wait_choice(env.pn.pn_actions, env.pn.net_partition(), 7)
        eps = [episode(SEED0 + i, "component", choose) for i in range(EPISODES)]
        return label, [eps]
    suffix, method, mode = ARMS[label]
    d = os.path.join(HERE, f"suite_results_bpm_insurer_slow_n1_ep40{suffix}", "train")
    runs = []
    for path in sorted(glob.glob(os.path.join(d, f"{method}__s*", "best_policy.pth"))):
        ch = policy_chooser(path)
        runs.append([episode(SEED0 + i, mode, ch) for i in range(EPISODES)])
    return label, runs


def report(results):
    keys = ["wait_underwriting", "wasted_wait_underwriting", "uw_matched", "uw_mismatch",
            "wait_claims", "wasted_wait_claims", "wait_complaints", "wasted_wait_complaints", "reward"]
    short = ["wait UW", "wasted", "UW match", "UW mism.", "wait CL", "wasted", "wait CO", "wasted", "reward"]
    print(f"{'arm':<22}{'runs':>5} " + "".join(f"{s:>10}" for s in short))
    out = {}
    for label, runs in results:
        per_run = {k: [np.mean([e[k] for e in eps]) for eps in runs] for k in keys}
        out[label] = per_run
        print(f"{label:<22}{len(runs):>5} " + "".join(f"{np.mean(per_run[k]):10.2f}" for k in keys))
    return out


def main():
    workers = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("workers=")), 5))
    labels = ["waiting heuristic"] + list(ARMS)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, len(labels))) as pool:
        results = pool.map(run_arm, labels)
    print("per-episode means over runs (each run = one seed's best checkpoint, 20 scenarios):")
    out = report(results)
    json.dump(out, open(os.path.join(HERE, "postpone_behaviour_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
