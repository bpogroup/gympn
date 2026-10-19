"""What allocation do the trained policies choose? Behaviour check for the
start-of-shift allocation variant (insurer_alloc, 1 region).

Reads each run's best_policy.pth from results/<dir>/train/, so the training
folders must still be present. Each checkpoint is played greedily on 20
scenarios (seeds 777000+i). Per run we record:
  * the allocation of the pool: where each of the two experts (skill 1) and the
    generalist (skill 0) went: cl = claims, co = complaints, uw0 / uw1 =
    underwriting as a type-0 / type-1 underwriter;
  * the policy's probabilities over the allocation bindings at the first
    decision (does it still hesitate?);
  * the policy's own return, and the return the heuristic's operations get
    with the SAME allocation on the same scenarios. The gap between the two is
    what the policy loses in operations; the gap between the latter and the
    best allocation is what it loses in allocation.

Run: python scripts/alloc_behaviour.py [workers=3]
"""
import glob
import json
import os
import random
import sys
from collections import Counter

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SUITE, "envs"))

import numpy as np  # noqa: E402

RESULTS = os.path.join(SUITE, "results")
LENGTH, EPISODES, SEED0 = 20, 20, 777_000
BASE = "bpm_insurer_alloc_n1_ep40_nopp_aepn_flat"
PN_ATTRS = {}      # observation flags of the trained policy, set on each net (e.g. global_view)
ARMS = {  # label: (results dir, method)
    "joint PPO": (BASE, "ppo"),
    "NF-GAE static": (BASE, "nfgae"),
    "NF-GAE marking": (BASE + "_mc", "nfgae"),
}


def _canon(alloc):
    """Allocation as a sorted tuple of (destination, skill): order-free."""
    return tuple(sorted(alloc))


def episode(seed, choose):
    """One greedy episode; returns (allocation, first-decision allocation
    probabilities, return)."""
    import torch
    from gympn.environment import AEPN_Env
    from insurer_env import make_insurer_alloc, _alloc_binding
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = make_insurer_alloc(1)
    pn.length = LENGTH
    for k, v in PN_ATTRS.items():
        setattr(pn, k, v)
    env = AEPN_Env(pn)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    alloc, first, total = [], None, 0.0
    while env.pn.pn_actions:
        B = env.pn.pn_actions
        a, probs = choose(env, obs)
        if first is None and probs is not None:
            first = Counter()
            for b, p in zip(B, probs):
                ab = _alloc_binding(b)
                first[ab if ab is not None else ("ops", None)] += float(p)
        ab = _alloc_binding(B[a])
        if ab is not None:
            alloc.append(ab)
        obs, r, done, _, _ = env.step(a)
        total += r
        if done:
            break
    return _canon(alloc), first, total


def heuristic_return(alloc, seed):
    """The heuristic's operations with a given allocation, same scenario."""
    from insurer_env import make_alloc_heuristic
    heur = make_alloc_heuristic(list(alloc))

    def choose(env, obs):
        B = env.pn.pn_actions
        return B.index(heur(None, None, B)), None
    return episode(seed, choose)[2]


def policy_chooser(path):
    import torch
    from gympn.networks import policy_global_view
    pol = torch.load(path, weights_only=False)
    pol.eval()
    view = policy_global_view(pol)          # the graphs this policy was trained on
    if view is not None:
        PN_ATTRS["global_view"] = view

    def choose(env, obs):
        with torch.no_grad():
            probs = pol({'graph': obs['graph']}).flatten()   # the actor outputs probabilities
            assert abs(float(probs.sum()) - 1.0) < 1e-4, "expected action probabilities"
            return int(torch.argmax(probs).item()), probs.tolist()
    return choose


def run_arm(label):
    d, method = ARMS[label]
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, d, "train", f"{method}__s*", "best_policy.pth"))):
        seed = int(path.split(f"{method}__s")[1].split(os.sep)[0].split("/")[0])
        ch = policy_chooser(path)
        eps = [episode(SEED0 + i, ch) for i in range(EPISODES)]
        allocs = Counter(e[0] for e in eps)
        main_alloc = allocs.most_common(1)[0][0]
        first = Counter()
        for e in eps:
            for k, v in (e[1] or {}).items():
                first[k] += v / EPISODES
        own = float(np.mean([e[2] for e in eps]))
        heur_same = float(np.mean([heuristic_return(main_alloc, SEED0 + i) for i in range(EPISODES)]))
        runs.append({"seed": seed, "allocations": {str(k): v for k, v in allocs.items()},
                     "main_allocation": [list(x) for x in main_alloc],
                     "first_decision_probs": {str(k): round(v, 3) for k, v in first.items()},
                     "return": own, "heuristic_ops_same_allocation": heur_same})
    return label, runs


def main():
    workers = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("workers=")), 3))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, len(ARMS))) as pool:
        results = dict(pool.map(run_arm, list(ARMS)))
    best = [("cl", 1), ("co", 1), ("uw1", 0)]
    best_val = float(np.mean([heuristic_return(_canon(best), SEED0 + i) for i in range(EPISODES)]))
    print(f"best allocation {_canon(best)} with heuristic operations: {best_val:.2f}")
    for label, runs in results.items():
        print(f"\n== {label}")
        for r in runs:
            print(f"  s{r['seed']}: {r['main_allocation']} ({len(r['allocations'])} distinct)  "
                  f"own {r['return']:.2f}  heuristic-ops same alloc {r['heuristic_ops_same_allocation']:.2f}")
        print(f"  mean own {np.mean([r['return'] for r in runs]):.2f}  "
              f"mean heuristic-ops same alloc {np.mean([r['heuristic_ops_same_allocation'] for r in runs]):.2f}")
    json.dump({"best_allocation_value": best_val, "arms": results},
              open(os.path.join(RESULTS, "alloc_behaviour_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
