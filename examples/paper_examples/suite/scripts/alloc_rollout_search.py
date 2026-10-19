"""Learning + search on the allocation variant (insurer_alloc, 1 region).

Can decision-time rollout search fix the allocation that the trained policies
fail to learn? For each run's best checkpoint (the operational policy):

  search   play out every one of the 40 allocations of the pool with the
           policy running operations, on K rollout scenarios shared by all
           candidates (common random numbers, seeds 900000+k), and pick the
           allocation with the best mean return;
  test     run the chosen allocation, policy on operations, on the 20 test
           scenarios of alloc_behaviour.py (seeds 777000+i, disjoint from the
           rollouts);
  truth    run all 40 allocations on the test scenarios, to see whether the
           search found the best allocation for this policy.

Every episode starts from the same deterministic marking, so the decision-time
search returns the same allocation in every scenario: one search per run.

Run: python scripts/alloc_rollout_search.py [K=16] [workers=12]
"""
import glob
import itertools
import json
import os
import sys

SUITE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SUITE, "envs"))
sys.path.insert(0, os.path.join(SUITE, "scripts"))

import numpy as np  # noqa: E402

from alloc_behaviour import ARMS, RESULTS, SEED0, EPISODES, episode, policy_chooser  # noqa: E402

ROLL0 = 900_000
K = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("K=")), 16))
DESTS = ("cl", "co", "uw0", "uw1")
ALLOCS = [[(e1, 1), (e2, 1), (g, 0)]
          for e1, e2 in itertools.combinations_with_replacement(DESTS, 2) for g in DESTS]


def _key(plan):
    return tuple(sorted(plan))


def plan_then_policy(plan, policy_choose):
    """Allocate by `plan`, then let the trained policy run operations."""
    from insurer_env import _alloc_binding, make_alloc_heuristic
    alloc_pick = make_alloc_heuristic(plan)

    def choose(env, obs):
        B = env.pn.pn_actions
        if any(_alloc_binding(b) is not None for b in B):
            return B.index(alloc_pick(None, None, B)), None
        return policy_choose(env, obs)
    return choose


def run_one(job):
    import torch
    torch.set_num_threads(1)
    label, path = job
    pol = policy_chooser(path)
    search = {}
    for plan in ALLOCS:
        ch = plan_then_policy(plan, pol)
        search[_key(plan)] = float(np.mean([episode(ROLL0 + k, ch)[2] for k in range(K)]))
    truth = {}
    for plan in ALLOCS:
        ch = plan_then_policy(plan, pol)
        truth[_key(plan)] = float(np.mean([episode(SEED0 + i, ch)[2] for i in range(EPISODES)]))
    chosen = max(search, key=search.get)
    best = max(truth, key=truth.get)
    rank = sorted(truth.values(), reverse=True).index(truth[chosen]) + 1
    seed = int(os.path.basename(os.path.dirname(path)).split("__s")[1])
    return {"arm": label, "seed": seed, "chosen": [list(x) for x in chosen],
            "chosen_test": truth[chosen], "best_for_policy": [list(x) for x in best],
            "best_for_policy_test": truth[best], "chosen_rank": rank,
            "search": {str(k): v for k, v in search.items()},
            "truth": {str(k): v for k, v in truth.items()}}


def main():
    workers = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("workers=")), 12))
    jobs = []
    for label, (d, method) in ARMS.items():
        for path in sorted(glob.glob(os.path.join(RESULTS, d, "train", f"{method}__s*", "best_policy.pth"))):
            jobs.append((label, path))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        out = pool.map(run_one, jobs)
    json.dump({"K": K, "runs": out},
              open(os.path.join(RESULTS, f"alloc_rollout_search_K{K}.json"), "w"), indent=1)
    print(f"K={K} rollouts per allocation; test = 20 held-out scenarios")
    for label in ARMS:
        rs = [r for r in out if r["arm"] == label]
        print(f"\n== {label}")
        for r in sorted(rs, key=lambda r: r["seed"]):
            print(f"  s{r['seed']}: chose {r['chosen']} test {r['chosen_test']:.2f} "
                  f"(rank {r['chosen_rank']}/40; policy's best {r['best_for_policy_test']:.2f})")


if __name__ == "__main__":
    main()
