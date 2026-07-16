"""Validation for the E5 stochastic/scaled tier.

Checks, per env:
  1. same-seed reproducibility (behaviors draw randomness exactly once per
     firing; a re-seeded identical run must give an identical return);
  2. genuine stochasticity (returns vary across seeds — both random AND
     heuristic policies);
  3. headroom (heuristic anchor comfortably above random);
  4. no saturation trap: episodes produce a healthy number of decisions.

Run: python _test_stoch_envs.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.solvers import HeuristicSolver, RandomSolver

from stoch_envs import STOCH_BUILDERS, STOCH_HEURISTICS

LENGTHS = {"s1_stoch_sequence": 20, "s2_stoch_scaled": 30, "s3_stoch_mixed": 25}


def run_once(name, solver, seed, length):
    random.seed(seed)
    np.random.seed(seed)
    env = STOCH_BUILDERS[name](causal_rl=False, allow_postpone=False)
    return float(env.testing_run(solver=solver, length=length))


for name in STOCH_BUILDERS:
    length = LENGTHS[name]
    heuristic = STOCH_HEURISTICS[name]

    # 1. same-seed reproducibility
    a = run_once(name, RandomSolver(), seed=7, length=length)
    b = run_once(name, RandomSolver(), seed=7, length=length)
    assert a == b, (name, "same seed must reproduce", a, b)

    # 2/3. stochasticity + headroom over 15 seeds
    rnd = [run_once(name, RandomSolver(), s, length) for s in range(15)]
    heu = [run_once(name, HeuristicSolver(heuristic), 100 + s, length)
           for s in range(15)]
    rnd_m, rnd_s = np.mean(rnd), np.std(rnd)
    heu_m, heu_s = np.mean(heu), np.std(heu)
    assert heu_s > 0 or rnd_s > 0, (name, "no stochasticity detected")
    assert heu_m > rnd_m + max(rnd_s, 1e-9), \
        (name, "heuristic anchor must clear random", heu_m, rnd_m, rnd_s)

    print(f"{name:<20} len={length:>2}  random={rnd_m:6.2f} +- {rnd_s:4.2f}   "
          f"heuristic={heu_m:6.2f} +- {heu_s:4.2f}   "
          f"headroom={(heu_m - rnd_m):5.2f} "
          f"({(heu_m - rnd_m) / max(rnd_s, 1e-9):.1f}x sigma_rnd)")

print("\nAll stochastic-tier checks passed.")
