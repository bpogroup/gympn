r"""How noisy is ONE greedy eval point on s1?

Each greedy point in the suite is test_in_train's mean over test_episodes=20
episodes (stoch_config), and test_in_train calls env.reset() with NO seed
(environment.py:103) while s1's arrivals/service times draw from the global
`random` module (stoch_envs.py:57,69). So an eval point is 20 fresh draws
taken from wherever training left the RNG -- NOT common random numbers across
arms.

That is unbiased but noisy, and the question is how noisy. This measures the
sampling distribution of a 20-episode batch mean directly, using a FIXED
deterministic policy (the perfect heuristic) so every bit of the spread is
scenario noise rather than policy change: 200 independent batches of 20.

Reports the SD of a single eval point, and the implied SD of a 14-point mean
(the `mean_greedy` statistic) if the points were independent.

Run: python _diag_eval_noise.py [batches]
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gympn.solvers import HeuristicSolver
from config import stoch_config
from envs import make_env, HEURISTICS, perfect_heuristic
from run_suite import _env_length, _set_seed

BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ENV = "s1_stoch_sequence"
EP_PER_EVAL = 20                       # stoch_config.test_episodes

cfg = stoch_config()
length = _env_length(cfg, ENV)
heur = HEURISTICS.get(ENV, perfect_heuristic)

_set_seed(99_000)                      # seed ONCE, then let the stream run on,
                                       # mimicking how evals sit at arbitrary
                                       # RNG positions during training
episode_returns = []
batch_means = []
for b in range(BATCHES):
    rs = []
    for _ in range(EP_PER_EVAL):
        env = make_env(ENV, causal_rl=False, allow_postpone=False)
        rs.append(float(env.testing_run(solver=HeuristicSolver(heur), length=length)))
    episode_returns.extend(rs)
    batch_means.append(float(np.mean(rs)))

ep = np.array(episode_returns)
bm = np.array(batch_means)
print(f"[eval-noise] {ENV}, fixed heuristic policy, {BATCHES} batches x {EP_PER_EVAL} episodes")
print(f"  per-episode      mean={ep.mean():6.3f}  sd={ep.std(ddof=1):6.3f}")
print(f"  20-ep batch mean mean={bm.mean():6.3f}  sd={bm.std(ddof=1):6.3f}  "
      f"(predicted sd={ep.std(ddof=1)/np.sqrt(EP_PER_EVAL):.3f})")
print(f"  => one greedy point carries +-{bm.std(ddof=1):.3f} SD of pure scenario noise")
print(f"  => paired diff of two arms' single points: +-{bm.std(ddof=1)*np.sqrt(2):.3f}")
print(f"  => 14-point mean_greedy (if independent):  +-{bm.std(ddof=1)/np.sqrt(14):.3f}")
