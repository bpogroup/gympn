r"""cfgae vs ppo_clip on s1, seed 0 -- paired (shared init, post seed-fix).

cfgae = ccf's realized component partition, consumed through PPO's OWN
SMDP-GAE recursion instead of as a Monte-Carlo Q-sample: per component, run
the ordinary recursion over the reward stream masked to that component; each
decision reads the advantage from its own component's pass. The point of the
construction is the K=1 limit -- one component means the mask is identically
one and the recursion is term-for-term PPO's, so unlike every lambda=1
Monte-Carlo scheme here (lrq/ccf/s_ccf, whose K=1 limit is mc_q) it can only
depart from PPO when the partition is actually non-trivial.

Both arms are re-run so they share the seed-fixed initial policy; the only
difference is the credit scheme.

s1 references (greedy_final): lrq2 11.21, mc_q 13.41, ppo_clip 13.59,
cfpk 14.03; paired cgae seed 0 = 12.10.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["ppo_clip", "cfgae"]
cfg.seeds = 1
cfg.output_dir = Path("suite_results_cfgae_paired")
print(f"[paired] {cfg.methods} x seed 0 x {cfg.epochs} epochs -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)   # 1 worker: a 2-worker pool died with BrokenProcessPool