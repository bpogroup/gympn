r"""s1 with num_layers=8 instead of the default 3 -- one seed, plain PPO.

Motivation (_diag_lineage_depth.py): the realized decision->reward lineage
depth on s1 has median 5.0, mean 5.80, p90 11 -- and 62.8% of causal links
exceed num_layers=3. So for most decisions the actor's embedding cannot reach
the reward that decision feeds. That is a structural limit, and it would
explain why every credit-side mechanism this project has tried lands in the
same narrow band regardless of what it computes.

Plain ppo_clip on purpose: it isolates the ARCHITECTURE from any credit
scheme. If a receptive-field ceiling is what everything has been hitting,
raising it should move PPO too. Reference: stored ppo_clip on s1 at the
default depth, 10 seeds, mean 13.59.

net_num_layers feeds run_suite._net_kwargs, which applies it to BOTH actor and
critic, so this is a whole-network depth change, not actor-only.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["ppo_clip"]
cfg.seeds = 1
cfg.net_num_layers = 8
cfg.output_dir = Path("suite_results_s1_deep")
print(f"[s1_deep] ppo_clip x 1 seed, num_layers={cfg.net_num_layers} -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)
