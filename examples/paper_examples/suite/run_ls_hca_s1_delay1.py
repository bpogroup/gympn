r"""Single-seed s1: ratio form + delay as the conditioning variable.

Progression on this exact config (stoch_config, s1, greedy_final):
    ls_hca difference form        1.45 +- 2.33  (5 seeds, collapse)
    ls_hca ratio + membership    12.55          (1 seed)
    lrq2                         12.02          (5 seeds)   <- the null
    ppo_clip                     13.60          (10 seeds)  <- the real bar
    cfpk                         14.03          (10 seeds)

Binary membership is constant across all k decisions sharing a reward (k
averages 8.48 on s1, never 1), so it cannot rank them; `delay` varies within a
lineage and moves the applied weight ~3x more often in the tail (frac |w-1|>0.2
rises 1.9% -> 5.6%). Whether that converts into policy quality is what this
measures. Beating 12.02 would show the reweighting does something; beating
13.60 would show it recovers lineage restriction's deficit.

ls_hca_z_feature defaults to 'delay' in Agent.__init__, so no plumbing is
needed -- the suite picks it up. Fresh output dir so the membership-variant
cell is not skipped as already-done.

Run: python run_ls_hca_s1_delay1.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import stoch_config  # noqa: E402
from run_suite import run_suite  # noqa: E402

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["ls_hca"]
cfg.seeds = 1
cfg.output_dir = Path("suite_results_ls_hca_s1_delay")
print(f"[s1_delay] ls_hca (z=delay) x 1 seed x {cfg.epochs} epochs -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)
