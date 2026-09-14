r"""Single-seed s1: ungated ratio form (null = mc_q instead of lrq2).

ls_hca_gate defaults to False in Agent.__init__, so the suite picks it up.
Ladder on this exact config (stoch_config, s1, greedy_final):
    ls_hca difference form     1.45 +- 2.33  (5 seeds, collapse)
    lrq2                      11.21 / 12.02  (10 / 5 seeds)  <- gated null
    ls_hca ratio + membership 12.55          (1 seed, gated)
    ls_hca ratio + delay      12.60          (1 seed, gated)
    mc_q                      13.41          (10 seeds)      <- ungated null
    ppo_clip                  13.59          (10 seeds)
    cfpk                      14.03          (10 seeds)

Expectation: ~13.4. The ungated null is mc_q by construction (verified exactly
on non-postpone decisions), so this should land near mc_q unless the delay
reweighting moves it. Landing near 13.4 would mean LS-HCA is finally on par
with PPO rather than 2.4 below it.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["ls_hca"]
cfg.seeds = 1
cfg.output_dir = Path("suite_results_ls_hca_s1_ungated")
print(f"[s1_ungated] ls_hca (ungated, z=delay) x 1 seed -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)
