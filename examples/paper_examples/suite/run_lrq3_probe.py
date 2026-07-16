"""LRQ-v3 quick comparison vs v2: s1 (v2's failure) + s3 (v2's strength).

v3 = exact decomposition, no mixing knob: A = c_lineage + q_off_theta(s,a) - V,
with V regressed on the full mc_q sample and the q_off head regressed on
(mc_q - lrq2) of the taken action. Cells go into suite_results_stoch so
analyze.py compares against all existing methods automatically.

Run: python run_lrq3_probe.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence", "s3_stoch_mixed"]
    cfg.methods = ["lrq3"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)
