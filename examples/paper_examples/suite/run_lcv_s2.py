"""Item 3: LCV on s2 (tier completeness). Cells into suite_results_stoch.

Run: python run_lcv_s2.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = stoch_config()
    cfg.envs = ["s2_stoch_scaled"]
    cfg.methods = ["lcv"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)
