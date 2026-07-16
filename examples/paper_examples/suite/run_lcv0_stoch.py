"""X1: lcv + lcv0 across the full stochastic tier (s1/s2/s3), 10 seeds each.
Cells into suite_results_stoch (resumable -- s1/s3 x lcv cells already exist
from the 2026-07-14 probe and are skipped; this recovers the s2 x lcv run
lost to the restart AND adds lcv0, LCV's c_hat=0 twin, everywhere).

Run: python run_lcv0_stoch.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = stoch_config()
    cfg.methods = ["lcv", "lcv0"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)