"""X0: lcv + lcv0 on the deterministic 8-topology grid, braked protocol,
10 seeds each. Cells into suite_results_paper (resumable -- the existing
ppo_clip/lrq/rudder/mc_q cells from the 2026-07-10 paper run and shared
baselines are reused untouched; only the 160 new lcv/lcv0 cells are trained).

Registered prediction (PAPER_PLAN_LCV.md Sec 5, X0): lcv floor holds
everywhere; gains on the concurrent topologies (c/d) where exploration noise
across cases is off-lineage; c_hat ~ 0 on strict sequences (a/b). lcv0 is
predicted to land on ppo_clip (the discount-alone control).

Run: python run_x0_lcv.py [workers]
"""
import sys

from config import paper_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = paper_config()
    cfg.methods = ["lcv", "lcv0"]
    run_suite(cfg, num_workers=workers)