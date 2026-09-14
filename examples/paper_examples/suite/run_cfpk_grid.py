"""X12b: G1 generality — cfpk on the deterministic a-h paper grid.

10 seeds x 8 topologies -> suite_results_paper (braked protocol), cells
added alongside the existing {ppo_clip, lrq, rudder, mc_q, lcv, lcv0, lva}
set (protocol-safe: cells are independent).

With the anneal removed the G1 floor is empirical, not by-construction —
so the grid run is the do-no-harm test that now matters: does CONSTANT
preference pressure damage envs that plain SMDP-PPO already solves?

Registered predictions (written before the run, 2026-07-19):
  P1 (do-no-harm): aggregate norm_final >= lcv0's 0.84 and no single env
      significantly below lcv0 paired by seed. Mechanism expectation: on
      solved envs the policy's top-2 gap is large, so fork gaps agree
      with the policy and preferences are either confirming or absent
      (the gate filters coupled branches) — pressure with nothing to
      teach must not destabilize.
  P2 (win where headroom): the disjoint topologies d/f/h carried the
      lcv0-era residuals (X0: 0.80/0.78/0.61). cfpk lifts f and h
      (predict f >= 0.90, h >= 0.75).
  P3 (cost): fork overhead stays < 2x lcv0 wall time per cell (grid
      episodes are short; lookahead-6 truncation binds often).

Run: python run_cfpk_grid.py [workers]
"""
import sys

from config import paper_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    cfg = paper_config()
    cfg.methods = ["cfpk"]
    run_suite(cfg, num_workers=workers)