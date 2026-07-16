"""Falsification run for the KL-brake fix: envs b & f, LRQ, 10 seeds.

At beta=0.5 WITHOUT a KL limit (no suite run ever set one — the flag was
silently None, and the old metric was unusable anyway), envs b/f showed
post-convergence collapses in ~2-3/10 LRQ seeds (suite_results_lrq: b drift
1.00, f drift 1.80 mean). Hypothesis: the collapses are single catastrophic
updates (per-state KL >= 1 on a near-deterministic policy) that the fixed KL
brake (exact per-state KL(old||new), limit 0.15) stops.

Prediction: greedy drift on b/f drops toward the healthy-seed level (~0.2)
with unchanged convergence speed. Compare against suite_results_lrq cells.

Run: python run_bf_kldfix.py [workers]
"""
import sys
from pathlib import Path

from config import SuiteConfig
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = SuiteConfig(
        envs=["b_sequence_disjoint", "f_loop_disjoint"],
        methods=["lrq"],
        seeds=10,
        output_dir=Path("suite_results_kldfix"),
        # policy_kld_limit inherits the new 0.15 default from SuiteConfig.
    )
    run_suite(cfg, num_workers=workers)
