"""LCV probe: the lineage control variate on s1 + s3
(see CAUSAL_LCV_CONTROL_VARIATE.md; registered predictions in its §4).

Run: python run_lcv_probe.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence", "s3_stoch_mixed"]
    cfg.methods = ["lcv"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)
