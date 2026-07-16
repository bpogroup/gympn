"""LQI probe: the Q-native consumer on s1 + s3 (see CAUSAL_LQI_QNATIVE.md).

Cells land in suite_results_stoch so analyze.py pairs LQI against all
existing methods (ppo_clip, lrq, lrq2, rudder, mc_q, lrq3) automatically.

Run: python run_lqi_probe.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence", "s3_stoch_mixed"]
    cfg.methods = ["lqi"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)
