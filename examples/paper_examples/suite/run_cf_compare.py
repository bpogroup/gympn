r"""Head-to-head: the measured counterfactual advantage (cf) vs PPO / mc_q / lrq
on a stochastic env, through the REAL training pipeline. cf forks every decision
(CRN, lineage-restricted returns) so it is the slow method -- kept to a modest
budget here for a first end-to-end signal.
"""
import sys
from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["ppo_clip", "mc_q", "lrq", "cf"]
    cfg.seeds = 3
    cfg.epochs = 12
    cfg.episodes_per_epoch = 8
    cfg.test_freq = 2
    cfg.output_dir = "cf_compare_s1"
    run_suite(cfg, num_workers=workers)