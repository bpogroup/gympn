r"""Causal-order GAE (cgae) on s1 -- one seed, quick direction check.

Standard GAE accumulates TD errors along the TRAJECTORY index. In an
interleaved queueing system consecutive decisions usually belong to different
cases, so most of what it propagates is noise w.r.t. the action at t. cgae runs
the identical recursion along the PROVENANCE DAG instead: each reward is owned
by the latest decision in its lineage, and credit flows backward through causal
successors with lam-decay at causal depth and critic bootstrapping.

s1 references (stoch_config, greedy_final):
    lrq2  11.21   mc_q  13.41   ppo_clip 13.59   cfpk 14.03
lrq2 is the lineage-restricted Q-sample (lam=1, no bootstrap); cgae is the same
family with the recursion run on the right axis and lam<1.
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["cgae"]
cfg.seeds = 1
cfg.output_dir = Path("suite_results_cgae_s1")
print(f"[cgae] 1 seed x {cfg.epochs} epochs -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)
