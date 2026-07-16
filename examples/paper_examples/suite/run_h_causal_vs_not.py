"""Run ONLY env h (exclusive-choice disjoint) with and without causal_rl.

Uses the suite's validated runner:
  - ppo_clip   = WITHOUT causal_rl (plain PPO)
  - causal_td0 = WITH causal_rl (flow_dag, sink-less postpone, beta=0.2, sc=0.5)

Resumable (per-cell JSON under h_causal_vs_not/cells/). Run after the PPO/causal
training fixes (PPO_IMPLEMENTATION_REVIEW.md), so this also re-validates env h on
the fixed code.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SuiteConfig          # noqa: E402
from run_suite import run_suite         # noqa: E402

cfg = SuiteConfig(
    envs=["h_exclusive_choice_disjoint"],
    methods=["ppo_clip", "causal_td0"],
    seeds=3,
    epochs=15,
    episodes_per_epoch=12,
    test_freq=3,
    length=10,
    output_dir=Path("h_causal_vs_not"),
)

if __name__ == "__main__":
    run_suite(cfg)