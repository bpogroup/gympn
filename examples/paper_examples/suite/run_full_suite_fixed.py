"""Full experimental suite on the FIXED code (strict horizon, PPO fixes, adv-norm OFF).

Canonical grid: 8 envs (a-h) x {ppo_clip (no causal), causal_td0 (flow_dag, beta=0.2)}
x 10 seeds, 30 epochs each. Fresh output dir because prior cached results predate the
simulator/training fixes (horizon drain removal, value_updates/vf_coeff/value-target
fixes, advantage normalization turned off by default).

Resumable: per-cell JSON under suite_results_fixed/cells/ (re-run to continue).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SuiteConfig          # noqa: E402
from run_suite import run_suite         # noqa: E402

# Defaults: seeds=10, epochs=30, episodes_per_epoch=20, test_freq=5, length=10,
# causal_beta=0.2, causal_self_credit=0.5, allow_postpone=True. normalize_advantages
# is OFF (new Agent default; suite does not override it).
cfg = SuiteConfig(output_dir=Path("suite_results_fixed"))

if __name__ == "__main__":
    run_suite(cfg)