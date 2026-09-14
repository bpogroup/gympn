r"""Single-seed smoke of the ratio-form ls_hca on s1.

The difference form collapsed here: 1.45 +- 2.33 vs lrq2's 12.02 (paired
-10.57, 0W/5L, p=.001), 3/5 seeds at exactly 0.0 -- below random (9.85).
The ratio form's null is exactly lrq2, so the check is simply: does it stop
collapsing? Prediction is NEAR lrq2 (~12), not above it -- s1's measured
signal is a median |factor| of 0.072, so w sits close to 1.

One seed, ls_hca only. lrq2 is not re-run: the 5-seed control on this exact
config gave 11.8/11.75/11.0/12.7/12.85 (mean 12.02) and the code touched
since is all behind ls_hca branches. Fresh output dir so the earlier
difference-form cells are not silently skipped as already-done.

Run: python run_ls_hca_s1_ratio1.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import stoch_config  # noqa: E402
from run_suite import run_suite  # noqa: E402

cfg = stoch_config()
cfg.envs = ["s1_stoch_sequence"]
cfg.methods = ["ls_hca"]
cfg.seeds = 1
cfg.output_dir = Path("suite_results_ls_hca_s1_ratio")
print(f"[s1_ratio] ls_hca x 1 seed x {cfg.epochs} epochs -> {cfg.output_dir}")
run_suite(cfg, num_workers=1)
