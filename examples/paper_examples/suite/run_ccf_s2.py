r"""ccf / s_ccf on s2_stoch_scaled -- the most promising untested cell.

Why s2. The K-diagnostic (_diag_k_all_envs.py) gives s2 K_realized = 6 while
K_static = 1. ccf partitions on the REALIZED lineage union-find, so it has
room to act; s_ccf partitions statically, so Remark R1 says it should
degenerate to mc_q exactly. Running both makes that a prediction, not an
assumption -- s_ccf is the control that says whether any difference is the
realized decomposition or just noise.

s2 is also the env with the largest causal-over-PPO margin on record
(suite_results_stoch, 10 seeds): ppo_clip 79.32, lrq2 87.11 (+7.8),
mc_q 89.59 (+10.3). Headroom exists and neither ccf nor s_ccf has ever run
here.

Same stoch_config protocol and seeds as that run, so the new cells pair
directly against the stored ppo_clip / mc_q / lrq2 cells rather than needing
them re-run.

Seeds 0-4 (n=5, 2026-08-05) gave ccf 90.36+-0.29 vs mc_q 89.67+-0.65 (+0.69,
4W/1L, p=.087) with s_ccf at 88.64 -- the K_static=1 control behaving exactly
as Remark R1 predicts. Now at 10 seeds to see whether that firms up. The
suite is resumable, so the existing seeds 0-4 cells are SKIPPED and only
5-9 are trained.

Run: python run_ccf_s2.py [num_workers]
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite, _default_workers

def main(nw=None):
    cfg = stoch_config()
    cfg.envs = ["s2_stoch_scaled"]
    cfg.methods = ["ccf", "s_ccf"]
    cfg.seeds = 10
    cfg.output_dir = Path("suite_results_ccf_s2")
    if nw is None:
        nw = _default_workers()
    print(f"[ccf_s2] {cfg.methods} x {cfg.seeds} seeds x {cfg.epochs} epochs "
          f"= {len(cfg.methods)*cfg.seeds} cells -> {cfg.output_dir} (workers={nw})")
    run_suite(cfg, num_workers=nw)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
