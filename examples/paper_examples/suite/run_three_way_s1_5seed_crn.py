r"""ppo_clip vs cgae vs cfgae on s1, 5 paired seeds, COMMON RANDOM NUMBERS.

Same design as run_three_way_s1_5seed.py, with cfg.eval_seed set so every
greedy eval point -- every epoch, every seed, every method -- scores on one
fixed scenario set (gympn.agents.Agent.test_in_train).

Why retrain rather than re-score. The level question was already answered
test-only by rescore_crn.py (load each best_policy.pth, score under CRN). What
that CANNOT answer is anything defined on the per-EPOCH greedy curve --
mean_greedy and greedy_drift -- because only the best checkpoint is saved, not
per-epoch policies. greedy_drift is both the metric most contaminated by
scenario noise (a max over ~15 points each +-0.231 SD, inflating drift by ~0.40
even for a genuinely flat policy) and the only near-significant effect in the
non-CRN run (cgae -0.94 vs ppo, p=0.071). This run is what decides it.

NOTHING is reused from the non-CRN run: all 15 cells train here. Without
eval_seed the eval CONSUMES the training stream (20 episodes of draws per eval
point), so a CRN run's trajectory diverges from a non-CRN run of the same seed
from the first eval onward -- mixing cells across the two would reintroduce
exactly the confound this run exists to remove. Hence a separate output dir.

Baseline (non-CRN, suite_results_three_way_s1), greedy_final / mean_greedy /
drift:  ppo 13.18 / 12.93 / 1.20   cgae 13.92 / 13.29 / 0.26
        cfgae 13.63 / 13.01 / 0.57
"""
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EVAL_SEED = 555_000        # same fixed scenario set rescore_crn.py used


def main():
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["ppo_clip", "cgae", "cfgae"]
    cfg.seeds = 5
    cfg.eval_seed = EVAL_SEED
    cfg.output_dir = HERE / "suite_results_three_way_s1_crn"

    print(f"[3way-crn] {cfg.methods} x seeds 0-4 x {cfg.epochs} epochs, "
          f"eval_seed={EVAL_SEED} -> {cfg.output_dir}")
    # __main__ guard is required for num_workers>1: Windows spawns children,
    # which re-import this file; without it they re-enter run_suite and the
    # pool dies (BrokenProcessPool).
    run_suite(cfg, num_workers=2)


if __name__ == '__main__':
    main()
