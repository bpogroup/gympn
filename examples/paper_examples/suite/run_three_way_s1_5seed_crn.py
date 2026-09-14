r"""ppo_clip vs cgae vs cfgae vs cgae_flow vs cgae_cflow on s1, 5 seeds, CRN.

cgae_flow added 2026-08-11: it is the method Proposition 3(iii) covers and the
one the paper leads with, but the "costs nothing where there is no structure"
negative control had only ever been run for cgae. Existing cells are reused,
so only the cgae_flow arm trains.

cgae_cflow added 2026-08-12. That negative control FAILED: cgae_flow scored
0.166 normalized against 0.650 (cgae), 0.668 (ppo), 0.701 (cfgae) -- 4/5 seeds
pinned to greedy_final 10.3 against a random baseline of 9.85, and a
greedy_best ceiling of ~12.15 where every other arm reaches ~14. Diagnosis: the
recursion's bootstrap coefficient is the row sum R(d) = sum_{s in succ(d)}
w(d->s), which Proposition 3's predecessor-normalization does not constrain;
|R-1| > 0.25 on 55.3% of s1 decisions (24.1% on ncopies N=4), so the advantage
carries a spurious term proportional to V set by DAG topology rather than by
the action. cgae_cflow normalizes over successors, making the bootstrap a
convex combination.

cgae_dag added 2026-08-12 alongside it, for the SECOND defect -- one both
cgae and cgae_flow share, so it is not the collapse cause but does cap every
weighted variant. Proposition 3(i) is a statement about the descendant SET, yet
all weighted variants compute it as a sum over PATHS; on s1 42.1% of (d, j)
pairs have >=2 distinct causal paths (up to 52) while 54.1% of successors have
in-degree 1, so weight-1 over-counts diamonds and inflow-normalized weights
under-credit joint causation. Measured P3 ratio against an ideal of 1.000:
flow 0.577, mean 0.602, cflow 0.626, decaying with causal depth to 0.329.
cgae_dag uses the closure directly (ratio exactly 1.000 by construction) with a
convex k-step critic mixture, and reduces to textbook GAE on a chain.

Cells are resumable, so each added arm trains alone.

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
    # cgae_cap added 2026-08-13. s1 is the env where cgae_flow COLLAPSES
    # (0.166) via R>1 inflation on 24.0% of decisions. cap normalizes exactly
    # those, but keeps flow-like weights on the 37.5% with R<1 -- so this run
    # is the test of whether capping is SUFFICIENT, or whether the s1 collapse
    # also needs the R<1 tail renormalized (which is what cgae_cflow does, and
    # what N=2 shows is harmful there).
    cfg.methods = ["ppo_clip", "cgae", "cfgae", "cgae_flow", "cgae_cflow",
                   "cgae_dag", "cgae_cap"]
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
