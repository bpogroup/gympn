r"""cgae_dag on s1, 5 paired seeds, CRN -- the second negative control.

Same env, seeds, eval_seed and output dir as run_three_way_s1_5seed_crn.py, so
the cells drop straight into that comparison. Only the cgae_dag arm is listed,
which lets this run alongside a cgae_cflow run already in flight: run_suite
writes one JSON per (env, method, seed), so the two never touch the same file
and the cached baselines are reused.

Why cgae_dag needs its own s1 number. It fixes a DIFFERENT defect from
cgae_cflow. cgae_cflow repairs the unanchored bootstrap -- the row sum
R(d) = sum_{s in succ(d)} w(d->s), which Proposition 3's predecessor
normalization leaves unconstrained (|R-1| > 0.25 on 55.3% of s1 decisions) --
and that is the collapse cause, since the mean variant pins the coefficient to
1 by construction and does not collapse. cgae_dag additionally fixes the fan-in
dilution that cgae, cgae_flow and cgae_cflow all share: Proposition 3(i) is a
statement about the descendant SET, but every weighted variant computes it as a
sum over PATHS, and on s1 42.1% of (d, j) pairs have >=2 distinct causal paths
(up to 52) while 54.1% of successors have in-degree 1. Measured P3 ratio
against an ideal of 1.000 -- flow 0.577, mean 0.602, cflow 0.626, decaying with
causal depth to 0.329; cgae_dag is exactly 1.000 by construction.

The prediction is therefore NOT that cgae_dag beats cgae_cflow here. s1 is a
negative control: with one reward-bearing component holding 100% of the reward
mass there is no factorization to exploit, so the bar both must clear is
PARITY with ppo_clip (13.14) and cgae (13.05), against cgae_flow's 10.67.

Reference, 5 CRN seeds, greedy_final mean (random 9.85, heuristic 14.78):
    ppo_clip 13.14   cgae 13.05   cfgae 13.30   cgae_flow 10.67
"""
import os, sys
from pathlib import Path

# Keep this run's footprint small: a cgae_cflow run may still be using two
# workers x four torch threads on the same eight physical cores.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EVAL_SEED = 555_000


def main():
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["cgae_dag", "cgae_cflow2"]
    cfg.seeds = 5
    cfg.eval_seed = EVAL_SEED
    cfg.output_dir = HERE / "suite_results_three_way_s1_crn"

    print(f"[dag-s1] cgae_dag x seeds 0-4 x {cfg.epochs} epochs, "
          f"eval_seed={EVAL_SEED} -> {cfg.output_dir}")
    run_suite(cfg, num_workers=2)


if __name__ == '__main__':
    main()
