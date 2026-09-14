r"""Does cgae_cflow's SAFETY property generalize beyond s1? s3 + s4, 5 CRN seeds.

The paper's central claim about the convex normalization is a safety claim: the
predecessor-normalized form (`cgae_flow`) leaves the bootstrap coefficient
R(d) = sum_{s in succ(d)} w(d->s) unbounded, and where fan-out exists that
collapses the estimator. On s1 this is established and significant --
cgae_flow 0.166 vs ppo 0.668 (0W/5L, p=0.025), cgae_cflow 0.792 (null vs ppo,
p=0.373). But ONE environment is not a safety claim.

The fan-out table (`_diag_fanout_table.py`) shows s1 is not an oddity: it is
representative of a whole quadrant of the suite -- K=1 (single reward-bearing
component, so there is no factorization to exploit and any effect is pure
aggregation-rule) combined with substantial fan-out (so the aggregation rule
bites). s3 and s4 are the sharpest remaining members of that quadrant:

    env                     K     fan-out   >1 succ   max|flow - cflow|
    s1_stoch_sequence      1.00    1.387     32.0%          1.32
    s3_stoch_mixed         1.00    1.425     27.5%         13.3
    s4_stoch_mixed_rework  1.00    1.443     27.1%         11.7

i.e. ~10x s1's credit divergence at the same single-blob component structure.

WHY NOT THE a-j DETERMINISTIC ARCHETYPES, despite f_loop_disjoint having the
highest fan-out in the suite (1.803, 60.1% multi-successor): config.py's own
stoch_config docstring records that "the deterministic a-h grid can no longer
discriminate above LRQ" -- it is saturated. A tie there would be evidence about
the benchmark, not about the estimator. The stochastic tier exists precisely
because it still discriminates.

PREDICTION, registered before running: cgae_flow degrades on both (the defect
is a function of fan-out, which is comparable to s1's), cgae_cflow stays at
parity with ppo (K=1, so nothing to gain -- the claim is safety, not
superiority). If cgae_flow does NOT degrade here, the s1 collapse is
env-specific and the safety claim needs rewording.

Fresh output dir: the stored suite_results_paper / suite_results_stoch cells for
these envs were run WITHOUT eval_seed, and mixing CRN with non-CRN cells
reintroduces exactly the scenario-noise confound CRN removes.

Run: python run_s3s4_safety.py [workers]
"""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EVAL_SEED = 555_000          # same fixed scenario set as the s1 / ncopies runs


def main(workers=4):
    cfg = stoch_config()
    cfg.envs = ["s3_stoch_mixed", "s4_stoch_mixed_rework"]
    cfg.methods = ["ppo_clip", "cgae_cflow", "cgae_flow"]
    cfg.seeds = 5
    cfg.eval_seed = EVAL_SEED
    cfg.output_dir = HERE / "suite_results_s3s4_safety"

    print(f"[safety] {cfg.envs} x {cfg.methods} x seeds 0-4 x {cfg.epochs} epochs, "
          f"eval_seed={EVAL_SEED} -> {cfg.output_dir}", flush=True)
    # __main__ guard required: Windows spawns workers, which re-import this file.
    run_suite(cfg, num_workers=workers)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4)
