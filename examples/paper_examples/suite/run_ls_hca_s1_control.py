r"""Minimal s1 control: does the FIXED ls_hca estimator behave differently?

Why s1 specifically. `j_mixed_rework` has the mechanism LS-HCA needs (24.8%
PURE mass + I(A;Z|X)=0.0303) but `lrq2` saturates it -- 5/5 seeds at the exact
optimum 30.0, sd 0 -- so nothing can be measured above the baseline there. s1
is the opposite: `lrq2` plateaus at 0.825 of the anchor (12.185+-0.372 against
14.775, a hard plateau given that sd), leaving 2.59 raw points of real
headroom, but s1 has PURE=set() so the exact lineage term is identically zero
and 100% of the credit is the hindsight correction.

That makes s1 the sharpest available test of the SCOPE hypothesis: LS-HCA
needs a PURE floor to stand on. Two competing predictions, and the run
distinguishes them:
  - scope hypothesis TRUE  -> the fixed estimator does WORSE than the old one.
    Making the correction honest shrinks it to its true (small) size, and on
    s1 that IS the entire learning signal. Weak supporting evidence already:
    with the consistent estimator, s1 records/epoch fell 715->343 over 10
    epochs, where the old independent-fit estimator rose 715->1000.
  - scope hypothesis FALSE -> it does BETTER. The old estimator was applying
    ~3.4x the true correction with ~85% of the magnitude being pi-vs-h
    mismatch, so removing that noise should help even with no PURE floor.

The prior ls_hca-on-s1 number (0.725 norm, 10 seeds, lost to lrq2 by a paired
-1.47, p=.087) predates ALL of this session's fixes -- flat-fallback removal,
residual parameterization, and the consistent P(z|x,a) form -- so it cannot
serve as the comparison point. Hence a fresh output dir: writing into
suite_results_ls_hca_s1/ would have silently SKIPPED these cells as already
done, and compared against the stale estimator's numbers.

Minimal: 5 seeds, 2 arms. `lrq2` is re-run rather than reused from the earlier
10-seed run so both arms come from the same code state -- this session edited
`run_episode`, and although every edit is gated behind `ls_hca_on`, a control
that shares no code state with its treatment is not a control.

Run: python run_ls_hca_s1_control.py [num_workers]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import stoch_config  # noqa: E402
from run_suite import run_suite, _default_workers  # noqa: E402


def main(num_workers=None):
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["lrq2", "ls_hca"]
    cfg.seeds = 5
    cfg.output_dir = Path("suite_results_ls_hca_s1_fixed")
    if num_workers is None:
        num_workers = _default_workers()
    print(f"[s1_control] {cfg.methods} x {cfg.seeds} seeds x {cfg.epochs} epochs "
          f"= {len(cfg.methods) * cfg.seeds} cells -> {cfg.output_dir} "
          f"(num_workers={num_workers})")
    run_suite(cfg, num_workers=num_workers)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)
