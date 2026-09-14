r"""First head-to-head of `ls_hca` against plain PPO on `i_mixed_credit`.

`i_mixed_credit` is the only env in the suite whose reward-types split
non-degenerately into LS-HCA's PURE and CONTESTED sets (`_diag_pure_scan.py`):
PURE={done1}/{done2} per stream, CONTESTED={doneFinal}, measured at 21% / 79%
of credit mass over a real run. Everywhere else the split collapses to a
trivial corner -- 9 envs are 100% CONTESTED (LS-HCA degenerates to plain HCA,
zero exact term) and 2 are 100% PURE (it degenerates to exact lineage credit,
which lrq2 already computes). So this is the first run in which LS-HCA's
actual premise -- lineage credits part of the return exactly, hindsight only
patches the shared remainder -- is exercised at all.

Run at 1 seed: this is a directional read on a new env, not a result. The
suite's own standard is >= 5 seeds judged on deterministic eval
(parallel-disjoint-low-headroom memory), and i_mixed_credit inherits its
parent d_parallel_disjoint's narrow headroom (random 27.65 vs optimum 30.0),
so a single seed cannot separate the arms. Treat the output as "does this
train sanely and roughly where does it land", then decide on a seeded run.

Reuses run_suite.py's resumable/parallel/baseline-cached machinery like every
other real run here. Safe to Ctrl-C and re-run; finished cells under
suite_results_ls_hca_imixed/cells/ are skipped.

Run: python run_ls_hca_imixed.py [num_workers]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SuiteConfig  # noqa: E402
from run_suite import run_suite, _default_workers  # noqa: E402


def main(num_workers=None):
    cfg = SuiteConfig()
    cfg.envs = ["i_mixed_credit"]
    cfg.methods = ["ppo_clip", "ls_hca"]
    cfg.seeds = 1
    cfg.test_freq = 2      # finer eval grid; eval is exact on this env
    cfg.output_dir = Path("suite_results_ls_hca_imixed")
    if num_workers is None:
        num_workers = _default_workers()
    print(f"[ls_hca_imixed] {cfg.methods} x {cfg.seeds} seed x {cfg.epochs} epochs "
          f"x {cfg.episodes_per_epoch} eps/epoch -> {cfg.output_dir} "
          f"(num_workers={num_workers})")
    run_suite(cfg, num_workers=num_workers)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)
