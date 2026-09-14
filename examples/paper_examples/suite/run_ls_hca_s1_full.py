r"""Full-budget stoch_config() head-to-head of 'lrq2' (fork-free lineage
baseline, no correction) vs 'ls_hca' (same baseline + the now-live fork-free
hindsight correction on instance-level contested reward-types -- see
causal-stability-suite memory, 2026-07-31: the classifier fix that made s1's
`done2` CONTESTED instead of PURE, so this is the first real-budget test of
whether the correction actually lifts lrq2's floor here, or whether the
lagged/state-unconditioned hhat table is too coarse).

Reuses run_suite.py's resumable, parallel, baseline-cached infrastructure
directly (same as every other real run in this project) -- just restricts
stoch_config() to s1 alone and these two methods. Safe to Ctrl-C / re-run;
cells already written under suite_results_ls_hca_s1/cells/ are skipped.

Run: python run_ls_hca_s1_full.py [num_workers]
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
    cfg.output_dir = Path("suite_results_ls_hca_s1")
    if num_workers is None:
        num_workers = _default_workers()
    print(f"[ls_hca_s1] {cfg.methods} x {cfg.seeds} seeds x {cfg.epochs} epochs "
          f"x {cfg.episodes_per_epoch} eps/epoch -> {cfg.output_dir} "
          f"(num_workers={num_workers})")
    run_suite(cfg, num_workers=num_workers)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)