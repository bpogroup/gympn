r"""Seeded head-to-head on `j_mixed_rework`: ppo_clip vs lrq2 vs ls_hca.

`j_mixed_rework` is the first env in the suite with BOTH properties LS-HCA
needs (`_diag_ls_hca_step0.py j_mixed_rework`):
  - a PURE floor: 24.8% of credit mass is exact lineage credit, 3958/5773
    decisions carrying it (every other env is 0%, except i_mixed_credit's 21%);
  - real contested signal: I(A;Z|X) = 0.0303 nats/record, p=0.005 -- 44x
    i_mixed_credit's and 1.5x the best rework loop, with 23.2% of decisions
    warranting |factor| > 0.2 and 1.23% > 1.0.
It also has ~2x i_mixed_credit's headroom (random 25.50 vs optimum 30.00).

Three arms, because two different claims are on trial and they need different
controls:
  - `ppo_clip`  : no causal credit at all -- does credit assignment help here?
  - `lrq2`      : full lineage return per decision, no PURE/CONTESTED split --
                  the arm LS-HCA has to beat to justify the split at all.
  - `ls_hca`    : PURE credited exactly + hindsight correction on CONTESTED,
                  now with the consistent P(z|x,a) parameterization (null and
                  marginal consistency exact by construction; applied |factor|
                  distribution matches the measured ideal to 1.00x at p90).

5 seeds: enough to see a real effect on a deterministic-eval env, not enough
to call a small one. Judge on greedy (deterministic) eval, per the
parallel-disjoint-low-headroom convention.

Resumable: cells already under suite_results_ls_hca_jrework/cells/ are skipped.

Run: python run_ls_hca_jrework.py [num_workers]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SuiteConfig  # noqa: E402
from run_suite import run_suite, _default_workers  # noqa: E402


def main(num_workers=None):
    cfg = SuiteConfig()
    cfg.envs = ["j_mixed_rework"]
    cfg.methods = ["ppo_clip", "lrq2", "ls_hca"]
    cfg.seeds = 5
    cfg.test_freq = 2      # eval is exact on this env, so a finer grid is free
    cfg.output_dir = Path("suite_results_ls_hca_jrework")
    if num_workers is None:
        num_workers = _default_workers()
    print(f"[jrework] {cfg.methods} x {cfg.seeds} seeds x {cfg.epochs} epochs "
          f"x {cfg.episodes_per_epoch} eps/epoch = "
          f"{len(cfg.methods) * cfg.seeds} cells -> {cfg.output_dir} "
          f"(num_workers={num_workers})")
    run_suite(cfg, num_workers=num_workers)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(nw)
