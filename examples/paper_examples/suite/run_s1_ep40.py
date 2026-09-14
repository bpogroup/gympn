r"""s1 at the 40-epoch budget, for protocol consistency with ncopies/multisite.

s1 was run at 30 epochs (stoch_config default) and, unlike ncopies-at-15, it IS
genuinely converged there -- final entropy 0.15-0.56 against ncopies' 0.92. So
this is a consistency/defensibility run, not a correction: multisite already
used 40 epochs and ncopies is being redone at 40, so s1 should match.

SEPARATE OUTPUT DIR is mandatory. Cells trained for 30 epochs and 40 epochs are
not comparable -- the greedy curve has a different length and the final point a
different meaning -- so they must never be pooled. suite_results_three_way_s1_crn
keeps the 30-epoch cells; this writes to suite_results_s1_ep40.

Reference, 30 epochs, 5 CRN seeds (random 9.85, heuristic 14.775):
    cgae_cflow 0.792 | cfgae 0.701 | ppo 0.668 | cgae 0.650
    cgae_cap 0.692 | cgae_dag 0.386 | cgae_flow 0.166 | cgae_cflow2 0.106

The result that must survive is the SAFETY contrast: cgae_flow collapses
(0.166, 0W/5L vs ppo, p=0.025) while cgae_cflow is null against ppo (p=0.373).
If longer training rescues cgae_flow here the way it rescued cgae_cflow at
ncopies N=2, the paper's central negative result changes and §6c needs rewriting.

Run: python run_s1_ep40.py [epochs] [workers]
"""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import stoch_config
from run_suite import run_suite

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
EVAL_SEED = 555_000
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5


def main():
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    # ccf and mc_q added 2026-08-17: ccf is the main prior-work comparator and
    # was missing from the negative control, so the paper could not state that it
    # is null at K=1 too. mc_q is the LINEAGE ablation and existed on multisite
    # only, which made the "the gain is the decomposition, not the discounting"
    # argument rest on one environment.
    cfg.methods = ["ppo_clip", "cgae", "cfgae", "cgae_flow", "cgae_cflow",
                   "cgae_dag", "cgae_cap", "ccf", "mc_q"]
    cfg.seeds = SEEDS
    cfg.epochs = EPOCHS
    cfg.eval_seed = EVAL_SEED
    cfg.output_dir = HERE / ("suite_results_s1_ep%d" % EPOCHS)

    print("[s1-ep%d] %s x seeds 0-4 x %d epochs, eval_seed=%d -> %s"
          % (EPOCHS, cfg.methods, EPOCHS, EVAL_SEED, cfg.output_dir), flush=True)
    run_suite(cfg, num_workers=WORKERS)


if __name__ == '__main__':
    main()
