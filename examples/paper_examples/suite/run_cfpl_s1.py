"""X13: does the LINEAGE actually buy anything inside the counterfactual? — s1.

cfpk wins by brute-force simulation and uses the lineage DAG for nothing.
This probe puts the lineage back in the one place it has a measured, proven
edge: lrq beat mc_q 28W/0L on the grid with an identical estimator and
discount, differing ONLY in restricting the summed reward to the decision's
causal descendants. That filters reward from CONCURRENT, causally-unrelated
activity — a different noise source than the one CRN pairing cancels (CRN
cancels noise SHARED by the two branches; it cannot touch contamination
from parallel events once the branches diverge).

cfpl applies exactly that test inside each forked branch: a branch's return
counts a reward only if the forked action's output tokens are an ancestor of
it in that branch's trace (the same walk _redistribute_lrq does over factual
episodes, keyed on one decision).

THREE-ARM ABLATION (all constant-coefficient, identical fork budget), so the
lineage effect is isolated rather than confounded with the value tail:
    cfpk = raw return-to-go + V(s) tail at truncation   (the X11 method)
    cfpn = raw return-to-go, NO tail                    (isolates the tail)
    cfpl = LINEAGE-restricted, no tail                  (isolates the lineage)
=> cfpl vs cfpn is the clean lineage ablation; cfpk vs cfpn prices the tail.
All three are freshly run here (into their own output dir) so they share one
code version and the new per-fork telemetry; X11's cfpk in
suite_results_stoch stays as an independent reproduction check.

Registered predictions (written before the run, 2026-07-20):
  M1 (MECHANISM, primary): cfpl's mean paired SE < cfpn's at equal fork
      budget, and its gate-pass rate is higher. This is the direct test of
      "the lineage removes concurrent-reward noise that CRN cannot", and it
      is the claim worth making regardless of where finals land — it prices
      the lineage in the units the extraction-gap story is about.
  M2 (efficiency): more preferences harvested per fork => more usable signal
      per unit of simulation, which is also the cheapest attack on cfpk's
      3.6x cost problem (fewer reps / shorter lookahead for the same gate).
  P1 (finals, weaker): cfpl >= cfpk's 0.849. Finals may simply saturate near
      the anchor, so this is secondary to M1 by design.
  FALSIFIER: SE unchanged (cfpl ~ cfpn) => at s1's horizons the concurrent
      contamination inside a 6-time-unit lookahead is negligible, the lineage
      genuinely adds nothing here, and THAT is the honest published finding —
      with a clean mechanism measurement behind it instead of an inference.

Run: python run_cfpl_s1.py [workers]
"""
import sys
from pathlib import Path

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["cfpl", "cfpn", "cfpk"]
    cfg.seeds = 10
    cfg.output_dir = Path("suite_results_cfpl")
    run_suite(cfg, num_workers=workers)