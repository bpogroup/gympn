"""X9: LVA (lineage as auxiliary critic target) across BOTH tiers.

Sequential: stochastic tier (s1/s2/s3 x 10 seeds -> suite_results_stoch)
first, then the deterministic 8-topology grid (x 10 seeds ->
suite_results_paper, braked protocol). Both resumable; existing cells of
other methods untouched.

Registered predictions (PAPER_PLAN_LCV.md X9, written before the run):
  P1 (floor):   lva ~ lcv0 everywhere it doesn't help; in particular s1
                (the foreclosure/discrimination env) must NOT collapse --
                value-side lineage consumption cannot bias the policy
                gradient, unlike lrq2's 0.28. Falsifier: lva << lcv0
                anywhere (representation interference).
  P2 (gain):    faster convergence than lcv0 where lrq2's speed edge was
                largest (s2: lrq2 3.6 ep vs lcv0 16.4; s3: 6.8 vs 15.2) --
                if the lineage's sample-efficiency value is extractable
                through the critic, lva should close part of that gap.
                Secondary: lower drift on b/f/h via a sharper critic.
  P3 (honest):  lva ~ lcv0 everywhere => representation-level consumption
                is also insufficient at this budget => the lineage's
                extractable value here is the discount + policy-side
                credit only, which strengthens the trade-off framing.

Run: python run_lva.py [workers]
"""
import sys

from config import stoch_config, paper_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    cfg = stoch_config()
    cfg.methods = ["lva"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)

    cfg2 = paper_config()
    cfg2.methods = ["lva"]
    run_suite(cfg2, num_workers=workers)