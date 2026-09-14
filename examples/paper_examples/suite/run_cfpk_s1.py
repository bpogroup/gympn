"""X11: cfpk = cfp with CONSTANT preference coefficient (anneal removed) — s1.

The X10 falsifier follow-up (CAUSAL_LINEAGE_RETHINK.md §7.2 G1 result
block). X10 showed the preference interface transmits (winners ranked ~1
logit up by the end) yet finals sat at the 0.72-0.76 band every
intact-floor method occupies. This run keeps the pairwise loss at full
strength through the last epoch — the floor is knowingly sacrificed — to
test the CEILING HYPOTHESIS against the strongest version of G1.

Registered predictions (written before the run, 2026-07-19):
  P1 (ceiling): cfpk norm_final lands in the same 0.72-0.76 band
      (specifically: not significantly above lcv0 0.756, paired by seed).
      This is the EXPECTED outcome and confirms the ceiling: constant
      counterfactual pressure, learned to satisfaction, still cannot move
      s1 finals => the binding constraint on s1 is optimization/
      exploration, not credit signal quality. Paper reframe follows.
  P2 (alt — anneal was the limiter): cfpk >= 0.85. Then the anneal
      schedule, not the ceiling, capped X10 — G1 tuning (floor-preserving
      late schedules) becomes the priority and the ceiling claim is dead.
  P3 (destructive tail risk): cfpk < lcv0 by more than noise (constant
      aux pressure outside the KL brake degrades the policy late). Also
      informative: bounds how much non-PPO pressure the policy tolerates.

Run: python run_cfpk_s1.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["cfpk"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)