"""X10: G1 forked counterfactual preferences (method 'cfp') — s1 probe.

cfp = lcv0 base (plain SMDP-GAE PPO, beta=0.5) + G1: per-decision simulator
forks (taken vs highest-prob alternative), 3 CRN reps, SNR gate 2xSE,
pairwise logistic aux loss on the policy logits, coefficient annealed
linearly to 0 (floor = lcv0 by construction). No env-specific code paths:
bindings from actions_dict, alternative from the policy's own logits,
greedy continuation, lookahead-6 truncation with critic value tail.
See CAUSAL_LINEAGE_RETHINK.md §7.2 (G1) and gympn/counterfactual.py.

Registered predictions (written before the run, 2026-07-18):
  P1 (gain):  s1 norm_final >= 0.85 — above every trained method to date
              (ppo 0.76, lcv0 0.76, lcv 0.72, lrq2 0.28). The R4 diagnostic
              showed 91% of match/cross states are decodable by exactly the
              paired signal cfp consumes; the gate should transmit the real
              gaps and reject the clustered ones.
  P2 (floor): entropy does NOT collapse early (>= 0.35 at epoch 15) and the
              final policy is no worse than lcv0 (0.76) on any seed pair —
              the anneal must leave plain SMDP-PPO behind.
  P3 (cost):  wall time <= 2x the lcv0 s1 cell (~13.5 min).
Falsifiers: s1 ~ lcv0 with healthy pref counts (>= 20/epoch) => the
preference interface doesn't transmit at this coefficient — inspect
cf_loss/gap stats before touching hyperparameters. s1 < lcv0 => the aux
pass escapes the KL brake destructively; halve cf_coef, don't add tricks.

Run: python run_cfp_s1.py [workers]
"""
import sys

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["cfp"]
    cfg.seeds = 10
    run_suite(cfg, num_workers=workers)