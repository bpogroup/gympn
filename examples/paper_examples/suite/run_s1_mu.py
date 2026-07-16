"""E4 probe: does the mu-hedge fix s1?

s1 diagnosis (2026-07-13): LRQ's credits ORDER the actions correctly (the
induced ordering scores 14.80 = anchor as a scripted policy) but the trained
policy fails to execute it — the within-stage discrimination gaps (~0.06-0.26
Q units) are comparable to per-sample noise and the policy over-commits early
(entropy 0.14). Hypothesis: mixing in the coarse, large-gap PPO/GAE advantage
(A = (1-mu)*A_LRQ + mu*A_GAE) supplies the missing discrimination signal.

This runs lrq2 with causal_mu on s1 only (10 seeds) into a separate results
dir; compare against suite_results_stoch (lrq2 mu=0: final* 0.28; ppo: 0.76).

Run: python run_s1_mu.py [mu] [workers]
"""
import json
import sys
from pathlib import Path

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    mu = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence"]
    cfg.methods = ["lrq2"]
    cfg.seeds = 10
    cfg.causal_mu = mu
    cfg.output_dir = Path(f"suite_results_s1_mu{str(mu).replace('.', '')}")
    run_suite(cfg, num_workers=workers)

    # Quick side-by-side against the mu=0 tier cells.
    def finals(d, method):
        out = []
        for p in sorted(Path(d, "cells").glob(f"s1_stoch_sequence__{method}__*.json")):
            c = json.loads(p.read_text())
            bl = c["baselines"]
            rng = bl["heuristic_mean"] - bl["random_mean"]
            out.append((c["greedy_final"] - bl["random_mean"]) / rng)
        return out

    new = finals(cfg.output_dir, "lrq2")
    old = finals("suite_results_stoch", "lrq2")
    ppo = finals("suite_results_stoch", "ppo_clip")
    fmt = lambda xs: f"mean {sum(xs)/len(xs):.2f}  per-seed {[round(x,2) for x in xs]}"
    print(f"\n[s1-mu] lrq2 mu={mu}: {fmt(new)}")
    print(f"[s1-mu] lrq2 mu=0.0: {fmt(old)}")
    print(f"[s1-mu] ppo_clip   : {fmt(ppo)}")
