"""Item 2 of the LCV validation list: the mirage-standard measurement.

Runs one instrumented LCV cell on s1 and one on s3 (fresh dir, full budget)
and reports the per-epoch adaptive coefficient c_hat and the fractional
advantage-variance reduction the CV achieved — the quantities the CV theorem
is actually about (per Tucker et al.'s critique, CV papers must report
variance, not just returns).

Run: python run_lcv_diag.py
"""
import json
from pathlib import Path

from config import stoch_config
from run_suite import run_suite

if __name__ == "__main__":
    cfg = stoch_config()
    cfg.envs = ["s1_stoch_sequence", "s3_stoch_mixed"]
    cfg.methods = ["lcv"]
    cfg.seeds = 1
    cfg.output_dir = Path("suite_results_lcv_diag")
    run_suite(cfg, num_workers=2)

    for env in cfg.envs:
        p = Path(cfg.output_dir, "cells", f"{env}__lcv__s0.json")
        c = json.loads(p.read_text())
        coef = c.get("cv_coef_curve", [])
        vred = c.get("cv_var_reduction_curve", [])
        print(f"\n[lcv-diag] {env}")
        print(f"  c_hat per epoch     : {[round(x, 3) for x in coef]}")
        print(f"  var reduction (frac): {[round(x, 3) for x in vred]}")
        if coef:
            print(f"  c_hat mean {sum(coef)/len(coef):.3f}   "
                  f"var-reduction mean {sum(vred)/len(vred):.3f}")
