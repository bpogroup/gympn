r"""Is the N=2 cgae_cflow deficit a real deficiency, or an early-training transient?

At ncopies N=2 with the standard 15-epoch config, cgae_cflow scores 0.401 vs
cgae_flow's 0.837 (4W/14L, p=0.0043). But three measurements say the estimators
are barely different there:

  * credit magnitude:  mean|Q| = 2.22 (cflow) vs 2.23 (flow), SD 1.12 vs 1.10
  * on-policy return:  sampled gain +1.41 vs +1.35 (of 6.73 headroom)
  * entropy:           1.00 -> 0.92 vs 1.00 -> 0.93

i.e. a ~1-5% perturbation of the credit vector, with near-identical policies by
every on-policy measure, producing a 0.44 swing in the greedy metric. The greedy
outcomes are moreover BIMODAL -- ~0.90 (found the good rule) or ~0.13 / -0.72
(did not) -- with essentially nothing in between.

That is the signature of a BIFURCATION rather than a quality difference: at 15
epochs the policy has barely moved (entropy 0.92, only 19-21% of the headroom
captured), so the run is still on a knife-edge between two attractors and a tiny
gradient perturbation decides which side it falls on.

If that reading is right, training longer lets both arms reach the good
attractor and the gap CLOSES. If cgae_cflow is genuinely deficient, the gap
PERSISTS or widens.

Either answer matters: a closing gap means N=2-at-15-epochs is not a valid
discriminator between estimators and must not be reported as one; a persisting
gap means cgae_cflow has a real weakness for cgae_cap to fix.

Replicates run_ncopies_three_way_crn.main() rather than monkeypatching it,
because that main() hardcodes cfg.epochs=15 after building the config, and the
config is pickled to spawned workers (so a wrapper object would not survive).

Run: python run_n2_epochs.py [epochs] [seeds] [workers]
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from config import stoch_config  # noqa: E402
from run_suite import _threads_per_worker  # noqa: E402
import run_ncopies_three_way_crn as R  # noqa: E402

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
N = int(sys.argv[4]) if len(sys.argv) > 4 else 2
METHODS = tuple((sys.argv[5].split(",") if len(sys.argv) > 5
                 else ["cgae_flow", "cgae_cflow", "cgae_cap"]))
OUTDIR = "suite_results_n%d_ep%d" % (N, EPOCHS)


def main():
    cfg = stoch_config()
    cfg.epochs = EPOCHS
    cfg.episodes_per_epoch = 8
    cfg.test_freq = 3

    out = Path(OUTDIR)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    logdir = str(out / "train")

    a, b = R.crn_precheck(N)
    print("[n2-ep] CRN pre-check N=%d: non-causal %.4f | causal %.4f -> %s"
          % (N, a, b, "MATCH" if a == b else "MISMATCH"), flush=True)

    bp = out / ("baselines_N%d.json" % N)
    if bp.exists():
        base = json.loads(bp.read_text())
    else:
        base = R._baselines(N)
        bp.write_text(json.dumps(base))
    r, h = base["random_mean"], base["heuristic_mean"]
    print("[n2-ep] N=%d random=%.2f heuristic=%.2f | %d epochs, %d seeds"
          % (N, r, h, EPOCHS, SEEDS), flush=True)

    pending = [(N, m, s) for m in METHODS for s in range(SEEDS)
               if not (out / "cells" / ("N%d__%s__s%d.json" % (N, m, s))).exists()]
    print("[n2-ep] methods=%s | %d cells, %d workers" % (str(METHODS), len(pending), WORKERS), flush=True)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = mp.get_context("spawn")
    tpw = _threads_per_worker(WORKERS)
    payloads = [(n, m, s, cfg, logdir, base, tpw) for (n, m, s) in pending]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as ex:
        futs = [ex.submit(R._worker, p) for p in payloads]
        for fut in as_completed(futs):
            n, m, s, res, err = fut.result()
            if err:
                print("[n2-ep] !!! %s_s%d FAILED:\n%s" % (m, s, err), flush=True)
                continue
            (out / "cells" / ("N%d__%s__s%d.json" % (n, m, s))).write_text(json.dumps(res))
            print("[n2-ep] <<< %s_s%d greedy_final=%s (%.1f min; %.1f total)"
                  % (m, s, res.get("greedy_final"), res["minutes"],
                     (time.time() - t0) / 60.0), flush=True)

    print("\n[n2-ep] === normalized final, N=2 @ %d epochs ===" % EPOCHS, flush=True)
    for m in METHODS:
        vals = []
        for s in range(SEEDS):
            c = out / "cells" / ("N%d__%s__s%d.json" % (N, m, s))
            if c.exists():
                mm = json.loads(c.read_text())
                if mm.get("greedy_final") is not None:
                    vals.append((mm["greedy_final"] - r) / (h - r))
        if vals:
            print("  %-11s %6.3f +- %.3f  (n=%d, collapses<=0.25: %d)  %s"
                  % (m, float(np.mean(vals)), float(np.std(vals)), len(vals),
                     sum(1 for x in vals if x <= 0.25),
                     [round(x, 2) for x in vals]), flush=True)


if __name__ == '__main__':
    main()
