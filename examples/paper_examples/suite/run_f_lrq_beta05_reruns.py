"""Rerun the two post-convergence-unstable LRQ seeds on env f with causal_beta=0.5.

At beta=0.2 seeds 0 and 1 reached the optimum and then slipped (s0: greedy
1.0 -> 0.25; s1: collapse into the postpone-everything attractor, greedy -1.25).
Hypothesis: a stronger timing discount (beta=0.5 => postpone Q is ~39% below
acting instead of ~18% at tau=1) widens the margin against the postpone
attractor and holds the optimum. Everything else identical to the 5-seed run.

Run: python run_f_lrq_beta05_reruns.py
"""
import os
import sys
import json
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SuiteConfig  # noqa: E402
from run_suite import compute_baselines, _threads_per_worker  # noqa: E402
from run_f_lrq_vs_ppo import run_one, ENV, OUT, CELLS  # noqa: E402

SEEDS = (0, 1)
BETA = 0.5


def _worker(payload):
    import torch
    seed, cfg, threads = payload
    torch.set_num_threads(max(1, int(threads)))
    try:
        return (seed, run_one("lrq", seed, cfg), None)
    except Exception:
        import traceback
        return (seed, None, traceback.format_exc())


if __name__ == "__main__":
    os.makedirs(CELLS, exist_ok=True)
    cfg = SuiteConfig(causal_beta=BETA)
    base = compute_baselines(ENV, cfg)
    rnd, heu = base["random_mean"], base["heuristic_mean"]

    def norm(x):
        return (x - rnd) / (heu - rnd)

    pending = []
    for seed in SEEDS:
        cell = os.path.join(CELLS, f"lrq_beta05_s{seed}.json")
        if not os.path.exists(cell):
            pending.append(seed)
    print(f"[beta05] rerunning LRQ seeds {pending} with causal_beta={BETA}", flush=True)

    t0 = time.time()
    if pending:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        tpw = _threads_per_worker(len(pending))
        payloads = [(s, cfg, tpw) for s in pending]
        with ProcessPoolExecutor(max_workers=len(pending), mp_context=ctx) as ex:
            futs = [ex.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                seed, res, err = fut.result()
                if err is not None:
                    print(f"[beta05] !!! s{seed} FAILED:\n{err}", flush=True)
                    continue
                with open(os.path.join(CELLS, f"lrq_beta05_s{seed}.json"), "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[beta05] <<< s{seed} done ({res['minutes']:.1f} min)", flush=True)

    # Side-by-side vs the beta=0.2 cells for the same seeds.
    n_test = cfg.epochs // cfg.test_freq
    greedy_epochs = [cfg.test_freq * (i + 1) for i in range(n_test)]
    print(f"\n[beta05] greedy eval epochs: {greedy_epochs}", flush=True)
    for seed in SEEDS:
        old = json.loads(open(os.path.join(CELLS, f"lrq_s{seed}.json")).read())
        new = json.loads(open(os.path.join(CELLS, f"lrq_beta05_s{seed}.json")).read())
        for tag, r in (("beta=0.2", old), ("beta=0.5", new)):
            g = [round(norm(v), 2) for v in r["greedy"][:n_test]]
            s_tail = [round(norm(v), 2) for v in r["sampled"][-8:]]
            print(f"[beta05] s{seed} {tag}: greedy={g}  sampled_tail={s_tail}  "
                  f"ent_final={r['entropy'][-1]:.2f}", flush=True)
    print(f"\n[beta05] total {(time.time()-t0)/60:.1f} min", flush=True)
