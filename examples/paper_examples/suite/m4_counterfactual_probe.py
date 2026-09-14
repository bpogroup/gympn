r"""DAG-replay counterfactual on M4 (abundant shared resource) -- the motif that
SEPARATES the counterfactual from s_ccf.

M4 (assembly_probe.make_two_chains): two independent chains share a resource with
2 units, so there is NO contention. Chain A's decision picks a high task (r_hi=5)
or low task (r_lo=2); chain B always runs its reward r_b, action-independent of A.
True effect of A's choice = r_hi - r_lo = 3; r_b is pure noise w.r.t. that choice.

What each method does here (established earlier):
  * mc_q  : unbiased, but dA's credit CARRIES r_b  -> full Var(r_b) in the advantage.
  * ccf   : unbiased AND factors r_b out (finest)  -> no Var(r_b).  [but ccf FLIPS on M2]
  * s_ccf : unbiased but CONSERVATIVE -- keeps r_b because A and B are statically
            resource-coupled (share the resource type)  -> carries Var(r_b).

The counterfactual computes dA's effect as G(A_hi) - G(A_lo) under CRN. r_b is
identical in both runs (abundant resource -> B is untouched by A's choice), so it
CANCELS -> the counterfactual factors r_b out (finest, like ccf) while staying
unbiased (like s_ccf). If so, it dominates: unbiased on M2 AND finest on M4,
resolving the bias/variance/granularity tradeoff we thought was fundamental.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import random
import numpy as np
from assembly_probe import make_two_chains, run_forced

BETA = 0.0  # abundant-resource timing is symmetric; keep it clean (undiscounted)


def dA(action_name, r_b):
    _, creds, forced = run_forced(action_name, make_two_chains, beta=BETA, length=12,
                                  r_hi=5, r_lo=2, r_b=r_b)
    assert forced
    return creds


if __name__ == "__main__":
    # ---------- Part A: credit levels on deterministic M4 ----------
    print("=" * 70)
    print("Part A -- credit granularity on M4 (r_b=4): who factors r_b out?")
    print("=" * 70)
    cH, cL = dA('A_hi', 4.0), dA('A_lo', 4.0)
    mcq_h = float(cH['mc_q'][0])                       # unfactored reference level
    print(f"  {'scheme':>10}  credit(hi)  credit(lo)   (hi-lo)   grain")
    for s in ('mc_q', 'ccf', 's_ccf'):
        h, l = float(cH[s][0]), float(cL[s][0])
        grain = "factors r_b out (=r_hi)" if h < mcq_h - 1e-6 else "keeps r_b (conservative)"
        print(f"  {s:>10}  {h:8.2f}  {l:9.2f}   {h-l:+6.2f}   {grain}")
    cf = float(cH['mc_q'][0] - cL['mc_q'][0])   # counterfactual = CRN return difference
    print(f"  {'counterfact':>10}  {'--':>8}  {'--':>9}   {cf:+6.2f}   factors r_b (cancels in the difference)")

    # ---------- Part B: variance with NOISY r_b ----------
    print("\n" + "=" * 70)
    print("Part B -- noisy r_b ~ U{0..20}: which credits carry Var(r_b)?")
    print("=" * 70)
    rng = random.Random(0)
    rows = {s: [] for s in ('mc_q', 'ccf', 's_ccf')}
    cf_adv, ws = [], []
    for _ in range(40):
        w = float(rng.randint(0, 20))
        cH = dA('A_hi', w); cL = dA('A_lo', w)
        for s in rows:
            rows[s].append(float(cH[s][0]))            # dA credit under A_hi
        cf_adv.append(float(cH['mc_q'][0] - cL['mc_q'][0]))   # counterfactual (CRN)
        ws.append(w)
    print(f"  over {len(ws)} episodes, Var(r_b)={np.var(ws):.1f}")
    print(f"  {'scheme':>12}  adv mean   adv Var   status")
    for s in ('mc_q', 's_ccf', 'ccf'):
        a = np.array(rows[s]); adv = a - a.mean()
        note = "carries Var(r_b)" if adv.var() > 1.0 else "factors r_b out"
        print(f"  {s:>12}   {adv.mean():+6.2f}   {adv.var():7.2f}   {note}")
    cf_adv = np.array(cf_adv)
    print(f"  {'counterfact':>12}   {cf_adv.mean():+6.2f}   {cf_adv.var():7.2f}   "
          f"factors r_b out (unbiased, true hi-lo=3)")

    # ---------- Summary: the dominance claim ----------
    print("\n" + "=" * 70)
    print("SUMMARY -- the counterfactual across both motifs")
    print("=" * 70)
    print("             |  M2 shared-R (bias test) |  M4 abundant (grain test)")
    print("  -----------+--------------------------+--------------------------")
    print("  ccf / lrq  |  FLIP (biased)           |  factors r_b (finest)")
    print("  s_ccf      |  unbiased                |  keeps r_b (conservative)")
    print("  mc_q       |  unbiased                |  keeps r_b (no factoring)")
    print("  counterfact|  unbiased                |  factors r_b (finest)")
    print("\n  => the counterfactual is the ONLY method that is unbiased on M2 AND")
    print("     finest-grained on M4 -- it computes the causal effect, so it factors")
    print("     exactly what is causally independent, resolving the tradeoff.")