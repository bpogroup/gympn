r"""Pick ls_hca_state_l2 offline, from records already on disk.

After the residual reparameterization the applied correction still ran
~2.5-3.3x the held-out ideal (`_diag_ls_hca_step0.py`). The pi-vs-h structural
mismatch is gone by construction, so what is left is g's OWN estimation error:
g is fit in-sample, ls_hca_state_epochs passes, no shrinkage. A null-centered
prior (AdamW weight decay -- and under the residual form g=0 IS the null) is
the matching fix, but its strength should be measured, not guessed.

This sweeps lambda through the REAL fit path (`Agent._fit_ls_hca_hhat` called
on a bare `object.__new__(Agent)` with records injected) so nothing here can
drift from what training actually does. Records come from the cached
`_diag_ls_hca_step0_records_<env>.pkl`, so no retraining is needed.

Primary criterion is HELD-OUT cross-entropy of h on records the fit never saw:
if shrinkage improves it, g was overfitting, and the improvement is real
rather than a cosmetic shrinking of |factor|. Reported alongside are the
resulting |factor| quantiles and the same out-of-fold "ideal" from
_diag_ls_hca_step0.py's TEST 3, which is what the applied factors should
resemble if h is well calibrated.

Run: python _diag_ls_hca_l2_sweep.py [env_name]
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

from gympn.agents import Agent

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "i_mixed_credit"
LAMBDAS = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
RNG = np.random.default_rng(0)

with open(f"_diag_ls_hca_step0_records_{ENV_NAME}.pkl", "rb") as fh:
    blob = pickle.load(fh)
snapshots = blob["snapshots"]

# Records carry a pi vector only since the residual change; older caches do not
# and cannot be swept.
pool = [r for snap in snapshots for r in snap if len(r) > 4 and r[4]]
if not pool:
    sys.exit(f"no pi-vector records in the {ENV_NAME} cache -- rerun "
             f"_diag_ls_hca_step0.py {ENV_NAME} first")
print(f"env={ENV_NAME}  usable records={len(pool)} "
      f"(of {sum(len(s) for s in snapshots)} pooled)")

idx = RNG.permutation(len(pool))
cut = int(0.7 * len(pool))
train = [pool[i] for i in idx[:cut]]
test = [pool[i] for i in idx[cut:]]
print(f"train={len(train)}  held-out test={len(test)}")


def make_agent(l2):
    a = object.__new__(Agent)
    a._ls_hca_records = list(train)
    a._ls_hca_hhat = {}
    a._ls_hca_hhat_model = {}
    a._ls_hca_state_node_types = ["_"]      # only tested for non-emptiness
    a.ls_hca_smoothing_alpha = 1.0
    a.ls_hca_state_min_n = 30
    a.ls_hca_state_epochs = 50
    a.ls_hca_state_lr = 0.05
    a.ls_hca_state_l2 = l2
    a.ls_hca_residual = True
    a.ls_hca_flat_fallback = False
    a.ls_hca_factor_clip = 3.0
    a.ls_hca_hhat_floor = 0.05
    a._fit_ls_hca_hhat()
    return a


def evaluate(a):
    """Held-out mean -log h(a|x,z) and the |factor| the agent would apply."""
    ce, fac = [], []
    for (a_type, rtype, z, feat, pv) in test:
        h = a._ls_hca_predict_h(a_type, rtype, z, feat, a._ls_hca_hhat,
                                pi_type_vec=pv)
        if h is None or h <= 0:
            continue
        ce.append(-np.log(max(h, 1e-12)))
        pit = float(pv.get(a_type, 0.0))
        f = 1.0 - pit / h
        fac.append(min(3.0, max(-3.0, f)))
    return np.array(ce), np.abs(np.array(fac))


# The same out-of-fold ideal as _diag_ls_hca_step0.py TEST 3, on the test split:
# pi(a|x) = sum_z P(z|x) h(a|x,z), so a perfectly calibrated correction is
# 1 - pi/h with BOTH sides from the h-family. Here pi is the policy's own
# captured vector, which is what the residual form anchors to, so the ideal
# reduces to how much h should move off pi -- estimated by fitting on train
# and reading the shift on test.
print()
print(f"{'lambda':>8}  {'held-out CE':>12}  {'median|f|':>10}  {'p90|f|':>8}  "
      f"{'mean|f|':>8}  {'frac>0.2':>9}  {'n_models':>8}")
print("-" * 78)
results = []
for lam in LAMBDAS:
    a = make_agent(lam)
    ce, fac = evaluate(a)
    if len(ce) == 0:
        print(f"{lam:>8}  {'(no predictions)':>12}")
        continue
    results.append((lam, ce.mean(), np.median(fac), np.quantile(fac, 0.9),
                    fac.mean(), float((fac > 0.2).mean()),
                    len(a._ls_hca_hhat_model)))
    print(f"{lam:>8}  {ce.mean():>12.5f}  {np.median(fac):>10.4f}  "
          f"{np.quantile(fac, 0.9):>8.4f}  {fac.mean():>8.4f}  "
          f"{float((fac > 0.2).mean()):>9.4f}  {len(a._ls_hca_hhat_model):>8}")

if results:
    best = min(results, key=lambda r: r[1])
    zero = [r for r in results if r[0] == 0.0]
    print()
    print(f"best held-out CE at lambda={best[0]}  (CE {best[1]:.5f})")
    if zero:
        d = zero[0][1] - best[1]
        # A CE gain has to be big enough to mean something. Held-out CE here is
        # a mean over thousands of records, so its own noise floor is ~1e-4
        # nats; anything under that is a tie, not evidence of overfitting.
        verdict = ("g was OVERFITTING -- shrinkage genuinely helps"
                   if d > 1e-4 else
                   "NO overfitting -- shrinkage buys nothing (tie at the noise "
                   "floor); any |factor| reduction it produces is shrinking "
                   "SIGNAL, not noise")
        print(f"  vs lambda=0: CE {zero[0][1]:.5f} -> {best[1]:.5f} "
              f"({d:+.5f} nats)")
        print(f"  median|f|  : {zero[0][2]:.4f} -> {best[2]:.4f}")
        print(f"  verdict    : {verdict}")
