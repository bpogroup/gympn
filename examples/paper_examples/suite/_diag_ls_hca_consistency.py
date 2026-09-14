r"""Why does a residual-parameterized h STILL over-correct?

The residual form guarantees the NULL: g == 0 => h == pi => factor == 0. The
L2 sweep (`_diag_ls_hca_l2_sweep.py`) ruled out the obvious follow-up
explanation -- held-out CE is best at lambda=0 on both envs, so g is not
overfitting and shrinking it destroys signal rather than noise. Yet the
applied |factor| still runs ~2.5-3.3x the held-out ideal.

This checks the remaining candidate, which is a property the residual form
does NOT give you. Any genuine hindsight distribution must satisfy the law of
total probability,

    sum_z P(z|x) h(a|x,z) = pi(a|x)                      (*)

-- marginalizing the conditioning variable back out has to return the policy.
The residual parameterization pins h to pi only at g == 0; for g != 0 each
(rtype, z) group is fit INDEPENDENTLY, so nothing couples the z=True and
z=False models and (*) is violated. The estimator then computes
factor = 1 - pi/h with a pi that is not the marginal of its own h, and the
violation shows up directly as excess correction.

Measures the violation on cached records via the real fit path, and compares
its size against the excess correction it is supposed to explain.

Run: python _diag_ls_hca_consistency.py [env_name]
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np
import torch

from gympn.agents import Agent

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "i_mixed_credit"
RNG = np.random.default_rng(0)

with open(f"_diag_ls_hca_step0_records_{ENV_NAME}.pkl", "rb") as fh:
    snapshots = pickle.load(fh)["snapshots"]
pool = [r for snap in snapshots for r in snap if len(r) > 4 and r[4]]
if not pool:
    sys.exit(f"no pi-vector records cached for {ENV_NAME}")

idx = RNG.permutation(len(pool))
cut = int(0.7 * len(pool))
train = [pool[i] for i in idx[:cut]]
test = [pool[i] for i in idx[cut:]]
print(f"env={ENV_NAME}  train={len(train)}  held-out={len(test)}")

# --- fit h exactly as training does ------------------------------------- #
a = object.__new__(Agent)
a._ls_hca_records = list(train)
a._ls_hca_hhat = {}
a._ls_hca_hhat_model = {}
a._ls_hca_state_node_types = ["_"]
a.ls_hca_smoothing_alpha = 1.0
a.ls_hca_state_min_n = 30
a.ls_hca_state_epochs = 50
a.ls_hca_state_lr = 0.05
a.ls_hca_state_l2 = 0.0
a.ls_hca_residual = True
a.ls_hca_flat_fallback = False
a._fit_ls_hca_hhat()
print(f"fitted groups: {sorted(a._ls_hca_hhat_model)}")

# --- P(z|x): logistic on the same state features ------------------------- #
Xtr = torch.tensor([r[3] for r in train], dtype=torch.float32)
ztr = torch.tensor([1 if r[2] else 0 for r in train], dtype=torch.long)
mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
pz_model = torch.nn.Linear(Xtr.shape[1], 2)
opt = torch.optim.Adam(pz_model.parameters(), lr=0.05)
for _ in range(300):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(pz_model((Xtr - mu) / sd), ztr).backward()
    opt.step()

# --- measure the violation of (*) on held-out records -------------------- #
viol, rel, applied, implied = [], [], [], []
for (a_type, rtype, z, feat, pv) in test:
    h_by_z = {}
    for zv in (False, True):
        h = a._ls_hca_predict_h(a_type, rtype, zv, feat, a._ls_hca_hhat,
                                pi_type_vec=pv)
        if h is None:
            break
        h_by_z[zv] = h
    if len(h_by_z) < 2:
        continue
    with torch.no_grad():
        x = (torch.tensor([feat], dtype=torch.float32) - mu) / sd
        p1 = float(torch.softmax(pz_model(x), dim=-1).reshape(-1)[1])
    marg = (1.0 - p1) * h_by_z[False] + p1 * h_by_z[True]
    pi = float(pv.get(a_type, 0.0))
    if pi <= 0:
        continue
    viol.append(marg - pi)
    rel.append((marg - pi) / pi)
    # what the estimator applies vs what it WOULD apply if pi were replaced by
    # h's own marginal -- i.e. with (*) enforced by construction
    applied.append(1.0 - pi / h_by_z[z])
    implied.append(1.0 - marg / h_by_z[z])

viol, rel = np.array(viol), np.array(rel)
applied, implied = np.abs(np.array(applied)), np.abs(np.array(implied))
print(f"\nevaluated on {len(viol)} held-out records")
print("=" * 66)
print("violation of  sum_z P(z|x) h(a|x,z) == pi(a|x)")
print("=" * 66)
print(f"  mean signed  : {viol.mean():+.5f}")
print(f"  median |viol|: {np.median(np.abs(viol)):.5f}")
print(f"  p90    |viol|: {np.quantile(np.abs(viol), 0.90):.5f}")
print(f"  median |viol| relative to pi: {np.median(np.abs(rel)):.4f} "
      f"({np.median(np.abs(rel)):.1%} of pi)")

print("\n" + "=" * 66)
print("does that violation account for the excess correction?")
print("=" * 66)
print(f"  |factor| as APPLIED          (1 - pi/h)   : median {np.median(applied):.4f}"
      f"  p90 {np.quantile(applied, 0.9):.4f}")
print(f"  |factor| with (*) ENFORCED   (1 - marg/h) : median {np.median(implied):.4f}"
      f"  p90 {np.quantile(implied, 0.9):.4f}")
if np.median(implied) > 0:
    print(f"  ratio applied/enforced                   : "
          f"{np.median(applied) / np.median(implied):.2f}x")
print("\n  (if enforcing (*) collapses the correction toward the measured"
      "\n   ideal, marginal inconsistency IS the remaining error source, and"
      "\n   coupling the z-groups at fit time is the fix)")
