r"""LS-HCA Step 0: does the conditioning variable carry ANY information?

The estimator is `factor = 1 - pi(a|x) / h(a|x, rtype, z)`. Its entire content
comes from h differing from pi, and h can only differ from pi through the
conditioning variable (rtype, z). So the precondition for LS-HCA to say
anything at all is

    I(A ; Z | X, rtype) > 0

If z carries no information about the action beyond what the state already
says, then h == pi in population, `factor` is estimation noise around 0, and
no amount of better fitting can rescue the method -- the honest conclusion
would be a negative result, not another tuning round.

Two tests, both on records the agent already pools (`_ls_hca_records`,
captured here by wrapping `_fit_ls_hca_hhat`, which is called once per epoch
with that epoch's full pool before `agents.py:416` clears it):

  TEST 1 (marginal, no fitting, exact): G-test of independence on the
    a_type x z contingency table within each rtype. This is exactly the
    flat Laplace-smoothed table the implementation falls back to.

  TEST 2 (state-conditional, mirrors the implementation): per-(rtype, z)
    logistic models on the state features -- the same split
    `_fit_ls_hca_hhat` uses -- scored by held-out cross-entropy against a
    PLACEBO whose z labels are shuffled within rtype. Shuffling preserves
    group sizes and the z marginal, and destroys only the a-z association,
    so the placebo distribution IS the null. p = fraction of placebo draws
    at least as good as the real split.

Also reports (the secondary question): how much credit mass flows through the
exact PURE term vs the hindsight-corrected CONTESTED term, by wrapping
`_redistribute_ls_hca`. If PURE is empty the lineage support contributes
nothing and this is plain HCA.

Run: python _diag_ls_hca_step0.py [env_name]     (default: s1_stoch_sequence)
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np
from scipy import stats

import gympn.agents as agents_mod
import gympn.causal_traces as traces_mod
import gympn.simulator as simulator_mod
import gympn.train as train_mod

N_PLACEBO = 200
N_FOLDS = 5
RIDGE = 1e-2
FIT_ITERS = 150
FIT_LR = 0.05
RNG = np.random.default_rng(0)

# ------------------------------------------------------------------ #
# capture: one snapshot of the pooled records per epoch               #
# ------------------------------------------------------------------ #
_snapshots = []           # list of per-epoch record lists
_mass = {"pure": 0.0, "contested": 0.0, "pure_nonzero": 0, "n_dec": 0}
_classify_seen = {}

_real_fit = agents_mod.Agent._fit_ls_hca_hhat


def _spy_fit(self):
    _snapshots.append(list(self._ls_hca_records))
    return _real_fit(self)


agents_mod.Agent._fit_ls_hca_hhat = _spy_fit

# TEST 3 needs the factors the implementation ACTUALLY applied, to compare
# against the ideal ones -- that is exactly what ls_hca_debug already logs.
_captured = {}
_real_make_agent = train_mod.make_agent


def _spy_make_agent(args, metadata=None):
    agent = _real_make_agent(args, metadata=metadata)
    agent.ls_hca_debug = True
    _captured["agent"] = agent
    return agent


train_mod.make_agent = _spy_make_agent
simulator_mod.make_agent = _spy_make_agent

_real_redist = traces_mod.CausalTraces._redistribute_ls_hca


def _spy_redist(self, *a, **kw):
    out = _real_redist(self, *a, **kw)
    try:
        _mass["pure"] += float(np.abs(np.asarray(out, dtype=float)).sum())
        _mass["pure_nonzero"] += int((np.asarray(out, dtype=float) != 0.0).sum())
        _mass["n_dec"] += len(out)
        for items in (self._ls_hca_pending or []):
            _mass["contested"] += sum(abs(float(c)) for (_, _, _, c) in items)
        cls = self._ls_hca_classify_cache
        if cls:
            for aid, (pure, contested) in cls.items():
                _classify_seen[aid] = (tuple(sorted(pure)), tuple(sorted(contested)))
    except Exception as e:  # diagnostic must never break the run
        print(f"[warn] mass accounting failed: {e}")
    return out


traces_mod.CausalTraces._redistribute_ls_hca = _spy_redist

# ------------------------------------------------------------------ #
# short real run on s1                                                #
# ------------------------------------------------------------------ #
from envs import make_env  # noqa: E402

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "s1_stoch_sequence"
# Horizon defaults to 10 -- NOT the suite's per-env length -- so that every
# measurement taken before this argument existed stays exactly reproducible
# and cross-env comparable. Pass it explicitly to measure an env at the
# horizon it is actually trained on (the stoch tier runs 20-25).
EP_LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 10
print(f"[diag] env = {ENV_NAME}  length = {EP_LENGTH}")
env = make_env(ENV_NAME, causal_rl=True, allow_postpone=True)

args = {
    "algorithm": "ppo-clip", "episodes": 20, "epochs": 10, "batch_size": 64,
    "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 3,
    "value_lr": 3e-4, "value_updates": 4, "gam": 0.99, "lam": 0.95,
    "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
    "causal_rl": True, "causal_scheme": "ls_hca", "causal_beta": 0.5,
    "verbose": 0, "use_gpu": False, "agent_seed": 0,
    "use_wandb": False, "open_tensorboard": False, "test_in_train": False,
    "save_freq": 1_000_000, "name": "ls_hca_step0", "datetag": False,
    "logdir": f"ls_hca_step0_train_{ENV_NAME}",
}
saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=EP_LENGTH, args_dict=args)
finally:
    sys.argv = saved_argv

# ------------------------------------------------------------------ #
# report 0: pure vs contested credit mass                             #
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("SECONDARY: is the lineage (PURE) support doing anything?")
print("=" * 70)
tot = _mass["pure"] + _mass["contested"]
if tot > 0:
    print(f"  |PURE| credit mass      = {_mass['pure']:.2f}  ({_mass['pure']/tot:.1%})")
    print(f"  |CONTESTED| credit mass = {_mass['contested']:.2f}  ({_mass['contested']/tot:.1%})")
print(f"  decisions with nonzero PURE credit: {_mass['pure_nonzero']} / {_mass['n_dec']}")
print("  static classification per action-type:")
for aid, (pure, contested) in sorted(_classify_seen.items()):
    print(f"    {aid}: PURE={set(pure) or 'set()'}  CONTESTED={set(contested) or 'set()'}")

# ------------------------------------------------------------------ #
# report 1: G-test on the a_type x z table, per rtype                 #
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("TEST 1 -- marginal association between action type and z (no state)")
print("=" * 70)

pool = [r for snap in _snapshots for r in snap]
print(f"pooled {len(pool)} records across {len(_snapshots)} epochs "
      f"(per-epoch: {[len(s) for s in _snapshots]})")

by_rtype = defaultdict(list)
for rec in pool:
    by_rtype[rec[1]].append(rec)


def g_test(recs, label):
    a_types = sorted({r[0] for r in recs})
    zs = sorted({bool(r[2]) for r in recs})
    if len(a_types) < 2 or len(zs) < 2:
        print(f"  [{label}] DEGENERATE: a_types={a_types} z_values={zs} "
              f"-- no association is even representable")
        return None
    table = np.zeros((len(a_types), len(zs)))
    for r in recs:
        table[a_types.index(r[0]), zs.index(bool(r[2]))] += 1
    g, p, dof, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
    n = table.sum()
    cramers_v = np.sqrt(g / (n * (min(table.shape) - 1))) if n else 0.0
    print(f"  [{label}] n={int(n)}  a_types={a_types}  z={zs}")
    print(f"    contingency (rows=a_type, cols=z):\n{table.astype(int)}")
    print(f"    P(a | z) columns:")
    for j, zv in enumerate(zs):
        col = table[:, j] / max(table[:, j].sum(), 1)
        print(f"      z={zv}: " + ", ".join(f"{a}={v:.3f}" for a, v in zip(a_types, col)))
    print(f"    G={g:.2f}  dof={dof}  p={p:.3e}  Cramer's V={cramers_v:.3f}")
    return p


for rtype, recs in sorted(by_rtype.items(), key=lambda kv: str(kv[0])):
    g_test(recs, f"rtype={rtype}, all epochs")

# pi changes across epochs, so pooling could in principle mask real
# within-epoch structure (or manufacture spurious structure via Simpson's
# paradox) -- check each epoch's own pool separately, compactly.
print("\n  per-epoch P(a|z=True) vs P(a|z=False) and its G-test p:")
for i, snap in enumerate(_snapshots):
    for rtype in sorted({r[1] for r in snap}, key=str):
        recs = [r for r in snap if r[1] == rtype]
        a_types = sorted({r[0] for r in recs})
        zs = sorted({bool(r[2]) for r in recs})
        if len(a_types) < 2 or len(zs) < 2:
            print(f"    epoch {i} rtype={rtype}: degenerate "
                  f"(a_types={a_types}, z={zs})")
            continue
        table = np.zeros((len(a_types), len(zs)))
        for r in recs:
            table[a_types.index(r[0]), zs.index(bool(r[2]))] += 1
        g, p, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
        cols = table / np.maximum(table.sum(axis=0, keepdims=True), 1)
        desc = "  ".join(
            f"P({a}|z={zv})={cols[ai, zi]:.3f}"
            for zi, zv in enumerate(zs) for ai, a in enumerate(a_types)
            if ai == 0)
        print(f"    epoch {i} rtype={rtype}: n={int(table.sum())}  {desc}  p={p:.3e}")

# ------------------------------------------------------------------ #
# report 2: state-conditional held-out CE, real z vs shuffled placebo #
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("TEST 2 -- state-conditional: does splitting on z beat a shuffled placebo?")
print("=" * 70)


def fit_predict_ce(X_tr, y_tr, X_te, y_te, n_cls):
    """Multinomial logistic (ridge-regularized, IRLS-free GD) -- the same
    model family as `_fit_ls_hca_hhat`'s nn.Linear, scored as held-out mean
    negative log-likelihood of the observed class."""
    d = X_tr.shape[1]
    W = np.zeros((d, n_cls))
    b = np.zeros(n_cls)
    Y = np.zeros((len(y_tr), n_cls))
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    for _ in range(FIT_ITERS):
        logits = X_tr @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        gW = X_tr.T @ (P - Y) / len(y_tr) + RIDGE * W
        gb = (P - Y).mean(axis=0)
        W -= FIT_LR * gW
        b -= FIT_LR * gb
    logits = X_te @ W + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    p_obs = np.clip(P[np.arange(len(y_te)), y_te], 1e-12, None)
    return -np.log(p_obs)


def cv_ce_for_split(X, y, zlab, n_cls, folds):
    """Mean held-out CE when a SEPARATE model is fit per z-group (exactly the
    per-(rtype, z) split `_fit_ls_hca_hhat` uses). Records whose z-group is
    absent/degenerate in a training fold fall back to that fold's pooled
    model, so real and placebo are always scored on the identical test set."""
    ce = np.empty(len(y))
    for k in range(folds):
        te = folds_idx == k
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        for zv in (False, True):
            te_z = te & (zlab == zv)
            if te_z.sum() == 0:
                continue
            tr_z = tr & (zlab == zv)
            if tr_z.sum() >= 2 * n_cls and len(np.unique(y[tr_z])) > 1:
                ce[te_z] = fit_predict_ce(X[tr_z], y[tr_z], X[te_z], y[te_z], n_cls)
            else:
                ce[te_z] = fit_predict_ce(X[tr], y[tr], X[te_z], y[te_z], n_cls)
    return ce.mean()


for rtype, recs in sorted(by_rtype.items(), key=lambda kv: str(kv[0])):
    recs = [r for r in recs if len(r) > 3 and r[3] is not None]
    if not recs:
        print(f"  [rtype={rtype}] no state features captured -- skipped")
        continue
    a_types = sorted({r[0] for r in recs})
    if len(a_types) < 2:
        print(f"  [rtype={rtype}] DEGENERATE: single action type {a_types}, "
              f"nothing to discriminate")
        continue
    dim = len(recs[0][3])
    recs = [r for r in recs if len(r[3]) == dim]
    X = np.array([r[3] for r in recs], dtype=float)
    y = np.array([a_types.index(r[0]) for r in recs])
    zlab = np.array([bool(r[2]) for r in recs])
    # standardize (the real fit uses Adam on raw counts; scaling only helps
    # both arms equally and makes the fixed-step GD here well-conditioned)
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)

    folds_idx = RNG.permutation(len(y)) % N_FOLDS
    real = cv_ce_for_split(X, y, zlab, len(a_types), N_FOLDS)
    null = np.array([cv_ce_for_split(X, y, RNG.permutation(zlab), len(a_types), N_FOLDS)
                     for _ in range(N_PLACEBO)])
    p = (np.sum(null <= real) + 1) / (len(null) + 1)
    print(f"  [rtype={rtype}] n={len(y)}  dim={X.shape[1]}  classes={a_types}  "
          f"z-rate={zlab.mean():.3f}")
    print(f"    held-out CE, real z split : {real:.5f}")
    print(f"    held-out CE, placebo      : {null.mean():.5f} +- {null.std():.5f}"
          f"  (min {null.min():.5f})")
    print(f"    improvement from z        : {null.mean() - real:+.5f} nats/record")
    print(f"    permutation p             : {p:.4f}   (n_placebo={N_PLACEBO})")

# ------------------------------------------------------------------ #
# report 3: EFFECT SIZE -- how big is the ideal correction, really?    #
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("TEST 3 -- effect size: ideal |factor| vs the |factor| actually applied")
print("=" * 70)
print("""In population pi(a|x) = sum_z P(z|x) h(a|x,z), so the IDEAL correction
  factor = 1 - pi(a|x)/h(a|x,z)
is computable from the h-family alone -- no policy network involved, hence no
pi-vs-h estimator mismatch. That is the correction a perfectly calibrated
LS-HCA would apply, i.e. an upper bound on what the method can be worth here.
Fitted out-of-fold, so this is a held-out estimate, not an in-sample one.""")


def fit_probs(X_tr, y_tr, X_te, n_cls):
    """Same ridge multinomial logistic as TEST 2, but returning the full
    predicted distribution rather than only the observed class's CE."""
    d = X_tr.shape[1]
    W = np.zeros((d, n_cls))
    b = np.zeros(n_cls)
    Y = np.zeros((len(y_tr), n_cls))
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    for _ in range(FIT_ITERS):
        logits = X_tr @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        W -= FIT_LR * (X_tr.T @ (P - Y) / len(y_tr) + RIDGE * W)
        b -= FIT_LR * (P - Y).mean(axis=0)
    logits = X_te @ W + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(axis=1, keepdims=True)


for rtype, recs in sorted(by_rtype.items(), key=lambda kv: str(kv[0])):
    recs = [r for r in recs if len(r) > 3 and r[3] is not None]
    a_types = sorted({r[0] for r in recs})
    if len(recs) < 100 or len(a_types) < 2:
        continue
    dim = len(recs[0][3])
    recs = [r for r in recs if len(r[3]) == dim]
    X = np.array([r[3] for r in recs], dtype=float)
    y = np.array([a_types.index(r[0]) for r in recs])
    zlab = np.array([bool(r[2]) for r in recs])
    sd = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)

    n_cls = len(a_types)
    folds_idx = RNG.permutation(len(y)) % N_FOLDS
    h_obs = np.zeros(len(y))        # h(a_i | x_i, z_i)
    pi_obs = np.zeros(len(y))       # sum_z P(z|x_i) h(a_i | x_i, z)
    for k in range(N_FOLDS):
        te, tr = folds_idx == k, folds_idx != k
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # P(z | x): the mixing weights that turn h back into pi
        pz = fit_probs(X[tr], zlab[tr].astype(int), X[te], 2)
        h_by_z = {}
        for zi, zv in enumerate((False, True)):
            tr_z = tr & (zlab == zv)
            src = tr_z if (tr_z.sum() >= 2 * n_cls
                           and len(np.unique(y[tr_z])) > 1) else tr
            h_by_z[zv] = fit_probs(X[src], y[src], X[te], n_cls)
        rows = np.arange(te.sum())
        y_te, z_te = y[te], zlab[te]
        h_obs[te] = np.array([h_by_z[z_te[i]][i, y_te[i]] for i in rows])
        pi_obs[te] = np.array([
            pz[i, 0] * h_by_z[False][i, y_te[i]] + pz[i, 1] * h_by_z[True][i, y_te[i]]
            for i in rows])

    ideal = 1.0 - pi_obs / np.maximum(h_obs, 1e-9)
    print(f"\n  [rtype={rtype}] n={len(y)}")
    print(f"    ideal factor : mean={ideal.mean():+.4f}  sd={ideal.std():.4f}  "
          f"median|f|={np.median(np.abs(ideal)):.4f}")
    print(f"                   p90|f|={np.quantile(np.abs(ideal), 0.90):.4f}  "
          f"max|f|={np.abs(ideal).max():.4f}")
    for thr in (0.05, 0.2, 1.0):
        print(f"      frac |ideal factor| > {thr:<4}: "
              f"{np.mean(np.abs(ideal) > thr):.4f}")

agent = _captured.get("agent")
applied = [f for (_, h, f, _) in (getattr(agent, "_ls_hca_debug_log", []) or [])
           if h is not None]
if applied:
    ap = np.abs(np.array(applied))
    print(f"\n  APPLIED factors (n={len(ap)}, from ls_hca_debug):")
    print(f"    mean|f|={ap.mean():.4f}  median|f|={np.median(ap):.4f}  "
          f"p90|f|={np.quantile(ap, 0.90):.4f}  max|f|={ap.max():.4f}")
    for thr in (0.05, 0.2, 1.0):
        print(f"      frac |applied factor| > {thr:<4}: {np.mean(ap > thr):.4f}")
    print(f"    frac AT the +-{getattr(agent, 'ls_hca_factor_clip', 3.0)} clip: "
          f"{np.mean(ap >= getattr(agent, 'ls_hca_factor_clip', 3.0) - 1e-9):.4f}")

# Persist the records so any follow-up analysis needs no retraining.
import pickle  # noqa: E402
with open(f"_diag_ls_hca_step0_records_{ENV_NAME}_L{EP_LENGTH}.pkl", "wb") as fh:
    pickle.dump({"snapshots": _snapshots,
                 "applied": list(getattr(agent, "_ls_hca_debug_log", []) or []),
                 "mass": _mass, "classify": _classify_seen}, fh)
print(f"\nrecords written to _diag_ls_hca_step0_records_{ENV_NAME}_L{EP_LENGTH}.pkl")

import shutil  # noqa: E402
shutil.rmtree(f"ls_hca_step0_train_{ENV_NAME}", ignore_errors=True)
