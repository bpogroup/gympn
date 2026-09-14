r"""Is sum-aggregation the wrong inductive bias for Petri-net enabling?

A transition fires only when EVERY input place is marked, and how many times it
could fire is  min_p (tokens in p)  -- a conjunction. HGTConv aggregates input
places with an attention-weighted SUM, which cannot separate

    3 tokens in one input place, 0 elsewhere   (capacity 0)
    1 token in each of 3 input places          (capacity 1)

The two look alike to a sum and are opposite in PN semantics. But HGT is not a
bare sum: multi-head attention over several layers can approximate a min. So
the question is not "does sum equal min" (it does not, trivially) but "is the
min RECOVERABLE from what the encoder actually produces".

Probe: collect real observations, then linear-probe three feature sets for the
per-transition firing capacity (the min):

  ENCODER : the actual HeteroActor encoder's transition embeddings
  SUM     : the summed input-place token counts -- the naive-aggregation bound
  MIN     : the true min -- the ceiling (sanity: must be ~1.0)

ENCODER near MIN  -> the architecture already exposes conjunction; a min
                     aggregator would buy nothing.
ENCODER near SUM  -> the inductive bias is genuinely wrong and a PN-native
                     conjunctive operator has something to fix.

Held-out R^2 with a shuffled-feature placebo, same discipline as the critic
probe. Run: python _diag_conjunction.py [env] [length]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np
import torch

from gympn.simulator import GymProblem
from gympn.networks import HeteroActor
from gympn.solvers import RandomSolver

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "s1_stoch_sequence"
LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RNG = np.random.default_rng(0)

_rows = []          # (graph, [(node_idx, min_cap, sum_cap), ...])
_real_obs = GymProblem.get_graph_observation


def _spy(self, *a, **kw):
    out = _real_obs(self, *a, **kw)
    try:
        g = out['graph'] if isinstance(out, dict) and 'graph' in out else out
        acts = list(getattr(self, 'pn_actions', []) or [])
        n = g['a_transition'].x.size(0) if hasattr(g['a_transition'], 'x') else 0
        if n == 0 or len(acts) < n:
            return out
        counts = {p._id: len(getattr(p, 'marking', None) or ()) for p in self.places}
        tgt = []
        for i in range(n):
            b = acts[i]
            tr = b[2] if len(b) > 2 else None
            inc = list(getattr(tr, 'incoming', ()) or ())
            if not inc:
                continue
            c = [counts.get(p._id, 0) for p in inc]
            tgt.append((i, float(min(c)), float(sum(c))))
        if tgt:
            _rows.append((g.clone(), tgt))
    except Exception:
        pass
    return out


GymProblem.get_graph_observation = _spy

from envs import make_env  # noqa: E402


def _build(name):
    """multisite lives in its own module, not envs.ENV_BUILDERS."""
    if name == "multisite":
        from multisite_env import make_multisite
        return make_multisite(causal_rl=False, allow_postpone=False)
    return make_env(name, causal_rl=False, allow_postpone=False)


pn = _build(ENV_NAME)
metadata = pn.make_metadata()
for ep in range(12):
    try:
        _build(ENV_NAME).testing_run(solver=RandomSolver(), length=LENGTH)
    except Exception as e:
        print(f"[warn] ep{ep}: {e}")
GymProblem.get_graph_observation = _real_obs
print(f"env={ENV_NAME}  observations captured={len(_rows)}")
if len(_rows) < 20:
    sys.exit("too few observations")

torch.manual_seed(0)
CKPT = sys.argv[3] if len(sys.argv) > 3 else None
if CKPT:
    # A TRAINED encoder. The untrained probe measures what the architecture
    # makes linearly accessible by inductive bias; this measures whether
    # training recovers it. Only the second justifies a new operator.
    actor = torch.load(CKPT, weights_only=False, map_location='cpu')
    print(f"loaded trained actor from {CKPT}")
else:
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3,
                        metadata=metadata, num_heads=2)
actor.eval()

E, S, M = [], [], []
for g, tgt in _rows:
    try:
        with torch.no_grad():
            _ = actor({'graph': g})           # materialize lazy modules
            x = actor.encoder(x_dict=g.x_dict, edge_index_dict=g.edge_index_dict,
                              input_size=-1, graph=g,
                              params_iter=iter(actor.parameters()))['a_transition']
    except Exception as e:
        continue
    for (i, mn, sm) in tgt:
        if i < x.size(0):
            E.append(x[i].numpy()); S.append([sm]); M.append(mn)
E = np.array(E, float); S = np.array(S, float); M = np.array(M, float)
print(f"probe samples={len(M)}  encoder dim={E.shape[1] if len(E) else 0}")
print(f"target (firing capacity = min over input places): "
      f"mean={M.mean():.2f} sd={M.std():.2f} unique={sorted(set(M.tolist()))[:8]}")
if M.std() < 1e-9:
    sys.exit("target is constant -- probe would be vacuous")


def r2(X, y, folds=5, lam=1.0):
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    X = np.hstack([X, np.ones((len(X), 1))])
    idx = RNG.permutation(len(y)) % folds
    pred = np.zeros(len(y))
    for k in range(folds):
        te, tr = idx == k, idx != k
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        pred[te] = X[te] @ np.linalg.solve(A, X[tr].T @ y[tr])
    return 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


enc = r2(E, M)
summ = r2(S, M)
mn = r2(M.reshape(-1, 1), M)
plac = np.mean([r2(E[RNG.permutation(len(M))], M) for _ in range(5)])
print("\n" + "=" * 60)
print("HELD-OUT R^2 predicting firing capacity (min over input places)")
print("=" * 60)
print(f"  MIN     (ceiling)            : {mn:.4f}")
print(f"  ENCODER (HGT embeddings)     : {enc:.4f}")
print(f"  SUM     (naive aggregation)  : {summ:.4f}")
print(f"  encoder, shuffled (placebo)  : {plac:.4f}")
print()
span = max(mn - summ, 1e-9)
print(f"  encoder recovers {(enc - summ) / span:.1%} of the SUM -> MIN gap")
print("\nNear 100% => conjunction is already accessible, a min-aggregator adds")
print("nothing. Near 0% => the inductive bias is wrong and worth fixing.")
