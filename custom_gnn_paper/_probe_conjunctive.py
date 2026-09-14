r"""Does PNConv close the conjunction gap that HGTConv leaves open?

The motivating measurement (suite/_diag_conjunction.py): a linear probe for
per-transition firing capacity (= min over input places) reaches R^2 ~0.19 on
s1 and ~0.73 on multisite from a SINGLE SCALAR (the summed input counts), while
the HGT encoder's embeddings sit at 0.011 / 0.065 -- placebo level -- and a
fully TRAINED actor is no better than a random one (0.0105 vs 0.0093).

This runs the identical probe against PNEncoder. Both encoders are UNTRAINED
and randomly initialised, so the comparison isolates INDUCTIVE BIAS: what the
architecture makes linearly accessible before any learning. That is the fair
test, because the trained-HGT measurement showed training does not move it.

Reports both encoders side by side with the same folds, the same placebo, and
the same ceiling.

Run: python _probe_conjunctive.py [env] [length]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples", "paper_examples", "suite"))

import numpy as np
import torch

from gympn.simulator import GymProblem
from gympn.networks import HeteroActor
from gympn.solvers import RandomSolver
from pn_conv import PNEncoder

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "s1_stoch_sequence"
LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 20
HIDDEN, LAYERS = 32, 3
RNG = np.random.default_rng(0)

_rows = []
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
    if name == "multisite":
        from multisite_env import make_multisite
        return make_multisite(causal_rl=False, allow_postpone=False)
    return make_env(name, causal_rl=False, allow_postpone=False)


pn = _build(ENV_NAME)
metadata = pn.make_metadata()
for _ in range(12):
    try:
        _build(ENV_NAME).testing_run(solver=RandomSolver(), length=LENGTH)
    except Exception as e:
        print(f"[warn] {e}")
GymProblem.get_graph_observation = _real_obs
print(f"env={ENV_NAME}  observations={len(_rows)}")
if len(_rows) < 20:
    sys.exit("too few observations")

torch.manual_seed(0)
hgt = HeteroActor(input_size=-1, hidden_size=HIDDEN, num_layers=LAYERS,
                  metadata=metadata, num_heads=2)
hgt.eval()
torch.manual_seed(0)
pne = PNEncoder(metadata, hidden=HIDDEN, num_layers=LAYERS)
pne.eval()

EH, EP, S, M = [], [], [], []
for g, tgt in _rows:
    try:
        with torch.no_grad():
            _ = hgt({'graph': g})
            xh = hgt.encoder(x_dict=g.x_dict, edge_index_dict=g.edge_index_dict,
                             input_size=-1, graph=g,
                             params_iter=iter(hgt.parameters()))['a_transition']
            xp = pne(g.x_dict, g.edge_index_dict).get('a_transition')
    except Exception:
        continue
    if xp is None:
        continue
    for (i, mn, sm) in tgt:
        if i < xh.size(0) and i < xp.size(0):
            EH.append(xh[i].numpy()); EP.append(xp[i].numpy())
            S.append([sm]); M.append(mn)
EH = np.array(EH, float); EP = np.array(EP, float)
S = np.array(S, float); M = np.array(M, float)
print(f"samples={len(M)}  HGT dim={EH.shape[1]}  PNConv dim={EP.shape[1]}")
print(f"target: mean={M.mean():.2f} sd={M.std():.2f} values={sorted(set(M.tolist()))[:6]}")
if M.std() < 1e-9:
    sys.exit("target constant -- vacuous")

folds = RNG.permutation(len(M)) % 5


def r2(X, y, lam=1.0):
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    X = np.hstack([X, np.ones((len(X), 1))])
    pred = np.zeros(len(y))
    for k in range(5):
        te, tr = folds == k, folds != k
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        pred[te] = X[te] @ np.linalg.solve(A, X[tr].T @ y[tr])
    return 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


mn = r2(M.reshape(-1, 1), M)
summ = r2(S, M)
r_h = r2(EH, M)
r_p = r2(EP, M)
pl_h = np.mean([r2(EH[RNG.permutation(len(M))], M) for _ in range(5)])
pl_p = np.mean([r2(EP[RNG.permutation(len(M))], M) for _ in range(5)])

print("\n" + "=" * 64)
print(f"HELD-OUT R^2  --  firing capacity (min over input places)  [{ENV_NAME}]")
print("=" * 64)
print(f"  MIN      (ceiling)          : {mn:.4f}")
print(f"  SUM      (naive scalar)     : {summ:.4f}")
print(f"  HGTConv  encoder            : {r_h:.4f}   (placebo {pl_h:+.4f})")
print(f"  PNConv   encoder            : {r_p:.4f}   (placebo {pl_p:+.4f})")
span = max(mn - summ, 1e-9)
print()
print(f"  gap to ceiling recovered -- HGT   : {(r_h - summ) / span:+.1%}")
print(f"  gap to ceiling recovered -- PNConv: {(r_p - summ) / span:+.1%}")
print("\nPNConv is UNTRAINED here, like HGT: this measures inductive bias, which")
print("is the fair comparison because training did not move HGT's number.")
