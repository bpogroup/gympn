r"""Does the actor's logit for site i depend on site j's state? (multisite)

Before building component-masked message passing, check there is leakage to
remove. Every credit method in this project modifies what the policy is trained
TOWARD; masking would modify what it is allowed to DEPEND ON. That only helps if
the actor currently sees across component boundaries.

multisite is the right probe: `flex` is ONE place shared by all sites, while
wait_i / local_i / busy_i / busyf_i are per-site. So site 0's `flex_start_0`
reaches site 3's `busyf_3` in two hops through `flex`, well inside the default
num_layers=3. ncopies would be useless here -- its copies share no place at all,
so its subgraphs are already disconnected and masking would be a no-op.

Method is the one that proved the original actor blind spot: build the real
observation, take an UNTRAINED network (reachability is a property of the
computation graph, not of learned weights), perturb one site's node features,
and measure how far the change propagates into other sites' action logits.
A perturbation is applied to node features directly rather than to the marking,
so exactly one site is touched and nothing else moves.

Run: python _diag_actor_cross_component.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np
import torch

from gympn.networks import HeteroActor
from gympn.solvers import RandomSolver
from multisite_env import make_multisite

torch.manual_seed(0)

pn = make_multisite(causal_rl=False, allow_postpone=False)
metadata = pn.make_metadata()

# Capture a LIVE mid-episode observation. Observing after testing_run() returns
# a terminal marking with zero enabled actions, which makes the probe vacuous
# (0 action logits), so snapshot from inside the run instead and keep the first
# one that offers a real choice across several sites.
from gympn.simulator import GymProblem  # noqa: E402
_snap = []
_real_obs = GymProblem.get_graph_observation


def _spy(self, *a, **kw):
    out = _real_obs(self, *a, **kw)
    try:
        g = out['graph'] if isinstance(out, dict) and 'graph' in out else out
        n = g['a_transition'].x.size(0) if hasattr(g['a_transition'], 'x') else 0
        if n >= 4:
            _snap.append(g.clone())
    except Exception:
        pass
    return out


GymProblem.get_graph_observation = _spy
try:
    pn.testing_run(solver=RandomSolver(), length=10)
finally:
    GymProblem.get_graph_observation = _real_obs

if not _snap:
    sys.exit("no observation with >=4 enabled actions was captured")
graph = _snap[len(_snap) // 2]

actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3,
                    metadata=metadata, num_heads=2)
actor.eval()


def logits(g):
    with torch.no_grad():
        out = actor({'graph': g})
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.reshape(-1).clone()


base = logits(graph)
n_act = base.numel()
print(f"node types: {len(graph.node_types)}   action logits: {n_act}")
print(f"a_transition nodes: {graph['a_transition'].x.size(0)}")

# which node types belong to which site
sites = sorted({nt.rsplit('_', 1)[-1] for nt in graph.node_types
                if nt.rsplit('_', 1)[-1].isdigit()})
print(f"sites detected: {sites}   shared places: "
      f"{[nt for nt in graph.node_types if not nt.rsplit('_',1)[-1].isdigit()]}")

print("\n" + "=" * 70)
print("perturb ONE site's node features -> how much do the logits move?")
print("=" * 70)
for target in sites:
    tts = [nt for nt in graph.node_types
           if nt.rsplit('_', 1)[-1] == target and nt not in ('a_transition',
                                                             'e_transition')]
    g2 = graph.clone()
    touched = 0
    for nt in tts:
        if hasattr(g2[nt], 'x') and g2[nt].x is not None and g2[nt].x.numel():
            g2[nt].x = g2[nt].x + 5.0
            touched += 1
    if not touched:
        print(f"  site {target}: no populated node types, skipped")
        continue
    d = (logits(g2) - base).abs()
    print(f"  site {target}: perturbed {touched} node types "
          f"({', '.join(tts[:4])}{'...' if len(tts) > 4 else ''})")
    print(f"     max |dlogit| = {d.max().item():.6f}   "
          f"mean = {d.mean().item():.6f}   "
          f"logits moved (>1e-9): {int((d > 1e-9).sum())}/{n_act}")

print()
print("If perturbing ONE site moves logits belonging to OTHER sites, message")
print("passing crosses component boundaries and masking has something to remove.")
print("If every perturbation moves only its own site's logits, the actor is")
print("already component-local and masking is a no-op.")
