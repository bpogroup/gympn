r"""Does knowing the QUEUE COMPOSITION change the optimal action?

Measured (custom_gnn_paper/README.md): an a_transition node is a BINDING wired
only to its own tokens (degree 1 per input place), so a binding cannot see what
else is queued. Exposing that costs either O(tokens x bindings) context edges or
a place-summary node -- plus the own/context edge-type split needed to keep
bindings distinguishable.

Before any of that: is the information WORTH having? A heuristic solver sees the
raw bindings, so it can compute the composition even though a policy cannot. If a
composition-aware oracle cannot beat the composition-blind anchor, no encoding of
composition into the observation will help.

Arms (none postpone, matching the tier anchor):
  anchor        `_tier_heuristic`: stage priority, then a type-matched employee.
                Never looks at what else is queued.
  scarce-first  among candidates, serve the task type that is RAREST among the
                available tasks -- composition-aware.
  abundant-1st  the opposite. CONTROL: if it scores like scarce-first, the
                ordering carries nothing and any difference is noise.
  opp-cost      opportunity cost: prefer pairs whose employee is NOT badly needed
                by other queued task types (do not spend a specialist on a task
                someone else could take). The classic composition-dependent rule.
  best-delay    ignore stage priority, take the globally fastest pair. Tests
                whether the anchor's stage ordering is itself costing anything.

Run: python _diag_composition_oracle.py [episodes]
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

from gympn.solvers import HeuristicSolver
from envs import make_env, HEURISTICS
from stoch_envs import _tier_heuristic

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def _parts(b):
    """(transition_id, case_value, resource_value) or Nones."""
    try:
        tr = getattr(b[2], '_id', None) or getattr(b[2], 'name', None)
        case = b[0][0][1].value
        res = b[0][1][1].value if len(b[0]) > 1 else None
        return str(tr), case, res
    except (IndexError, TypeError, AttributeError):
        return None, None, None


def _delay(case, res, n_types):
    """The tier service model: exact match 1, cross 2, generalist 3."""
    t = case.get('task_type', -1)
    c = res.get('code_employee', -1) if res else -1
    if c == t:
        return 1
    return 2 if 0 <= c < n_types else 3


def composition_heuristic(stage_order, n_types, mode):
    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None
        parsed = [(b,) + _parts(b) for b in bindings]
        parsed = [p for p in parsed if p[1] is not None]
        if not parsed:
            return None
        # composition of the tasks currently actionable
        comp = Counter(p[2].get('task_type') for p in parsed if p[2])
        # how many queued tasks each employee code is the exact match for
        need = Counter()
        for t, k in comp.items():
            need[t] += k

        def score(p):
            _b, tr, case, res = p
            t = case.get('task_type', -1)
            c = res.get('code_employee', -1) if res else -1
            d = _delay(case, res, n_types)
            if mode == 'scarce':
                return (d, comp.get(t, 0))
            if mode == 'abundant':
                return (d, -comp.get(t, 0))
            if mode == 'opp':
                # penalty for using an employee whose matched type is queued
                # elsewhere (c != t means this pair is not that match)
                penalty = need.get(c, 0) if c != t else 0
                return (d + penalty, comp.get(t, 0))
            return (d, 0)                      # 'best-delay'

        if mode == 'best-delay':
            return min(parsed, key=score)[0]
        for want in stage_order:               # keep the anchor's stage priority
            cands = [p for p in parsed if p[1].startswith(want)]
            if cands:
                return min(cands, key=score)[0]
        return parsed[0][0]
    return heuristic


def evaluate(env_name, heuristic, length, n_ep, seed0=70_000):
    from gympn import seed_everything
    out = []
    for i in range(n_ep):
        seed_everything(seed0 + i)
        env = make_env(env_name, causal_rl=False, allow_postpone=False)
        out.append(float(env.testing_run(solver=HeuristicSolver(heuristic),
                                         length=length)))
    return np.array(out)


CASES = [
    ("s1_stoch_sequence", ["start2", "start1"], 2, 20),
    ("s2_stoch_scaled",   ["assign"],           3, 30),
    ("s3_stoch_mixed",    ["start_L2", "start_L1", "start_S"], 1, 25),
]

print(f"COMPOSITION-ORACLE TEST  ({N_EP} episodes per arm, identical seeds)")
print("=" * 78)
for env_name, order, n_types, length in CASES:
    arms = {
        "anchor":      HEURISTICS.get(env_name, _tier_heuristic(order, n_types)),
        "scarce-1st":  composition_heuristic(order, n_types, 'scarce'),
        "abundant-1st": composition_heuristic(order, n_types, 'abundant'),
        "opp-cost":    composition_heuristic(order, n_types, 'opp'),
        "best-delay":  composition_heuristic(order, n_types, 'best-delay'),
    }
    print(f"\n{env_name}  (horizon {length})")
    base = None
    for name, h in arms.items():
        try:
            v = evaluate(env_name, h, length, N_EP)
        except Exception as e:
            print(f"  {name:<12} ERROR {type(e).__name__}: {e}")
            continue
        if base is None:
            base = v
        extra = "" if name == "anchor" else f"   vs anchor {(v - base).mean():+7.3f}"
        print(f"  {name:<12} mean={v.mean():8.3f}  sd={v.std():6.3f}{extra}")
print()
print("If no composition arm beats the anchor, queue composition does not change")
print("the optimal action here and exposing it to the network will not help.")
