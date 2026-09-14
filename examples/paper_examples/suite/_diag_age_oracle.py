r"""Is token AGE worth anything on these envs? Oracle test, no training.

`get_graph_observation` builds token features from `token.value.items()` only,
so a policy sees a case's ATTRIBUTES but never how long it has been waiting.
On s1 that value dict is just {'task_type': t}. The policy therefore cannot
implement FIFO, SRPT, EDD or any other age/deadline rule -- a hard restriction
on the policy class, and in scheduling those are the classic optimal rules.

Before building any feature plumbing, check whether the information is worth
having AT ALL. A heuristic solver sees the raw bindings, and SimToken carries
`.time`, so a heuristic CAN sort by age even though a policy cannot. If an
age-aware oracle cannot beat the age-blind anchor, the information is useless
here and no encoding of it will help.

Arms (all never postpone, like the tier anchor):
  anchor    -- the shipped `_tier_heuristic`: stage priority, then type match.
  age       -- same stage priority, then OLDEST token first (type ignored).
  age+type  -- same stage priority, type match first, ties broken by oldest.
  youngest  -- control: same as `age` but NEWEST first. If this scores like
               `age`, the ordering carries nothing and any difference is noise.

Run: python _diag_age_oracle.py [episodes]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

from gympn.solvers import HeuristicSolver
from envs import make_env, HEURISTICS
from stoch_envs import _tier_heuristic

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def _parts(b):
    """(transition_id, case_value, resource_value, case_token) or Nones."""
    try:
        tr = getattr(b[2], '_id', None) or getattr(b[2], 'name', None)
        tok = b[0][0][1]
        case = tok.value
        res = b[0][1][1].value if len(b[0]) > 1 else None
        return str(tr), case, res, tok
    except (IndexError, TypeError, AttributeError):
        return None, None, None, None


def age_heuristic(stage_order, n_types, use_type, oldest=True):
    """Stage priority first (identical to the anchor), then an age ordering."""
    def heuristic(observable_net, tokens_comb, bindings=None):
        if not bindings:
            return None
        for want in stage_order:
            cands = []
            for b in bindings:
                tr, case, res, tok = _parts(b)
                if tr is None or not tr.startswith(want):
                    continue
                matched = (res is not None and case is not None and
                           res.get('code_employee', -1) % n_types ==
                           case.get('task_type', -2))
                t = getattr(tok, 'time', None)
                t = float(t) if t is not None else 0.0
                # sort key: type match first (if used), then age
                cands.append(((0 if matched else 1) if use_type else 0,
                              t if oldest else -t, b))
            if cands:
                cands.sort(key=lambda x: (x[0], x[1]))
                return cands[0][2]
        for b in bindings:
            if _parts(b)[0] is not None:
                return b
        return None
    return heuristic


def evaluate(env_name, heuristic, length, n_ep, seed0=50_000):
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

print(f"AGE-ORACLE TEST  ({N_EP} episodes per arm, identical seeds)")
print("=" * 78)
for env_name, order, n_types, length in CASES:
    arms = {
        "anchor":   HEURISTICS.get(env_name, _tier_heuristic(order, n_types)),
        "age":      age_heuristic(order, n_types, use_type=False, oldest=True),
        "age+type": age_heuristic(order, n_types, use_type=True, oldest=True),
        "youngest": age_heuristic(order, n_types, use_type=False, oldest=False),
    }
    print(f"\n{env_name}  (horizon {length})")
    base = None
    for name, h in arms.items():
        try:
            v = evaluate(env_name, h, length, N_EP)
        except Exception as e:
            print(f"  {name:<9} ERROR {type(e).__name__}: {e}")
            continue
        if base is None:
            base = v
        d = v - base
        extra = "" if name == "anchor" else f"   vs anchor {d.mean():+7.3f}"
        print(f"  {name:<9} mean={v.mean():8.3f}  sd={v.std():6.3f}{extra}")
print()
print("If no age arm beats the anchor, age carries nothing here and no")
print("encoding of it into the observation will help.")
