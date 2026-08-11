r"""Gate for the common-random-numbers eval (test_in_train(eval_seed=...)).

Three properties, all of which must hold for the fix to be usable:

  T1 PINNED    two evals of the SAME policy at DIFFERENT RNG positions return
               exactly the same score with eval_seed set, and different scores
               without it. That is the whole point: arms are compared on one
               fixed scenario set.
  T2 NO-DRIFT  the eval does not disturb the surrounding RNG stream, so turning
               CRN on cannot change the training trajectory it is measuring.
  T3 REAL      the pinned score still discriminates policies (a good policy
               must still outscore a bad one) -- i.e. we pinned the scenarios,
               not the answer.

Run: python _test_eval_crn.py
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from gympn.agents import Agent
from gympn.environment import AEPN_Env
from envs import make_env

ENV = "s1_stoch_sequence"
LENGTH = 20
EPISODES = 8
SEED = 4242
fails = []


class _NoOpModel:
    def train(self):
        pass


class ScriptedAgent(Agent):
    """Bypasses the networks entirely: picks a fixed index into the action
    list, so the POLICY is deterministic and any score change comes from the
    scenarios. index=0 assigns the first available binding (a working policy);
    index=-1 is always postpone (the degenerate one)."""

    def __init__(self, index):
        self.index = index
        self.best_test_metric = float('inf')   # never trips the save-best path
        # test_in_train flips both models back to train() on the way out
        self.policy_model = _NoOpModel()
        self.value_model = _NoOpModel()

    def act(self, state, deterministic=True, return_logprob=False):
        n = len(state['actions_dict']) if isinstance(state, dict) and 'actions_dict' in state else None
        if n is None:
            n = 1
        return (n - 1) if self.index == -1 else min(self.index, n - 1)


def make():
    pn = make_env(ENV, causal_rl=False, allow_postpone=True)
    pn.length = LENGTH
    return AEPN_Env(pn)


def score(agent, eval_seed, burn):
    """Evaluate after consuming `burn` draws, mimicking evals landing at
    different points of a training run's RNG stream."""
    for _ in range(burn):
        random.random()
    return agent.test_in_train(make(), episodes=EPISODES, eval_seed=eval_seed)['mean_returns']


good = ScriptedAgent(0)

# ---- T1: pinned vs unpinned ------------------------------------------------
random.seed(SEED); np.random.seed(SEED)
a = score(good, SEED, burn=0)
b = score(good, SEED, burn=37)
if a != b:
    fails.append(f"T1 pinned: same policy scored {a} then {b} with eval_seed set")

random.seed(SEED); np.random.seed(SEED)
c = score(good, None, burn=0)
d = score(good, None, burn=37)
print(f"T1 pinned   {a:.3f} == {b:.3f}   | unpinned {c:.3f} vs {d:.3f}")
if c == d:
    print("   (note: unpinned scores coincided by chance; not a failure)")

# ---- T2: eval must not disturb the surrounding stream ----------------------
random.seed(SEED); np.random.seed(SEED)
before = [random.random() for _ in range(3)]
random.seed(SEED); np.random.seed(SEED)
_ = [random.random() for _ in range(3)]
tstate = torch.get_rng_state()
score(good, SEED, burn=0)
after = [random.random() for _ in range(3)]
random.seed(SEED); np.random.seed(SEED)
_ = [random.random() for _ in range(3)]
expected = [random.random() for _ in range(3)]
if after != expected:
    fails.append(f"T2 stream disturbed: {after} != {expected}")
if not torch.equal(tstate, torch.get_rng_state()):
    fails.append("T2 torch RNG state not restored")
print(f"T2 stream   continues {after} == {expected}")

# ---- T3: pinned eval still separates policies ------------------------------
bad = ScriptedAgent(-1)
g = score(good, SEED, burn=0)
z = score(bad, SEED, burn=0)
if not g > z:
    fails.append(f"T3 no discrimination: assign={g} vs postpone-always={z}")
print(f"T3 separates assign={g:.3f} > postpone-always={z:.3f}")

print()
if fails:
    print("FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PASS 3/3")
