r"""Does a trained (good) policy see FEWER enabled bindings on average than a
random (bad) one on s1? If a good policy drains queues faster, its action
sets would shrink, which would push `pit_type` toward 1.0 for purely
mechanical reasons (fewer bindings to split probability mass across) --
independent of genuine confidence, compounding LS-HCA's diagnosed
pit-vs-hhat lag gap (causal-stability-suite memory, "LS-HCA REVISITED").

Instruments AEPN_Env.step by monkeypatching it to record len(pn.pn_actions)
(action-set size INCLUDING postpone if present) before each decision, then
runs N episodes under RandomSolver vs a trained GymSolver (an s1 lrq2
checkpoint -- a policy that reached ~0.47 norm_final, clearly better than
random) and compares the distributions.

Run: python _diag_action_set_size.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np
from scipy import stats

from gympn.simulator import GymProblem
from gympn.solvers import RandomSolver, GymSolver
from envs import make_env

_sizes = {"tag": None, "log": []}
_real_aug = GymProblem._augment_bindings_with_postpone
_real_obs = GymProblem.get_graph_observation


def _spy_aug(self, bindings):
    result = _real_aug(self, bindings)
    _sizes["log"].append((_sizes["tag"], len(result)))
    return result


def _spy_obs(self, *args, **kwargs):
    result = _real_obs(self, *args, **kwargs)
    n = len(result.get('actions_dict', [])) if isinstance(result, dict) else 0
    if n > 0:
        _sizes["log"].append((_sizes["tag"], n))
    return result


GymProblem._augment_bindings_with_postpone = _spy_aug
GymProblem.get_graph_observation = _spy_obs

N_EPISODES = 20
LENGTH = 10

ckpt = "suite_results_ls_hca_s1/train/s1_stoch_sequence__lrq2__s0/best_policy.pth"

for tag, solver in (
    ("random", RandomSolver()),
    ("trained_lrq2", GymSolver(weights_path=ckpt, metadata=make_env(
        "s1_stoch_sequence", causal_rl=False, allow_postpone=True).make_metadata())),
):
    _sizes["tag"] = tag
    rewards = []
    for i in range(N_EPISODES):
        env = make_env("s1_stoch_sequence", causal_rl=False, allow_postpone=True)
        rewards.append(float(env.testing_run(solver=solver, length=LENGTH)))
    print(f"{tag}: mean episode reward = {np.mean(rewards):.2f}")

GymProblem._augment_bindings_with_postpone = _real_aug
GymProblem.get_graph_observation = _real_obs

for tag in ("random", "trained_lrq2"):
    ns = np.array([n for (t, n) in _sizes["log"] if t == tag])
    print(f"\n{tag}: n_decisions={len(ns)}  "
          f"mean action-set size={ns.mean():.2f}  median={np.median(ns):.1f}  "
          f"min={ns.min()}  max={ns.max()}  std={ns.std():.2f}")

ns_r = np.array([n for (t, n) in _sizes["log"] if t == "random"])
ns_t = np.array([n for (t, n) in _sizes["log"] if t == "trained_lrq2"])
u, p = stats.mannwhitneyu(ns_r, ns_t)
print(f"\nMann-Whitney U(random vs trained action-set sizes): p={p:.2e}")
print("(H: trained policy sees SMALLER action sets on average, if it drains queues faster)")
