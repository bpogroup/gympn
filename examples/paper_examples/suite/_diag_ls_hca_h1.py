r"""Cheap empirical check of LS-HCA hypothesis 1: is the hindsight correction
`factor = 1 - pit/hhat` systematically most NEGATIVE exactly when pit (the
policy's confidence in the taken action) is HIGH -- rather than symmetric
noise around hhat? If so, the type-level-marginalized hhat (no state
conditioning) is punishing confident, state-appropriate decisions, not just
adding noise.

Short run (a handful of epochs -- just enough for hhat to populate and get
applied at least once), agent.ls_hca_debug=True intercepted via a
monkeypatched make_agent, then correlate the logged (pit, factor) pairs.

Run: python _diag_ls_hca_h1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np
from scipy import stats

import gympn.simulator as simulator_mod
import gympn.train as train_mod

_captured = {}
_real_make_agent = train_mod.make_agent


def _spy_make_agent(args, metadata=None):
    agent = _real_make_agent(args, metadata=metadata)
    agent.ls_hca_debug = True
    _captured["agent"] = agent
    return agent


train_mod.make_agent = _spy_make_agent
simulator_mod.make_agent = _spy_make_agent

from envs import make_env  # noqa: E402

env = make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True)

args = {
    "algorithm": "ppo-clip", "episodes": 20, "epochs": 8, "batch_size": 64,
    "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 3,
    "value_lr": 3e-4, "value_updates": 4, "gam": 0.99, "lam": 0.95,
    "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
    "causal_rl": True, "causal_scheme": "ls_hca", "causal_beta": 0.5,
    "verbose": 0, "use_gpu": False, "agent_seed": 0,
    "use_wandb": False, "open_tensorboard": False,
    "test_in_train": False,
    "save_freq": 1_000_000, "name": "ls_hca_h1_diag", "datetag": False,
    "logdir": "ls_hca_h1_diag_train",
}
saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=10, args_dict=args)
finally:
    sys.argv = saved_argv

agent = _captured["agent"]
log = agent._ls_hca_debug_log
print(f"logged {len(log)} (pit, hhat, factor, key) triples")

# Only entries where hhat had data (key was seen) -- the cold-start-safe
# entries (hhat is None -> factor forced to 0) carry no information about H1.
seen = [(pit, hhat, factor) for (pit, hhat, factor, key) in log if hhat is not None]
print(f"{len(seen)} of those had a fitted hhat entry (the rest were cold-start factor=0)")

if len(seen) < 10:
    print("Too few seen-key entries to test H1 meaningfully; try more epochs.")
else:
    pits = np.array([s[0] for s in seen])
    hhats = np.array([s[1] for s in seen])
    factors = np.array([s[2] for s in seen])

    r, p = stats.pearsonr(pits, factors)
    print(f"\nPearson corr(pit, factor) = {r:+.3f}  (p={p:.2e})")
    print("H1 predicts a STRONG NEGATIVE correlation (high pit -> more negative factor).")

    # Quartile view: mean factor by pit quartile
    qs = np.quantile(pits, [0.25, 0.5, 0.75])
    bins = np.digitize(pits, qs)
    print("\npit quartile -> mean factor (n):")
    for b in range(4):
        mask = bins == b
        if mask.sum() > 0:
            lo = 0.0 if b == 0 else qs[b-1]
            hi = 1.0 if b == 3 else qs[b]
            print(f"  pit in [{lo:.2f},{hi:.2f}): mean factor = {factors[mask].mean():+.3f}  (n={mask.sum()})")

    print(f"\nmean hhat={hhats.mean():.3f} (median {np.median(hhats):.3f}), "
          f"mean pit={pits.mean():.3f} (median {np.median(pits):.3f})")

import shutil
shutil.rmtree("ls_hca_h1_diag_train", ignore_errors=True)
