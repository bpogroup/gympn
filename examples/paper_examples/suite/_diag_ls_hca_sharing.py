r"""Can the hindsight reweighting matter at all? Two measurements.

Context: the ratio form fixed s1's collapse (12.55 vs the difference form's
1.45), but landed at lrq2's level (12.02) -- and lrq2 itself LOSES to plain
ppo_clip on s1 (11.21 vs 13.60 over 10 seeds, paired -2.385, 1W/9L, p=.003).
So "ties lrq2" is not good enough; the question is whether the reweighting can
recover lineage restriction's deficit. Two things have to hold, and neither
has been measured.

Q1 -- IS w ACTUALLY MOVING?  The applied weight is w = P(z|x,a)/P(z|x). If it
sits at 1.0 everywhere the correction is an identity function and no amount of
env-picking will help. Read from the ls_hca_debug log (in ratio mode its third
field is w, not the old factor).

Q2 -- IS THERE ANYTHING TO DISCRIMINATE?  lrq2 hands a reward's full mass to
every one of the k decisions in its lineage, undiscriminated. The reweighting's
entire job is to discriminate among those k. At k == 1 there is nothing to
discriminate and the method is structurally moot on that env -- REGARDLESS of
I(A;Z|X), which is the metric we previously selected envs on. That metric
predicted where the DIFFERENCE form had signal; k is the one that governs the
RATIO form, so the envs chosen for the old estimator may be the wrong ones.
Read from `CausalTraces._ls_hca_lineage_sizes`.

Run: python _diag_ls_hca_sharing.py [env_name] [length]
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as traces_mod
import gympn.simulator as simulator_mod
import gympn.train as train_mod

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "s1_stoch_sequence"
EP_LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 20

_k = defaultdict(list)          # rtype -> [k per reward instance]
_captured = {}

_real_redist = traces_mod.CausalTraces._redistribute_ls_hca


def _spy_redist(self, *a, **kw):
    out = _real_redist(self, *a, **kw)
    for rtype, k in (getattr(self, '_ls_hca_lineage_sizes', None) or []):
        _k[rtype].append(k)
    return out


traces_mod.CausalTraces._redistribute_ls_hca = _spy_redist

_real_make_agent = train_mod.make_agent


def _spy_make_agent(args, metadata=None):
    agent = _real_make_agent(args, metadata=metadata)
    agent.ls_hca_debug = True
    _captured["agent"] = agent
    return agent


train_mod.make_agent = _spy_make_agent
simulator_mod.make_agent = _spy_make_agent

from envs import make_env  # noqa: E402

print(f"[diag] env={ENV_NAME} length={EP_LENGTH}")
env = make_env(ENV_NAME, causal_rl=True, allow_postpone=True)
args = {
    "algorithm": "ppo-clip", "episodes": 20, "epochs": 8, "batch_size": 64,
    "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 3,
    "value_lr": 3e-4, "value_updates": 4, "gam": 0.99, "lam": 0.95,
    "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
    "causal_rl": True, "causal_scheme": "ls_hca", "causal_beta": 0.5,
    "verbose": 0, "use_gpu": False, "agent_seed": 0,
    "use_wandb": False, "open_tensorboard": False, "test_in_train": False,
    "save_freq": 1_000_000, "name": "sharing", "datetag": False,
    "logdir": f"sharing_train_{ENV_NAME}",
}
saved = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=EP_LENGTH, args_dict=args)
finally:
    sys.argv = saved

# ------------------------------------------------------------------ #
print("\n" + "=" * 68)
print("Q1 -- is the applied weight w actually moving off 1.0?")
print("=" * 68)
agent = _captured.get("agent")
log = [w for (_pit, h, w, _key) in (getattr(agent, "_ls_hca_debug_log", []) or [])
       if h is not None]
if not log:
    print("  no weights logged (no fitted model reached the apply step)")
else:
    w = np.array(log, dtype=float)
    dev = np.abs(w - 1.0)
    print(f"  n={len(w)}")
    print(f"  w        : mean={w.mean():.4f}  median={np.median(w):.4f}  "
          f"min={w.min():.4f}  max={w.max():.4f}")
    print(f"  |w - 1|  : median={np.median(dev):.4f}  p90={np.quantile(dev,0.9):.4f}  "
          f"max={dev.max():.4f}")
    for thr in (0.01, 0.05, 0.20, 0.50):
        print(f"    frac |w-1| > {thr:<5}: {float((dev > thr).mean()):.4f}")
    print("  -> w pinned at 1.0 would mean the correction is an identity "
          "function\n     and no choice of env can rescue it.")

print("\n" + "=" * 68)
print("Q2 -- lineage sharing degree k (how many decisions share a reward)")
print("=" * 68)
if not _k:
    print("  no reward instances recorded")
for rtype, ks in sorted(_k.items()):
    a = np.array(ks, dtype=float)
    print(f"  [{rtype}] n_reward_instances={len(a)}")
    print(f"    k: mean={a.mean():.2f}  median={np.median(a):.1f}  "
          f"min={int(a.min())}  max={int(a.max())}  p90={np.quantile(a,0.9):.1f}")
    hist = Counter(int(x) for x in a)
    top = ", ".join(f"k={k}:{v}" for k, v in sorted(hist.items())[:8])
    print(f"    distribution: {top}")
    print(f"    frac k==1 (nothing to discriminate): {float((a == 1).mean()):.4f}")

import shutil  # noqa: E402
shutil.rmtree(f"sharing_train_{ENV_NAME}", ignore_errors=True)
