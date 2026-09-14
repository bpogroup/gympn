r"""The ratio form's defining property: at the null it IS lrq2.

`ls_hca_ratio` weights each lineage-gated contested reward by w = h/pi instead
of adding the mean-zero advantage (1 - pi/h). The point of that change is the
null: with w == 1 the credit must equal lrq2's lineage credit EXACTLY, so
LS-HCA degrades gracefully to a method that works instead of to zero credit
(which on a PURE=set() env collapsed s1 to 1.45 vs lrq2's 12.02).

Checked on real traces from a live training run -- the trace has to be built
by the simulator to carry token ids, so this cannot be done on a synthetic
object. For every episode:

    pure(d) + sum_{contested r : d in lineage(r)} contrib(d,r) * 1  ==  lrq2(d)

Also asserts the cold-start path gives w == 1 (no fitted model -> full lrq2
credit, not zero), and that w != 1 actually moves the credit.

Run: python _test_ls_hca_ratio_null.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as traces_mod

_checks = {"episodes": 0, "worst": 0.0, "nonzero": 0, "decisions": 0}
_real = traces_mod.CausalTraces._redistribute_ls_hca


def _spy(self, *a, **kw):
    pure = _real(self, *a, **kw)
    try:
        beta = kw.get('beta')
        if beta is None and len(a) >= 4:
            beta = a[3]
        lrq2 = self.redistribute_rewards(scheme='lrq2', beta=beta)
        combined = list(pure)
        for t, items in enumerate(self._ls_hca_pending or []):
            for entry in items:
                # entries carry (a_type, rtype, z, contrib, delay, depth,
                # share); only membership and the contribution matter here
                z, contrib = entry[2], entry[3]
                if z:
                    combined[t] += contrib          # w == 1, the null
        x = np.asarray(lrq2, dtype=float)
        y = np.asarray(combined, dtype=float)
        n = min(len(x), len(y))
        if n:
            _checks["episodes"] += 1
            _checks["worst"] = max(_checks["worst"],
                                   float(np.abs(x[:n] - y[:n]).max()))
            _checks["nonzero"] += int((x[:n] != 0).sum())
            _checks["decisions"] += n
    except Exception as e:
        print(f"[warn] null check skipped: {e}")
    return pure


traces_mod.CausalTraces._redistribute_ls_hca = _spy

from envs import make_env  # noqa: E402

env = make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=True)
args = {
    "algorithm": "ppo-clip", "episodes": 6, "epochs": 2, "batch_size": 64,
    "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 2,
    "value_lr": 3e-4, "value_updates": 2, "gam": 0.99, "lam": 0.95,
    "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
    "causal_rl": True, "causal_scheme": "ls_hca", "causal_beta": 0.5,
    "verbose": 0, "use_gpu": False, "agent_seed": 0,
    "use_wandb": False, "open_tensorboard": False, "test_in_train": False,
    "save_freq": 1_000_000, "name": "ratio_null", "datetag": False,
    "logdir": "ratio_null_train",
}
saved = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=20, args_dict=args)
finally:
    sys.argv = saved

print("\n" + "=" * 62)
print("NULL CHECK: pure + sum_{z=True} contrib   ==   lrq2")
print("=" * 62)
print(f"  episodes compared        : {_checks['episodes']}")
print(f"  decisions compared       : {_checks['decisions']}")
print(f"  nonzero lrq2 credits     : {_checks['nonzero']}")
print(f"  worst |difference|       : {_checks['worst']:.3e}")
assert _checks["episodes"] > 0, "no episodes were compared"
assert _checks["nonzero"] > 0, "lrq2 was all zeros -- the check would be vacuous"
assert _checks["worst"] < 1e-9, f"null does NOT reproduce lrq2 (max diff {_checks['worst']})"
print("  -> ratio-form null reproduces lrq2 EXACTLY")

# --- cold start and sensitivity, at the combine-step level ---------------- #
import torch  # noqa: E402
from gympn.agents import Agent  # noqa: E402

ag = object.__new__(Agent)
ag.ls_hca_consistent = True
ag.ls_hca_flat_fallback = False
ag._ls_hca_zmodel = {}
ag._ls_hca_hhat_model = {}
pv = {'start1': 0.6, 'start2': 0.4}
h = ag._ls_hca_predict_h('start1', 'done2', True, [1.0, 2.0], {}, pi_type_vec=pv)
print(f"\n  cold start (no model)    : h={h} -> w=1.0 (full lrq2 credit, not 0)")
assert h is None

# 2 action types x 2 levels, softmaxed over levels
zm = torch.nn.Linear(2, 4)
with torch.no_grad():
    zm.weight.zero_(); zm.bias.copy_(torch.tensor([0.0, 1.2, 0.0, -0.4]))
# (model, classes, n_levels); 2 levels reproduces the binary case
ag._ls_hca_zmodel = {'done2': (zm, ['start1', 'start2'], 2)}
h = ag._ls_hca_predict_h('start1', 'done2', True, [0.0, 0.0], {}, pi_type_vec=pv)
w = h / pv['start1']
print(f"  action-dependent P(z|x,a): w={w:.4f}  (moves credit off lrq2)")
assert abs(w - 1.0) > 0.05, "w should depart from 1 when P(z|x,a) is action-dependent"

import shutil  # noqa: E402
shutil.rmtree("ratio_null_train", ignore_errors=True)
print("\nall ratio-null tests passed")
