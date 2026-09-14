r"""Ungated null check: with w == 1 the credit must be exactly mc_q.

`ls_hca_gate=False` pays EVERY contested reward (weighted by w), not only the
ones whose lineage the decision sits in. The point is the fallback: the gated
null is lrq2, which LOSES to plain PPO on s1 (11.21 vs 13.59, 1W/9L, p=.003),
so a gated LS-HCA inherits that deficit. The ungated null is mc_q, which never
significantly loses to PPO on any measured env.

Verified on real traces from a live run (traces must be simulator-built to
carry token ids). On an env where PURE is empty -- s1 -- the identity is exact:

    sum_{contested r, ALL decisions} contrib(d,r) * 1   ==   mc_q(d)

Also asserts the gated null still reproduces lrq2, so the knob really does
select between the two fallbacks rather than replacing one with the other.

Run: python _test_ls_hca_ungated_null.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.causal_traces as traces_mod

_c = {"n": 0, "worst_mcq": 0.0, "worst_lrq2": 0.0, "nonzero": 0, "pure": 0.0,
      "postpone": 0, "decisions": 0}
_real = traces_mod.CausalTraces._redistribute_ls_hca


def _spy(self, *a, **kw):
    pure = _real(self, *a, **kw)
    try:
        beta = kw.get('beta')
        if beta is None and len(a) >= 4:
            beta = a[3]
        mcq = self.redistribute_rewards(scheme='mc_q', beta=beta)
        lrq2 = self.redistribute_rewards(scheme='lrq2', beta=beta)
        ungated = list(pure)
        gated = list(pure)
        for t, items in enumerate(self._ls_hca_pending or []):
            for e in items:
                z, contrib = e[2], e[3]
                ungated[t] += contrib               # w == 1, every reward
                if z:
                    gated[t] += contrib             # w == 1, lineage only
        # ls_hca gives postpone decisions no credit at all
        # (causal_traces.py:800), whereas mc_q credits every decision
        # including postpone -- "there is no causal filtering, that is the
        # point". So the identity can only hold on NON-POSTPONE decisions;
        # comparing on all of them measures the postpone convention, not the
        # gate. Mask them out and count them so the difference stays visible.
        acts = self.transition_history.get_action_transitions()
        is_pp = np.array([
            bool(isinstance(getattr(a.get('transition'), '_id', None), str)
                 and a['transition']._id.startswith('postpone_'))
            for a in acts])
        m = np.asarray(mcq, dtype=float)
        l = np.asarray(lrq2, dtype=float)
        u = np.asarray(ungated, dtype=float)
        g = np.asarray(gated, dtype=float)
        n = min(len(m), len(u), len(is_pp))
        keep = ~is_pp[:n]
        _c["postpone"] += int(is_pp[:n].sum())
        _c["decisions"] += n
        m, l, u, g = m[:n][keep], l[:n][keep], u[:n][keep], g[:n][keep]
        n = int(keep.sum())
        if n:
            _c["n"] += 1
            _c["pure"] += float(np.abs(np.asarray(pure, dtype=float)).sum())
            _c["worst_mcq"] = max(_c["worst_mcq"], float(np.abs(m - u).max()))
            _c["worst_lrq2"] = max(_c["worst_lrq2"], float(np.abs(l - g).max()))
            _c["nonzero"] += int((m != 0).sum())
    except Exception as e:
        print(f"[warn] skipped: {e}")
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
    "save_freq": 1_000_000, "name": "ungated_null", "datetag": False,
    "logdir": "ungated_null_train",
}
saved = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=20, args_dict=args)
finally:
    sys.argv = saved

print("\n" + "=" * 64)
print("NULL CHECKS on s1 (PURE is empty here, so the mc_q identity is exact)")
print("=" * 64)
print(f"  episodes compared      : {_c['n']}")
print(f"  nonzero mc_q credits   : {_c['nonzero']}")
print(f"  |PURE| mass (expect 0) : {_c['pure']:.6f}")
print(f"  decisions / of which postpone: {_c['decisions']} / {_c['postpone']}"
      f"  ({_c['postpone']/max(_c['decisions'],1):.1%} excluded -- ls_hca pays them 0, mc_q pays them in full)")
print(f"  worst |ungated - mc_q| : {_c['worst_mcq']:.3e}   (non-postpone decisions)")
print(f"  worst |gated   - lrq2| : {_c['worst_lrq2']:.3e}")
assert _c["n"] > 0, "no episodes compared"
assert _c["nonzero"] > 0, "mc_q all zero -- check would be vacuous"
assert _c["worst_mcq"] < 1e-9, f"ungated null != mc_q (max {_c['worst_mcq']})"
assert _c["worst_lrq2"] < 1e-9, f"gated null != lrq2 (max {_c['worst_lrq2']})"
print("  -> ungated null == mc_q EXACTLY, gated null == lrq2 EXACTLY")
print("     (the knob selects the fallback; it does not replace one with the other)")

import shutil  # noqa: E402
shutil.rmtree("ungated_null_train", ignore_errors=True)
print("\nall ungated-null tests passed")
