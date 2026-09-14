r"""Is the critic blind to in-flight timing, and does it matter?

`SimToken(value, delay=...)` keeps `delay`/`time` OUTSIDE `token.value`, and
`get_graph_observation` builds node features from `token.value.items()` only.
So a token in `busy1` looks identical whether it completes in 0.1 time units
or 3.4. Under the SMDP discount e^{-beta*t} with beta=0.5 those two futures
differ by e^{-1.65} ~ 0.19 in discounted value -- a 5x difference the value
network cannot see.

This matters differently from every input-side experiment run before. Those
(phi-shaping, structural features, actor global-context, the age oracle) all
asked "does more information change which ACTION is best?" -- and the age
oracle answered no. This asks "does more information make the RETURN easier to
PREDICT?". A better critic is a better baseline; a baseline cannot bias the
policy gradient, only its variance. So if the answer is yes, it is a free win.

Offline, no training loop needed: collect (state, return-to-go) pairs from a
short real run, then fit a ridge regressor twice --

  OBSERVABLE : per-place token counts + per-place means of the observable
               token attributes. Everything the current featurization can see.
  + TIMING   : the same, plus per-place remaining-time statistics
               (max(0, token.time - clock)) and global in-flight summaries.

-- and compare HELD-OUT R^2 on the same folds. If timing adds nothing, the
marking already proxies it and the idea dies here for the cost of one run.

Run: python _diag_critic_timing.py [env_name] [length]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")

import numpy as np

import gympn.simulator as sim_mod
import gympn.causal_traces as tm
from gympn.simulator import GymProblem

ENV_NAME = sys.argv[1] if len(sys.argv) > 1 else "s1_stoch_sequence"
LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RNG = np.random.default_rng(0)

_obs_rows = []          # per get_graph_observation call
_episodes = []          # (rows_for_episode, returns_for_episode)

_real_obs = GymProblem.get_graph_observation


def _snapshot(pn):
    """(observable_features, timing_features) from the raw net state."""
    clock = float(getattr(pn, 'clock', 0.0) or 0.0)
    obs, tim = [], []
    n_inflight, rem_all = 0, []
    for p in pn.places:
        mk = list(getattr(p, 'marking', None) or ())
        obs.append(float(len(mk)))
        # observable: mean of each numeric token attribute
        vals = {}
        for t in mk:
            v = getattr(t, 'value', None)
            if isinstance(v, dict):
                for k, x in v.items():
                    if isinstance(x, (int, float)):
                        vals.setdefault(k, []).append(float(x))
        obs.append(float(np.mean(vals[k])) if (vals and (k := sorted(vals)[0])) else 0.0)
        # NOT observable: remaining time until each token becomes available
        rem = []
        for t in mk:
            tt = getattr(t, 'time', None)
            if tt is not None:
                rem.append(max(0.0, float(tt) - clock))
        if rem:
            n_inflight += sum(1 for r in rem if r > 0)
            rem_all.extend(rem)
        tim.extend([min(rem) if rem else 0.0,
                    float(np.mean(rem)) if rem else 0.0,
                    max(rem) if rem else 0.0])
    tim.extend([float(n_inflight),
                min(rem_all) if rem_all else 0.0,
                float(np.mean(rem_all)) if rem_all else 0.0])
    return obs, tim


def _spy_obs(self, *a, **kw):
    try:
        _obs_rows.append(_snapshot(self))
    except Exception:
        _obs_rows.append(None)
    return _real_obs(self, *a, **kw)


GymProblem.get_graph_observation = _spy_obs

_real_redis = tm.CausalTraces.redistribute_rewards


def _spy_redis(self, scheme="lrq", **kw):
    out = _real_redis(self, scheme=scheme, **kw)
    if scheme == "lrq2":                       # once per finished episode
        try:
            rets = _real_redis(self, scheme="mc_q", beta=kw.get("beta", 0.0))
            n = len(rets)
            rows = [r for r in _obs_rows if r is not None]
            # the observation is rebuilt at each decision; keep the LAST n so a
            # trailing/extra call cannot shift the alignment
            if len(rows) >= n and n > 0:
                _episodes.append((rows[-n:], list(rets)))
        except Exception as e:
            print(f"[warn] {e}")
        _obs_rows.clear()
    return out


tm.CausalTraces.redistribute_rewards = _spy_redis

from envs import make_env  # noqa: E402

env = make_env(ENV_NAME, causal_rl=True, allow_postpone=True)
args = {"algorithm": "ppo-clip", "episodes": 20, "epochs": 3, "batch_size": 64,
        "max_episode_length": None, "policy_lr": 3e-4, "policy_updates": 1,
        "value_lr": 3e-4, "value_updates": 1, "gam": 0.99, "lam": 0.95,
        "eps": 0.2, "vf_coeff": 0.5, "ent_bonus": 0.01, "policy_kld_limit": 0.15,
        "causal_rl": True, "causal_scheme": "lrq2", "causal_beta": 0.5,
        "verbose": 0, "use_gpu": False, "agent_seed": 0, "use_wandb": False,
        "open_tensorboard": False, "test_in_train": False, "save_freq": 10**9,
        "name": "critic_timing", "datetag": False, "logdir": "critic_timing_train"}
saved = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=LENGTH, args_dict=args)
finally:
    sys.argv = saved
    import shutil
    shutil.rmtree("critic_timing_train", ignore_errors=True)

X_obs, X_tim, y = [], [], []
for rows, rets in _episodes:
    for (o, t), r in zip(rows, rets):
        X_obs.append(o); X_tim.append(t); y.append(float(r))
X_obs = np.array(X_obs, float); X_tim = np.array(X_tim, float); y = np.array(y, float)
print(f"\nenv={ENV_NAME}  samples={len(y)}  observable dims={X_obs.shape[1]}  "
      f"timing dims={X_tim.shape[1]}")
if len(y) < 100:
    sys.exit("too few samples")


def r2_cv(X, y, folds=5, lam=1.0):
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    X = np.hstack([X, np.ones((len(X), 1))])
    idx = RNG.permutation(len(y)) % folds
    pred = np.zeros(len(y))
    for k in range(folds):
        te, tr = idx == k, idx != k
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ y[tr])
        pred[te] = X[te] @ w
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


base = r2_cv(X_obs, y)
both = r2_cv(np.hstack([X_obs, X_tim]), y)
tim_only = r2_cv(X_tim, y)
# control: timing columns shuffled -- destroys the information, keeps the
# dimensionality, so any gain from merely adding columns shows up here.
placebo = np.mean([r2_cv(np.hstack([X_obs, X_tim[RNG.permutation(len(y))]]), y)
                   for _ in range(5)])

print("\n" + "=" * 62)
print("HELD-OUT R^2 predicting the return-to-go")
print("=" * 62)
print(f"  observable only          : {base:.4f}")
print(f"  observable + TIMING      : {both:.4f}   ({both - base:+.4f})")
print(f"  observable + shuffled    : {placebo:.4f}   ({placebo - base:+.4f})  <- placebo")
print(f"  timing only              : {tim_only:.4f}")
print()
print(f"  residual variance removed by timing: "
      f"{(both - base) / max(1.0 - base, 1e-9):.1%}")
print("\nA gain over the PLACEBO (not over the baseline) is the real signal;")
print("extra columns alone can lift R^2 without carrying information.")
