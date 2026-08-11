r"""Why is cfgae's epoch-2 greedy eval exactly 0.00 on s1?

test_in_train records ``info['pn_reward']``, which is the episode's CUMULATIVE
env reward, and s1 pays 1 per completed case. So 0.00 means literally zero
cases completed across all 100 test episodes -- not a missing-metric artifact.

The suspected mechanism is a POSTPONE LOCK: the greedy eval is argmax
(agents.py:556), postpone is always the last entry of the action list
(simulator.py:662, environment.py:65), and an early policy whose argmax lands
on postpone in every decision state never assigns anyone, so no case ever
completes. Sampling can't lock this way, which is why the sampled curve is
~9 at the same epoch.

This probe trains the real cfgae cell for a few epochs and instruments the
GREEDY eval only, reporting per-eval postpone share, episode lengths and
returns. Postpone share ~1.0 with return 0.0 confirms the lock; anything else
refutes it.

Run: python _diag_greedy_zero.py [epochs]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gympn.environment as gpenv
from gympn.agents import Agent
from config import stoch_config
from run_suite import _make_args, _set_seed
from envs import make_env

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

cfg = stoch_config()
cfg.epochs = EPOCHS
cfg.test_freq = 2

# ---- instrumentation: count postpone choices, but only inside greedy eval ---
STATE = {"on": False, "postpone": 0, "total": 0}
_orig_step = gpenv.AEPN_Env.step


def counting_step(self, action, build_obs=True):
    if STATE["on"]:
        acts = self.pn.pn_actions
        STATE["total"] += 1
        if action == len(acts) - 1 and acts[-1][0] == ['postpone']:
            STATE["postpone"] += 1
    return _orig_step(self, action, build_obs)


gpenv.AEPN_Env.step = counting_step

_orig_test = Agent.test_in_train


def reporting_test(self, env, episodes=100, max_episode_length=None,
                   deterministic=True, logdir=None):
    STATE.update(on=True, postpone=0, total=0)
    try:
        m = _orig_test(self, env, episodes=episodes,
                       max_episode_length=max_episode_length,
                       deterministic=deterministic, logdir=logdir)
    finally:
        STATE["on"] = False
    share = STATE["postpone"] / STATE["total"] if STATE["total"] else float('nan')
    print(f"[greedy-eval] return={m['mean_returns']:.2f} "
          f"ep_len mean={m['mean_ep_lens']:.1f} min={m['min_ep_lens']:.0f} "
          f"max={m['max_ep_lens']:.0f} | postpone {STATE['postpone']}/"
          f"{STATE['total']} = {share:.3f}", flush=True)
    return m


Agent.test_in_train = reporting_test

# ---- the real cfgae cell, same construction as run_suite.train_cell ---------
_set_seed(0)
env = make_env("s1_stoch_sequence", causal_rl=True, allow_postpone=cfg.allow_postpone,
               causal_postpone_tokenflow=True)
env.use_structural_features = False
args = _make_args("s1_stoch_sequence", "cfgae", 0, cfg, "_diag_greedy_zero_train")
print(f"[probe] cfgae s1 seed 0, {cfg.epochs} epochs, scheme={args['causal_scheme']}",
      flush=True)

saved = sys.argv
sys.argv = sys.argv[:1]
try:
    env.training_run(length=20, args_dict=args)
finally:
    sys.argv = saved

h = getattr(env, "training_history", {}) or {}
print("[probe] greedy curve:", list(h.get("test_mean_returns", [])))
print("[probe] sampled curve:", list(h.get("mean_returns", [])))
