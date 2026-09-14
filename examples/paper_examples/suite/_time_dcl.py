"""Time a single DCL planner call and a short episode, per arm, to budget."""
import sys, time
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, ".")
import random, uuid, types
import numpy as np
from gympn.dcl_planner import PlannerConfig, compute_target_pi, compute_target_pi_lineage
from gympn.environment import AEPN_Env
from stoch_envs import make_s1_stoch_sequence


def traced_env(lin, seed=0):
    random.seed(seed); np.random.seed(seed)
    pn = make_s1_stoch_sequence(causal_rl=lin, allow_postpone=True,
                                causal_postpone_tokenflow=lin)
    pn.length = 20
    if lin:
        for p in pn.places:
            for t in p.marking:
                setattr(t, "_id", str(uuid.uuid4()))
        pn.causal_trace.flush()
        sent = types.SimpleNamespace(_id="__initial__")
        toks = [t for p in pn.places for t in p.marking]
        for t in toks:
            pn.causal_trace.register_token(t, sent, [], time=0)
        pn.causal_trace.register_transition(sent, [], toks, is_action=False, reward=0.0, time=0)
    env = AEPN_Env(pn)
    return env, env.reset()


class Actor:
    def __call__(self, s):
        import torch
        return torch.ones(12)


cfg = PlannerConfig(horizon=8, rollouts_per_action=12, gamma=0.95,
                    temperature=0.5, use_crn=True, use_lineage=True)

# plain planner
env, obs = traced_env(False)
t0 = time.time()
for _ in range(5):
    compute_target_pi(env, obs, Actor(), cfg)
print(f"plain      : {(time.time()-t0)/5*1000:.0f} ms/call")

# lineage raw tally
env, obs = traced_env(True)
cfg2 = PlannerConfig(**{**cfg.__dict__, 'lineage_tally': False})
t0 = time.time()
for _ in range(5):
    compute_target_pi_lineage(env, obs, Actor(), cfg2)
print(f"lin (raw)  : {(time.time()-t0)/5*1000:.0f} ms/call")

# lineage restricted tally (sharing active)
env, obs = traced_env(True)
cfg3 = PlannerConfig(**{**cfg.__dict__, 'lineage_tally': True})
t0 = time.time()
for _ in range(5):
    compute_target_pi_lineage(env, obs, Actor(), cfg3)
print(f"lin (tally): {(time.time()-t0)/5*1000:.0f} ms/call")

# how many steps in an s1 episode? (drives per-episode cost)
env, obs = traced_env(False)
n = 0; done = False
while not done and n < 100:
    acts = obs["actions_dict"]
    obs, r, done, _, _ = env.step(0)
    n += 1
print(f"steps/episode ~ {n}")