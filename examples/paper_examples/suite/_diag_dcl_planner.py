"""Isolate planner quality from distillation: at real s1 states, print the
planner's target_pi against the action categories. A competent planner must
put mass on the FAST (matched) assignment and low mass on postpone.
"""
import sys, random, uuid, types
sys.path.insert(0, r"C:\Users\lobia\PycharmProjects\gympn")
sys.path.insert(0, ".")
import numpy as np
import torch
from gympn.dcl_planner import PlannerConfig, compute_target_pi
from gympn.environment import AEPN_Env
from stoch_envs import make_s1_stoch_sequence


def classify(binding):
    if binding is None or (isinstance(binding[0], list) and binding[0] == ['postpone']):
        return "postpone"
    tr = str(getattr(binding[2], '_id', ''))
    stage = 's1' if tr.startswith('start1') else 's2' if tr.startswith('start2') else '?'
    task = emp = None
    for _, tok in binding[0]:
        v = getattr(tok, 'value', None)
        if isinstance(v, dict):
            if 'task_type' in v: task = v['task_type']
            elif 'code_employee' in v: emp = v['code_employee']
    if task is None or emp is None:
        return f"{stage}_?"
    pair = 'gen' if emp == 2 else ('match' if task == emp else 'cross')
    return f"{stage}_{pair}"


class UniformActor:
    def __call__(self, s):
        n = len(getattr(s, 'actions_dict', [])) if hasattr(s, 'actions_dict') else 10
        return torch.ones(max(n, 10))


def trained_critic_stub(s):
    # deliberately zero: isolate the ROLLOUT reward signal, no value tail
    return torch.zeros(1)


random.seed(0); np.random.seed(0)
pn = make_s1_stoch_sequence(causal_rl=False, allow_postpone=True)
pn.length = 20
env = AEPN_Env(pn)
obs = env.reset()

def _plan_and_rank(env, obs, beta):
    cfg = PlannerConfig(horizon=10, rollouts_per_action=32, gamma=0.9,
                        temperature=0.3, use_crn=True, use_lineage=False,
                        value_bootstrap=False, beta=beta)
    pi, stats, enabled = compute_target_pi(env, obs, UniformActor(), cfg,
                                           critic=None)
    cats = [classify(b) for b in obs['actions_dict']]
    pp = [j for j, c in enumerate(cats) if c == "postpone"]
    pp_rank = None
    if pp:
        order = list(np.argsort(pi)[::-1])
        pp_rank = order.index(pp[0]) + 1
    top = int(np.argmax(pi))
    return pi, stats['means'], cats, top, pp_rank, len(cats)


# Compare flat gamma vs SMDP beta=0.5 on WHERE POSTPONE RANKS, at loaded
# states reached by the SMDP planner (postpone collapse = postpone ranked #1).
for step in range(6):
    for label, beta in (("flat_gamma", 0.0), ("smdp_b0.5", 0.5)):
        pi, means, cats, top, pp_rank, A = _plan_and_rank(env, obs, beta)
        print(f"[{label:>10}] step {step} A={A}: top={cats[top]} "
              f"postpone_rank={pp_rank}/{A} "
              f"(pi_top={pi[top]:.3f})")
    # advance with the SMDP planner's greedy choice
    pi, means, cats, top, pp_rank, A = _plan_and_rank(env, obs, 0.5)
    _, _, enabled = compute_target_pi(
        env, obs, UniformActor(),
        PlannerConfig(horizon=10, rollouts_per_action=8, gamma=0.9,
                      temperature=0.3, beta=0.5, value_bootstrap=False,
                      use_lineage=False), critic=None)
    obs, r, done, _, _ = env.step(int(enabled[top]))
    if done:
        break