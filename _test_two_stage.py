"""Quick test: verify causal RL with advantage replacement converges on two-stage."""
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

from gympn.simulator import GymProblem
from simpn.simulator import SimToken

def create_two_stage_env(causal_rl=False):
    agency = GymProblem(allow_postpone=True, causal_rl=causal_rl)
    arrival = agency.add_var("arrival", var_attributes=['case_id'])
    waiting_A = agency.add_var("waiting_A", var_attributes=['case_id'])
    busy_A = agency.add_var("busy_A", var_attributes=['case_id', 'resource_id'])
    waiting_B = agency.add_var("waiting_B", var_attributes=['case_id'])
    busy_B = agency.add_var("busy_B", var_attributes=['case_id', 'resource_id'])
    arrival.put({'case_id': 0})
    arrival.put({'case_id': 0})
    resource = agency.add_var("resource", var_attributes=['resource_id'])
    resource.put({'resource_id': 0})
    resource.put({'resource_id': 1})

    def arrive(a):
        return [SimToken({'case_id': a['case_id']}, delay=1), SimToken(a)]
    agency.add_event([arrival], [arrival, waiting_A], arrive)

    def start_A(c, r):
        delay = 0.5 if r['resource_id'] == 0 else 2
        return [SimToken((c, r), delay=delay)]
    agency.add_action([waiting_A, resource], [busy_A], behavior=start_A, name="start_A")

    def complete_A(b):
        case, res = b
        return [SimToken(res), SimToken(case)]
    agency.add_event([busy_A], [resource, waiting_B], complete_A, name='complete_A')

    def start_B(c, r):
        delay = 0.5 if r['resource_id'] == 1 else 2
        return [SimToken((c, r), delay=delay)]
    agency.add_action([waiting_B, resource], [busy_B], behavior=start_B, name="start_B")

    def complete_B(b):
        _, res = b
        return [SimToken(res)]
    agency.add_event([busy_B], [resource], complete_B, name='complete_B', reward_function=lambda x: 1)
    return agency

args_dict = {
    'episodes': 20,
    'epochs': 25,
    'batch_size': 32,
    'max_episode_length': None,
    'policy_lr': 3e-4,
    'value_lr': 1e-3,
    'gam': 0.99,
    'lam': 0.95,
    'eps': 0.2,
    'vf_coeff': 0.5,
    'ent_bonus': 0.01,
    'policy_kld_limit': 0.2,
    'causal_rl': True,
    'algorithm': 'ppo-clip',
    'verbose': 1,
    'use_gpu': False,
    'agent_seed': None,
    'use_wandb': False,
    'open_tensorboard': False,
    'test_in_train': False,
    'save_freq': 1000000,
}

print("=" * 60)
print("Testing Causal RL (advantage replacement) on Two-Stage")
print("=" * 60)
env = create_two_stage_env(causal_rl=True)
env.training_run(length=10, args_dict=args_dict)
h = env.training_history

if h and "mean_returns" in h:
    means = [round(float(x), 2) for x in h["mean_returns"]]
    print(f"\nMean returns: {means}")
    first5 = np.mean(h["mean_returns"][:5])
    last5 = np.mean(h["mean_returns"][-5:])
    best = np.max(h["mean_returns"])
    print(f"First 5 mean:  {first5:.2f}")
    print(f"Last 5 mean:   {last5:.2f}")
    print(f"Best epoch:    {best:.2f}")
    if last5 > first5:
        print("PASS: Causal RL is improving")
    else:
        print("WARN: Not improving (might need more epochs or different seed)")
    # Check for catastrophic collapse
    peak_idx = np.argmax(h["mean_returns"])
    post_peak = h["mean_returns"][peak_idx:]
    if len(post_peak) > 3 and np.mean(post_peak[-3:]) < 0.5 * best:
        print("FAIL: Catastrophic collapse detected after peak!")
    else:
        print("PASS: No catastrophic collapse")
else:
    print("WARNING: No training history returned")

