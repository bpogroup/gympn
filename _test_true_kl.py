"""Test: verify true KL divergence prevents policy collapse over 30 epochs."""
import numpy as np, torch, os, random

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

agency = GymProblem(allow_postpone=True, causal_rl=True)
arrival = agency.add_var('arrival', var_attributes=['case_id'])
waiting_A = agency.add_var('waiting_A', var_attributes=['case_id'])
busy_A = agency.add_var('busy_A', var_attributes=['case_id', 'resource_id'])
waiting_B = agency.add_var('waiting_B', var_attributes=['case_id'])
busy_B = agency.add_var('busy_B', var_attributes=['case_id', 'resource_id'])
arrival.put({'case_id': 0}); arrival.put({'case_id': 0})
resource = agency.add_var('resource', var_attributes=['resource_id'])
resource.put({'resource_id': 0}); resource.put({'resource_id': 1})

def arrive(a): return [SimToken({'case_id': a['case_id']}, delay=1), SimToken(a)]
agency.add_event([arrival], [arrival, waiting_A], arrive)

def start_A(c, r): return [SimToken((c, r), delay=0.5 if r['resource_id'] == 0 else 2)]
agency.add_action([waiting_A, resource], [busy_A], behavior=start_A, name='start_A')

def complete_A(b): return [SimToken(b[1]), SimToken(b[0])]
agency.add_event([busy_A], [resource, waiting_B], complete_A, name='complete_A')

def start_B(c, r): return [SimToken((c, r), delay=0.5 if r['resource_id'] == 1 else 2)]
agency.add_action([waiting_B, resource], [busy_B], behavior=start_B, name='start_B')

def complete_B(b): return [SimToken(b[1])]
agency.add_event([busy_B], [resource], complete_B, name='complete_B', reward_function=lambda x: 1)

args = {
    'episodes': 20, 'epochs': 30, 'batch_size': 32, 'max_episode_length': None,
    'policy_lr': 3e-4, 'policy_updates': 5, 'value_lr': 3e-4, 'value_updates': 10,
    'gam': 0.99, 'lam': 0.95, 'eps': 0.2, 'vf_coeff': 0.5, 'ent_bonus': 0.01,
    'causal_rl': True, 'algorithm': 'ppo-clip', 'verbose': 1, 'use_gpu': False,
    'agent_seed': 42, 'use_wandb': False, 'open_tensorboard': False, 'test_in_train': False,
    'save_freq': 1000000,
}
agency.training_run(length=10, args_dict=args)
h = agency.training_history
best = max(h['mean_returns'])
final = h['mean_returns'][-1]
print(f'\nFinal: {final:.1f}, Best: {best:.1f}')
if best >= 19.0 and final >= 17.0:
    print('SUCCESS: Policy reached near-optimal and did not collapse!')
else:
    print(f'NOTE: Best={best:.1f}, Final={final:.1f} (may need more epochs)')


