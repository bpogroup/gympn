"""End-to-end smoke test: run a couple of causal-RL training epochs with the
new flow_dag default to confirm the redistribution wires through the pipeline.

Run: python _smoke_flow_dag.py
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from simpn.simulator import SimToken
from gympn.simulator import GymProblem

agency = GymProblem(allow_postpone=True, causal_rl=True)

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


agency.add_event([busy_B], [resource], complete_B, name='complete_B',
                 reward_function=lambda x: 1)

args = {
    "algorithm": "ppo-clip",
    "gam": 0.99, "lam": 0.95, "eps": 0.2, "c": 0.2, "ent_bonus": 0.01,
    "agent_seed": 0,
    "policy_model": "gnn", "policy_kwargs": {"hidden_layers": [32]},
    "policy_lr": 3e-4, "policy_updates": 2, "policy_kld_limit": 0.2,
    "policy_weights": "", "policy_network": "", "score": False, "score_weight": 1e-3,
    "value_model": "gnn", "value_kwargs": {"hidden_layers": [32]},
    "value_lr": 3e-4, "value_updates": 3, "value_weights": "", "vf_coeff": 0.5,
    "episodes": 4, "epochs": 2, "max_episode_length": None, "batch_size": 32,
    "sort_states": False, "use_gpu": False, "load_policy_network": False, "verbose": 1,
    "name": "smoke_flow_dag", "datetag": True, "logdir": "data/train",
    "save_freq": 1, "open_tensorboard": False, "use_wandb": False,
    "causal_rl": True, "causal_scheme": "flow_dag",
}

if __name__ == "__main__":
    print("causal_scheme =", args["causal_scheme"])
    agency.training_run(length=10, args_dict=args)
    print("\nSMOKE OK: flow_dag causal training completed end-to-end.")