"""
Tests for the gym simulator.
"""
import sys
import copy
import numpy as np
sys.path.append("C:/Users/20183272/OneDrive - TU Eindhoven/Documents/GitHub/gympn_project")

import torch
from gympn.networks import HeteroActor
from torch_geometric.nn.conv.han_conv import HANConv
from torch_geometric.nn.aggr.basic import SumAggregation

# Add all necessary classes to safe globals
torch.serialization.add_safe_globals([HeteroActor,HANConv,SumAggregation])

from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from simpn.reporters import SimpleReporter
from gympn.solvers import HeuristicSolver, GymSolver, RandomSolver


def run_experiments(problem, solver, num_experiments, reporter=None, length=None):
    rewards = []
    for i in range(num_experiments):
        # Create a fresh copy of the problem
        problem_copy = copy.deepcopy(problem)

        # Run the experiment
        reward = problem_copy.testing_run(solver, reporter=reporter, length=length)
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards)

def test_training():
    """
    Test a training in the Gym Simulator.
    """

    train = False  # if True, train a model, else test the trained model
    test_episodes = 1000
    path_to_trained_model = "C:/Users/lobia/PycharmProjects/gympn/data/train/2025-10-28-18-07-02_run/best_policy.pth"


    # Instantiate a simulation problem.
    supply_chain = GymProblem(allow_postpone=True)

    # Define ordering variables
    supply_pool = supply_chain.add_var("supply_pool", var_attributes=['product_type'])
    ordered = supply_chain.add_var("ordered", var_attributes=['product_type'])  # orders stay are dispatched and their existence can be observed (for now)
    ordering_budget = supply_chain.add_var("ordering_budget", var_attributes=['budget'])

    # Define stock variables
    stock_phone_cases = supply_chain.add_var("stock_phone_cases", var_attributes=['id'])
    stock_chips = supply_chain.add_var("stock_chips", var_attributes=['id'])
    stock_game_cases = supply_chain.add_var("stock_game_cases", var_attributes=['id'])
    stock_phone_NL = supply_chain.add_var("stock_phone_NL", var_attributes=['id'])
    stock_game_NL = supply_chain.add_var("stock_game_NL", var_attributes=['id'])
    stock_phone_DE = supply_chain.add_var("stock_phone_DE", var_attributes=['id'])

    # Define demand variables
    demand = supply_chain.add_var("demand", var_attributes=['node', "count"])
    # Put initial values in the variables
    supply_pool.put({'product_type': 0})  # phone cases
    supply_pool.put({'product_type': 1})  # chips
    supply_pool.put({'product_type': 2})  # game cases
    supply_pool.put({'product_type': 3})  # external phone order
    supply_pool.put({'product_type': 4})  # external game order


    ordering_budget.put({'budget': 6})

    stock_phone_cases.put({'phone_cases_id': 1})
    stock_chips.put({'chips_id': 1})
    stock_game_cases.put({'game_cases_id': 1})

    demand.put({'node': 0, 'count': 0})

    # Define events.
    def order(s, b):
        po = s['product_type']

        return [SimToken({'product_type': po}), SimToken({'budget': b['budget'] - 1}), SimToken(
            {'product_type': po})]  # TODO: check if task assignment works like this and not like tuple (s,b)

    # Ordering action
    supply_chain.add_action([supply_pool, ordering_budget], [supply_pool, ordering_budget, ordered], behavior=order,
                            guard=lambda x, y: y['budget'] > 0)
    supply_chain.add_event([ordering_budget], [ordering_budget], lambda x: [SimToken({'budget': 6}, delay=1)],
                           name="replenish_budget", guard=lambda x: x['budget'] == 0)

    # Execute PO action
    supply_chain.add_event([ordered], [stock_phone_cases], lambda x: [SimToken(x, delay=1)],
                           name="execute_phone_case_po", guard=lambda x: x['product_type'] == 0)
    supply_chain.add_event([ordered], [stock_chips], lambda x: [SimToken(x, delay=2)], name="execute_chips_po",
                           guard=lambda x: x['product_type'] == 1)
    supply_chain.add_event([ordered], [stock_game_cases], lambda x: [SimToken(x, delay=1)], name="execute_game_case_po",
                           guard=lambda x: x['product_type'] == 2)
    supply_chain.add_event([ordered], [stock_phone_NL], lambda x: [SimToken(x, delay=1)], name="execute_external_phone_order",
                           guard=lambda x: x['product_type'] == 3)
    supply_chain.add_event([ordered], [stock_game_NL], lambda x: [SimToken(x, delay=1)], name="execute_external_game_order",
                           guard=lambda x: x['product_type'] == 4)

    # Assemble actions
    supply_chain.add_action([stock_phone_cases, stock_chips], [stock_phone_NL],
                            behavior=lambda x, y: [SimToken({'product_type': 4}, delay=1)],
                            name="produce_phone")
    supply_chain.add_action([stock_game_cases, stock_chips], [stock_game_NL],
                            behavior=lambda x, y: [SimToken({'product_type': 5}, delay=1)],
                            name="produce_game")

    # Transportation actions
    supply_chain.add_action([stock_phone_NL], [stock_phone_DE],
                            behavior=lambda x: [SimToken({'product_type': 4}, delay=1)],
                            name="transport_phone") # TODO: how not to act

    # Demand fulfillment

    supply_chain.add_event([stock_phone_NL, demand], [demand],
                           behavior= lambda x,y: [SimToken({'node': 0, 'count': y['count'] - 1})],
                           guard= lambda x,y: y['node'] == 0 and y['count'] > 0, name="demand_phone_NL",
                           reward_function=lambda x,y: 1)
    supply_chain.add_event([stock_phone_DE, demand], [demand],
                           behavior= lambda x,y: [SimToken({'node': 1, 'count': y['count'] - 1})],
                           guard= lambda x,y: y['node'] == 1 and y['count'] > 0, name="demand_phone_DE",
                           reward_function=lambda x,y: 1)
    supply_chain.add_event([stock_game_NL, demand], [demand],
                           behavior= lambda x,y: [SimToken({'node': 2, 'count': y['count'] - 1})],
                           guard= lambda x,y: y['node'] == 2 and y['count'] > 0, name="demand_game_NL",
                           reward_function=lambda x,y: 1)

    def demand_function(d):
        # Generate next turn demand
        node = int(np.random.randint(0, 3))
        count = int(np.random.randint(1, 7))

        return [SimToken({'node': node, 'count': count}, delay=1)]


    supply_chain.add_event([demand], [demand],
                           behavior = demand_function,
                           #guard = lambda x: x['count'] == 0, #I don't get this guard... and it makes the system go into deadlock when using postpone!
                           name="demand_next_round")

    # Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    supply_chain.set_unobservable(simvars = ['demand'], token_attrs = {'demand': ['node', 'count']})


    def demand_aware_heuristic(pn, actions_dict):
        # Get current demand
        demand_tokens = HeuristicSolver.get_place_tokens(place_id='demand', pn=pn)
        demand_counts = {0: 0, 1: 0, 2: 0}
        for t in demand_tokens:
            node = t['node']
            count = t['count']
            demand_counts[node] += count

        # Prioritize fulfilling demand directly
        if 'demand_phone_NL' in actions_dict and demand_counts[0] > 0:
            return {'demand_phone_NL': actions_dict['demand_phone_NL'][0]}
        if 'demand_phone_DE' in actions_dict and demand_counts[1] > 0:
            return {'demand_phone_DE': actions_dict['demand_phone_DE'][0]}
        if 'demand_game_NL' in actions_dict and demand_counts[2] > 0:
            return {'demand_game_NL': actions_dict['demand_game_NL'][0]}

        # If we can produce something, do it
        for action in ['produce_phone', 'produce_game']:
            if action in actions_dict:
                return {action: actions_dict[action][0]}

        # If we can order and have budget, order the most needed item
        if 'order' in actions_dict:
            # Estimate which product type is most needed
            # (e.g., based on stock levels or demand)
            # For now, just alternate between chips and phone cases
            return {'order': actions_dict['order'][0]}

        # Fallback: pick any available action
        for k, v in actions_dict.items():
            return {k: v[0]}

    # Default training arguments (change them as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "gam": 1,
        "lam": 0.99,
        "eps": 0.2,
        "c": 0.2,
        "ent_bonus": 0.1,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 64]},
        "policy_lr": 3e-4,
        "policy_updates": 4,
        "policy_kld_limit": 0.5,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 64]},
        "value_lr": 3e-4,
        "value_updates": 10,
        "value_weights": "",

        # Training Parameters
        "episodes": 20,
        "epochs": 100,
        "max_episode_length": None,
        "batch_size": 64,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 0,

        # Saving Parameters
        "name": "run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False, # Open TensorBoard during training (defaults to False)
    }


    frozen_pn = copy.deepcopy(supply_chain)

    if train:
        supply_chain.training_run(length=10, args_dict=default_args)


    supply_chain = copy.deepcopy(frozen_pn)
    solver_random = RandomSolver()
    random_average, random_std = run_experiments(supply_chain, solver_random, test_episodes, length=10)

    supply_chain = copy.deepcopy(frozen_pn)
    solver_heuristic = HeuristicSolver(heuristic_function=demand_aware_heuristic)
    heuristic_average, heuristic_std = run_experiments(supply_chain, solver_heuristic, test_episodes, length=10)

    supply_chain = copy.deepcopy(frozen_pn)
    w_p = path_to_trained_model
    solver_test = GymSolver(weights_path=w_p, metadata=supply_chain.make_metadata())
    supply_chain.set_solver(solver_test)
    ppo_average, ppo_std = run_experiments(supply_chain, solver_test, test_episodes, length=10)


    print("--------------------------------")
    print("Summary of results:")
    print(f'Random policy: average {random_average}, std {random_std}')
    print(f'Heuristic policy: average {heuristic_average}, std {heuristic_std}')
    print(f'PPO policy: average {ppo_average}, std {ppo_std}')
    print("--------------------------------")

if __name__ == "__main__":
    test_training()
