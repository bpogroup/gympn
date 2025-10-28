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
from gympn.visualisation import Visualisation
from gympn.utils import sim_tokens_values_from_bindings, binding_from_tokens_values


def test_training():
    """
    Test a training in the Gym Simulator.
    """

    train = False  # if True, train a model, else test the trained model
    test = True
    random = True
    heuristic = True

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

    def max_phone_production(pn, actions_dict):
        # Get current stock levels by checking the marking length of each place
        ordered_token_values = HeuristicSolver.get_place_tokens(place_id='ordered', pn=pn)
        phone_cases_stock_values = [t.value for p in pn.places if p._id == 'stock_phone_cases' for t in p.marking]
        chips_stock_values = [t.value for p in pn.places if p._id == 'stock_chips' for t in p.marking]

        # Get total expected inventory (current stock + orders in transit)
        ordered_phone_cases = len([t for t in ordered_token_values if t['product_type'] == 0]) #(2) get_place_tokens_type(place_id = 'ordered', type = 0). Probably best without len
        ordered_chips = len([t for t in ordered_token_values if t['product_type'] == 1])

        total_phone_cases = len(phone_cases_stock_values) + ordered_phone_cases
        total_chips = len(chips_stock_values) + ordered_chips

        # If we can produce phones, always do that
        if 'produce_phone' in actions_dict.keys():
            assignable = actions_dict['produce_phone'] #(3) get_available_assignments(action)

            return {'produce_phone': assignable[0]}

        # If we can't produce phones but can order
        if 'order' in actions_dict.keys():
            assignable = actions_dict['order']

            if total_phone_cases >= total_chips:  # order chips if we have more phone cases than chips
                # for i in assignable:
                #    if i[0][1].value['product_type'] == 1:
                #        return {'order': i} #(4) attribute_based_assignments(action="order", attribute="product_type", value=1)
                ret_val = HeuristicSolver.get_attribute_based_assignments(action="order",
                                                                          attribute_dict={'product_type': 1},
                                                                          actions_dict=actions_dict)
                return {'order': ret_val[0]}
            else:
                # for i in assignable:
                #    if i[0][1].value['product_type'] == 0:
                #        return {'order': i}
                ret_val = HeuristicSolver.get_attribute_based_assignments(action="order",
                                                                          attribute_dict={'product_type': 0},
                                                                          actions_dict=actions_dict)
                return {'order': ret_val[0]}

        else:
            #return a random binding (the first in list)
            k = list(actions_dict.keys())[0]
            v = actions_dict[k][0]
            return {k: v}


    # Default training arguments (change them as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "gam": 1,
        "lam": 0.99,
        "eps": 0.2,
        "c": 0.2,
        "ent_bonus": 0.05,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 64]},
        "policy_lr": 1e-4,
        "policy_updates": 4,
        "policy_kld_limit": 0.5,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 64]},
        "value_lr": 1e-4,
        "value_updates": 20,
        "value_weights": "",

        # Training Parameters
        "episodes": 100,
        "epochs": 100,
        "max_episode_length": None,
        "batch_size": 32,
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

    res_r = 0
    res_h = 0
    res_p = 0

    frozen_pn = copy.deepcopy(supply_chain)

    if train:
        supply_chain.training_run(length=10, args_dict=default_args)

    if random:
        supply_chain = copy.deepcopy(frozen_pn)
        solver_random = RandomSolver()
        reporter = SimpleReporter()
        res_r = supply_chain.testing_run(solver_random, reporter=reporter)
        print(f'Random policy: {res_r}')

    if heuristic:
        supply_chain = copy.deepcopy(frozen_pn)
        solver_heuristic = HeuristicSolver(max_phone_production)
        reporter = SimpleReporter()
        res_h = supply_chain.testing_run(solver_heuristic, reporter=reporter)
        print(f'Heuristic policy: {res_h}')

    if test:
        supply_chain = copy.deepcopy(frozen_pn)
        w_p = "C:/Users/lobia/PycharmProjects/gympn/data/train/2025-10-24-14-24-42_run/best_policy.pth"
        solver_test = GymSolver(weights_path=w_p, metadata=supply_chain.make_metadata())
        reporter = SimpleReporter()
        supply_chain.set_solver(solver_test)
        #visual = Visualisation(supply_chain)
        #visual.show()
        res_p = supply_chain.testing_run(solver_test, reporter=reporter, length=10)
        print(f'PPO policy: {res_p}')
    
    print("--------------------------------")
    print("Summary of results:")
    print(f'Random policy: {res_r}')
    print(f'Heuristic policy: {res_h}')      
    print(f'PPO policy: {res_p}')
    print("--------------------------------")

if __name__ == "__main__":
    test_training()
