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
    w_p = "C:/Users/20215143/tue_repos/gympn/examples/data/train/2025-07-09-17-35-48_run/best_policy.pth"


    # Instantiate a simulation problem.
    supply_chain = GymProblem()

    # Define ordering variables
    supply_pool = supply_chain.add_var("supply_pool", var_attributes=['product_type'])
    ordered = supply_chain.add_var("ordered", var_attributes=['product_type'])  # orders stay are dispatched and their existence can be observed (for now)
    ordering_budget = supply_chain.add_var("ordering_budget", var_attributes=['budget'])

    # Define stock variables
    stock_phone_cases = supply_chain.add_var("stock_phone_cases", var_attributes=['id'])
    stock_chips = supply_chain.add_var("stock_chips", var_attributes=['id'])
    stock_game_cases = supply_chain.add_var("stock_game_cases", var_attributes=['id'])
    stock_phone_NL = supply_chain.add_var("stock_phone_NL", var_attributes=['id']) #can transport is a boolean variable that indicates if the stock can be transported to DE
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
                            name="transport_phone") # TODO: how not to act (for now, it only postpones the availability of the inventory in NL)

    # Do not transport with delay (maybe this breaks the heuristic?)
    def not_transport_phone(s):
        """
        This function is called when the phone stock is not transported.
        :param s: the stock of phones in NL
        :return: a list of SimTokens representing the stock of phones in NL
        """
        tb, is_active, computed_delay = supply_chain.bindings()
        # The delay is the

        return [SimToken(s, delay=computed_delay)]

    #supply_chain.add_action([stock_phone_NL], [stock_phone_NL], behavior=not_transport_phone,
    #                        name="not_transport_phone")

    # Do not transport with extra place
    stock_phone_NL_on_hold = supply_chain.add_var("stock_phone_NL_on_hold", var_attributes=['product_type'])
    supply_chain.add_action([stock_phone_NL], [stock_phone_NL_on_hold], behavior=not_transport_phone,
                            name="not_transport_phone")
    supply_chain.add_action([stock_phone_NL_on_hold], [stock_phone_NL], behavior= lambda x: [SimToken(x, delay=0)], name="make_stock_available")

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

    def demand_function(a, d):
        # Generate next turn demand
        node = int(np.random.randint(0, 3))
        count = int(np.random.randint(1, 7))

        return [a, SimToken({'node': node, 'count': count}, delay=1)]


    supply_chain.add_event([stock_phone_NL.queue, demand], [stock_phone_NL.queue, demand],
                           behavior = demand_function,
                           guard = lambda x, y: y['count'] == 0,
                           name="demand_next_round")

    # Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    supply_chain.set_unobservable(simvars = ['demand'], token_attrs = {'demand': ['node', 'count']})


    def balanced_heuristic(pn, actions_dict):
        """
        A heuristic that balances the production of phones and games.
        :param pn: the Petri net.
        :param actions_dict: the available actions.
        :return: a dictionary with the action to take.
        """
        # Get current stock levels by checking the marking length of each place
        ordered_token_values = HeuristicSolver.get_place_tokens(place_id='ordered', pn=pn)
        phone_cases_stock_values = [t.value for p in pn.places if p._id == 'stock_phone_cases' for t in p.marking]
        chips_stock_values = [t.value for p in pn.places if p._id == 'stock_chips' for t in p.marking]
        game_cases_stock_values = [t.value for p in pn.places if p._id == 'stock_game_cases' for t in p.marking]

        # Get total expected inventory (current stock + orders in transit)
        ordered_phone_cases = len([t for t in ordered_token_values if t['product_type'] == 0])
        ordered_chips = len([t for t in ordered_token_values if t['product_type'] == 1])
        ordered_game_cases = len([t for t in ordered_token_values if t['product_type'] == 2])

        total_phone_cases = len(phone_cases_stock_values) + ordered_phone_cases
        total_chips = len(chips_stock_values) + ordered_chips
        total_game_cases = len(game_cases_stock_values) + ordered_game_cases

        # Get total inventory of completed phones NL, completed games NL, and completed phones DE
        phone_NL_stock_values = [t.value for p in pn.places if p._id == 'stock_phone_NL' for t in p.marking]
        game_NL_stock_values = [t.value for p in pn.places if p._id == 'stock_game_NL' for t in p.marking]
        phone_DE_stock_values = [t.value for p in pn.places if p._id == 'stock_phone_DE' for t in p.marking]

        # Get total ordering budget
        ordering_budget_value = [t.value for p in pn.places if p._id == 'ordering_budget' for t in p.marking][0]['budget']

        # If we can order, first we do that
        if 'order' in actions_dict.keys():

            # if we have less than 3 chips, order chips
            if ordering_budget_value > 3:
                ret_val = HeuristicSolver.get_attribute_based_assignments(action="order",
                                                                          attribute_dict={'product_type': 1},
                                                                          actions_dict=actions_dict)
                return {'order': ret_val[0]}

            elif total_phone_cases <= 2*total_game_cases:
                # order phone cases if we have more phone cases 2*game cases
                ret_val = HeuristicSolver.get_attribute_based_assignments(action="order",
                                                                          attribute_dict={'product_type': 0},
                                                                          actions_dict=actions_dict)
                return {'order': ret_val[0]}
            elif 2*total_game_cases < total_phone_cases:
                # order game cases otherwise
                ret_val = HeuristicSolver.get_attribute_based_assignments(action="order",
                                                                          attribute_dict={'product_type': 2},                                                  actions_dict=actions_dict)
                return {'order': ret_val[0]}

        # If we can produce phones, we do
        if 'produce_phone' in actions_dict.keys() and (len(phone_NL_stock_values) <= len(game_NL_stock_values) or 'produce_game' not in actions_dict.keys()):
            assignable = actions_dict['produce_phone']

            return {'produce_phone': assignable[0]}
        elif 'produce_game' in actions_dict.keys():
            assignable = actions_dict['produce_game']

            # If we can only produce games, do that
            return {'produce_game': assignable[0]}


        #TRANSPORTATION
        if 'transport_phone' in actions_dict.keys() and len(phone_NL_stock_values) > len(phone_DE_stock_values):
            assignable = actions_dict['transport_phone']
            return {'transport_phone': assignable[0]}
        elif 'not_transport_phone' in actions_dict.keys() and len(phone_NL_stock_values) <= len(phone_DE_stock_values):
            assignable = actions_dict['not_transport_phone']
            return {'not_transport_phone': assignable[0]}

        return

    # Default training arguments (change them as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "gam": 1,
        "lam": 0.99,
        "eps": 0.2,
        "c": 0.2,
        "ent_bonus": 0.01,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128]},
        "policy_lr": 3e-4,
        "policy_updates": 4,
        "policy_kld_limit": 0.1,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 3e-4,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128]},
        "value_lr": 3e-4,
        "value_updates": 10,
        "value_weights": "",

        # Training Parameters
        "episodes": 20,
        "epochs": 500,
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
        "open_tensorboard": True,  # Open TensorBoard during training (defaults to False)
    }

    frozen_pn = copy.deepcopy(supply_chain)

    if train:
        supply_chain.training_run(length=10, args_dict=default_args)

    import numpy as np

    def run_experiments(problem, solver, num_experiments, reporter=None, length=10):
        """
        Run multiple experiments with a given solver and return the mean and standard deviation of rewards.
        :param problem: The supply chain problem instance.
        :param solver: The solver to use (e.g., RandomSolver, HeuristicSolver, GymSolver).
        :param num_experiments: Number of experiments to run.
        :param reporter: Optional reporter to log results.
        :param length: Optional length of each experiment.
        :return: Mean and standard deviation of total rewards.
        """
        rewards = []
        for _ in range(num_experiments):
            problem_copy = copy.deepcopy(problem)
            problem_copy.length = length if length is not None else 10
            problem_copy.set_solver(solver)
            #visual = Visualisation(problem_copy)
            #visual.show()
            reward = problem_copy.testing_run(solver, reporter=reporter, length=length)

            rewards.append(reward)
        return np.mean(rewards), np.std(rewards)

    # Number of experiments to run for each method
    num_experiments = 10

    # Run experiments for RandomSolver
    if random:
        mean_r, std_r = run_experiments(frozen_pn, RandomSolver(), num_experiments, reporter=SimpleReporter())
        print(f'Random policy: Mean = {mean_r}, Std = {std_r}')

    # Run experiments for HeuristicSolver
    if heuristic:
        mean_h, std_h = run_experiments(frozen_pn, HeuristicSolver(balanced_heuristic), num_experiments,
                                        reporter=SimpleReporter())
        print(f'Heuristic policy: Mean = {mean_h}, Std = {std_h}')

    # Run experiments for GymSolver
    if test:
        solver_test = GymSolver(weights_path=w_p, metadata=frozen_pn.make_metadata())
        mean_p, std_p = run_experiments(frozen_pn, solver_test, num_experiments, reporter=SimpleReporter(), length=10)
        print(f'PPO policy: Mean = {mean_p}, Std = {std_p}')

    # Summary of results
    print("--------------------------------")
    print("Summary of results:")
    if random:
        print(f'Random policy: Mean = {mean_r}, Std = {std_r}')
    if heuristic:
        print(f'Heuristic policy: Mean = {mean_h}, Std = {std_h}')
    if test:
        print(f'PPO policy: Mean = {mean_p}, Std = {std_p}')
    print("--------------------------------")

if __name__ == "__main__":
    test_training()
