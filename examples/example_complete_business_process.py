"""
This file implements a complete business process with multiple decision points.
The goal is to assign tasks to employees and to decide between alternative task flows in a way that minimizes the total time taken to complete all tasks.
The simulation uses a heuristic solver to assign tasks to employees based on their availability, as well as DRL and a random policy.
"""
import copy
import os

import numpy as np
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.visualisation import Visualisation
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    ###########################################################################
    # Run configurations
    train = False #set to False to test a trained model
    run_name = '2025-10-02-11-14-25_run' #specify the run name to load the weights from
    num_experiments = 1000 #number of test experiments to run (if train=False)
    visualize_random = False  # Set to True to visualize the random solver
    visualize_ppo = False  # Set to True to visualize the PPO solver

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth") #customize if needed

    ###########################################################################

    # Instantiate a simulation problem.
    agency = GymProblem()

    # Define cases.
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy_register_application = agency.add_var("busy_register_application", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})

    # Define choice
    waiting_choice = agency.add_var("waiting_choice", var_attributes=['task_type'])
    busy_simple_product = agency.add_var("busy_simple_product", var_attributes=['task_type', 'resource_id'])
    busy_complex_product = agency.add_var("busy_complex_product", var_attributes=['task_type', 'resource_id'])

    # Define rework with 20% probability
    rework_junior_stage = agency.add_var("rework_junior_stage", var_attributes=['task_type'])
    rework_senior_stage = agency.add_var("rework_senior_stage", var_attributes=['task_type'])

    # Define draft proposal
    draft_proposal_stage = agency.add_var("draft_proposal_stage", var_attributes=['task_type'])
    busy_draft_proposal = agency.add_var("busy_draft_proposal", var_attributes=['task_type', 'resource_id'])

    # Define resources.
    junior_employee = agency.add_var("junior_employee", var_attributes=['code_employee'])
    junior_employee.put({'code_employee': 0})
    junior_employee.put({'code_employee': 0})

    senior_employee = agency.add_var("senior_employee", var_attributes=['code_employee'])
    senior_employee.put({'code_employee': 0})
    senior_employee.put({'code_employee': 0})


    # Define events.
    def arrive(a):
        return [SimToken(a, delay=random.expovariate(3)), SimToken(a)]


    agency.add_event([arrival], [arrival, waiting], arrive)


    def start_register_application(c, r):
        """"
        This function is called on the event that assigns a the application to register to a junior employee.
        :param c: the task
        :param r: the resource
        :return: a list of SimTokens representing the task and the resource that were assigned to them
        """
        return [SimToken((c, r), delay=1 / 10)]  # delay=random.expovariate(10))]


    agency.add_event([waiting, junior_employee], [busy_register_application], behavior=start_register_application,
                     name="start")


    def complete_register_application(b):
        """
        This function is called when the registration of an application is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[1]), SimToken(b[0])]


    agency.add_event([busy_register_application], [junior_employee, waiting_choice], complete_register_application,
                     name='complete', reward_function=lambda x: 1)


    # The decision point - choice of the next task and which resource pool to use
    def choice_simple_product(c, r):
        return [SimToken((c, r), delay=1 / 5)]  # delay=random.expovariate(5))]


    def choice_complex_product(c, r):
        return [SimToken((c, r), delay=1 / 2.5)]  # delay=random.expovariate(2.5))]


    agency.add_action([waiting_choice, junior_employee], [busy_simple_product], behavior=choice_simple_product,
                      name="choice_simple_product")
    agency.add_action([waiting_choice, senior_employee], [busy_complex_product], behavior=choice_complex_product,
                      name="choice_complex_product")


    def complete_product(b):
        """
        This function is called when a simple product task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[1]), SimToken(b[0])]


    agency.add_event([busy_simple_product], [junior_employee, rework_junior_stage], complete_product,
                     name='complete_simple_product')
    agency.add_event([busy_complex_product], [senior_employee, rework_senior_stage], complete_product,
                     name='complete_complex_product')


    def rework_junior(b):
        prob = random.uniform(0, 1)
        if prob > 0.6:
            return [SimToken(b), None]
        else:
            return [None, SimToken(b)]


    agency.add_event([rework_junior_stage], [waiting, draft_proposal_stage], rework_junior, name='rework_junior')


    def rework_senior(b):
        prob = random.uniform(0, 1)
        if prob > 0.9:
            return [SimToken(b), None]
        else:
            return [None, SimToken(b)]


    agency.add_event([rework_senior_stage], [waiting, draft_proposal_stage], rework_senior, name='rework_senior')


    def draft_proposal(b, r):
        return [SimToken((b, r), delay=1 / 2.5)]  # delay=random.expovariate(2.5))]


    agency.add_event([draft_proposal_stage, junior_employee], [busy_draft_proposal], draft_proposal,
                     name="draft_proposal_event")


    def final_complete_case(b):
        return [SimToken(b[1])]


    agency.add_event([busy_draft_proposal], [junior_employee], behavior=final_complete_case,
                     name="complete_draft_proposal", reward_function=lambda x: 1)

    ###########################################################################

    # Default training arguments (change them as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "gam": 1, # With finite horizon, it is better to use gam=1
        "lam": 0.99,
        "eps": 0.2,
        "c": 0.2,
        "ent_bonus": 0.0,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [64]},
        "policy_lr": 3e-4,
        "policy_updates": 2,
        "policy_kld_limit": 0.01,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [64]},
        "value_lr": 3e-4,
        "value_updates": 10,
        "value_weights": "",

        # Training Parameters
        "episodes": 100, #TODO: add a warning to the batch collection to hint if no complete batch is available
        "epochs": 200,
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


    if train:
        #training functions
        agency.training_run(length=10, args_dict=default_args)

    else:
        def run_experiments(problem, solver, num_experiments, reporter=None, length=None):
            rewards = []
            for i in range(num_experiments):
                # Create a fresh copy of the problem
                problem_copy = copy.deepcopy(problem)

                # Run the experiment
                reward = problem_copy.testing_run(solver, reporter=reporter, length=length)
                rewards.append(reward)
            return np.mean(rewards), np.std(rewards)


        if visualize_random:
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.length = 100 #TODO: parameterize in a better way
            frozen_agency.set_solver(RandomSolver())
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            solver = RandomSolver()
            random_average, random_std = run_experiments(agency, solver, num_experiments, length=10)

            print(f"Random solver average reward: {random_average}, std: {random_std}")

        if visualize_ppo:
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.set_solver(GymSolver(weights_path=weights_path, metadata=agency.make_metadata()))
            frozen_agency.length = 100
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
            ppo_average, ppo_std = run_experiments(agency, solver, num_experiments, length=10)
            print(f"DRL solver average reward: {ppo_average}, std: {ppo_std}")

        if not visualize_random and not visualize_ppo:

            #perform z-test to verify the statistical significance of the difference in average rewards
            from math import sqrt
            from scipy.stats import norm

            def check_statistical_significance(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
                # Calculate the z-score
                z = (mean1 - mean2) / sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

                # Calculate the p-value (two-tailed test)
                p_value = 2 * norm.sf(abs(z))

                # Check significance
                significant = p_value < alpha

                return z, p_value, significant

            z, p_value, significant = check_statistical_significance(ppo_average, ppo_std, num_experiments, random_average, random_std, num_experiments)
            print(f"Z-score: {z}, P-value: {p_value}, Statistically Significant: {significant}")
            print(f"The percentage difference between the two average rewards is {100 * (ppo_average - random_average) / abs(random_average)}%")
