"""
This file implements a complete business process with multiple decision points.
The goal is to assign tasks to employees and to decide between alternative task flows in a way that minimizes the total time taken to complete all tasks.
The simulation uses a heuristic solver to assign tasks to employees based on their availability, as well as DRL and a random policy.
"""
import copy
import os
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
    train = True #set to False to test a trained model
    run_name = '2025-09-29-13-23-43_run'
    visualize_random = False  # Set to True to visualize the random solver
    visualize_heuristic = False # Set to True to visualize the heuristic solver
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


    # define perfect heuristic
    def perfect_heuristic(observable_net, tokens_comb):
        """
        This heuristic function selects the best binding based on the task type and resources code.
        :param observable_net: the observable net.
        :param tokens_comb: the list of all possible bindings.
        :return: the best binding.
        """
        # The perfect heuristic is to always assign the task to the resources that can do it the fastest (so taks type 0 to resource 0 and task type 1 to resource 1)
        for k, el in tokens_comb.items():
            for binding in el:
                task = binding[0][1].value
                resource = binding[1][1].value
                if task['task_type'] == resource['code_employee']:
                    return {k: binding}


    if train:
        #training functions
        agency.training_run(length=10, args_dict=default_args)

    else:
        random_average = 0
        random_std = 0
        random_reward = []
        if visualize_random:
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.length = 100 #TODO: parameterize in a better way
            frozen_agency.set_solver(RandomSolver())
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            for i in range(100):
                frozen_agency = copy.deepcopy(agency)
                res = frozen_agency.testing_run(length=100, solver=RandomSolver())
                random_reward.append(res)
                random_average += res
                random_std += res ** 2

            random_average /= 100
            random_std = (random_std / 100 - random_average ** 2) ** 0.5
            print(f"Random solver average reward: {random_average}, std: {random_std}")

        #heuristic_average = 0
        #heuristic_std = 0
        #heuristic_reward = []
        #for i in range(10):
        #    frozen_agency = copy.deepcopy(agency)
        #    solver = HeuristicSolver(perfect_heuristic)
        #    res = frozen_agency.testing_run(length=10, solver=solver)
        #    heuristic_reward.append(res)
        #    heuristic_average += res
        #    heuristic_std += res ** 2

        #heuristic_average /= 10
        #heuristic_std = (heuristic_std / 10 - heuristic_average ** 2) ** 0.5
        #print(f"Heuristic solver average reward: {heuristic_average}, std: {heuristic_std}")



        ppo_average = 0
        ppo_std = 0
        ppo_reward = []

        if visualize_ppo:
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.set_solver(GymSolver(weights_path=weights_path, metadata=agency.make_metadata()))
            frozen_agency.length = 100
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            for i in range(100):
                frozen_agency = copy.deepcopy(agency)
                solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
                res = agency.testing_run(length=100, solver=solver)
                ppo_reward.append(res)
                ppo_average += res
                ppo_std += res ** 2

            ppo_average /= 100
            ppo_std = (ppo_std / 100 - ppo_average ** 2) ** 0.5
            print(f"DRL solver average reward: {ppo_average}, std: {ppo_std}")

        if not visualize_random and not visualize_ppo:
            #create a boxplot
            import matplotlib.pyplot as plt
            data = [random_reward, ppo_reward]  # List of lists for boxplot
            labels = ['Random', 'DRL']  # Labels for the solvers

            fig, ax = plt.subplots()
            ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))

            ax.set_ylabel('Reward')
            ax.set_title('Reward Distribution for Different Solvers')
            plt.show()

            print("Run finished. If you want to visualize the results, set visualize_random or visualize_ppo to True.")
