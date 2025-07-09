"""
This file implements a task assignment problem with two employees and a queue of tasks.
The goal is to assign tasks to employees in a way that minimizes the total time taken to complete all tasks.
The simulation uses a heuristic solver to assign tasks to employees based on their availability, as well as DRL and a random policy.
"""
import copy
import os
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.visualisation import Visualisation

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    ###########################################################################
    # Run configurations
    train = True #set to False to test a trained model
    run_name = ''
    visualize_random = False  # Set to True to visualize the random solver
    visualize_heuristic = False # Set to True to visualize the heuristic solver
    visualize_ppo = True  # Set to True to visualize the PPO solver

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth") #customize if needed

    ###########################################################################

    # Instantiate a simulation problem.
    agency = GymProblem()

    # Define cases.
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy = agency.add_var("busy", var_attributes=['task_type', 'code_employee'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 1})

    # Define resources.
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})
    employee.put({'code_employee': 1})

    # Define events.
    def arrive(a):
        return [SimToken(a, delay=1), SimToken(a)]
    agency.add_event([arrival], [arrival, waiting], arrive)

    def start(c, r):
        """"
        This function is called when a task is assigned to an employee.
        :param c: the task
        :param r: the resource
        :return: a list of SimTokens representing the task and the resource that were assigned to them
        """
        if c['task_type'] == r['code_employee']:
            return [SimToken((c, r), delay=1)]
        else:
            return [SimToken((c, r), delay=2)]


    agency.add_action([waiting, employee], [busy], behavior=start, name="start")

    def complete(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[1])]

    agency.add_event([busy], [employee], complete, name='done', reward_function=lambda x: 1)



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
        "episodes": 10,
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
        "open_tensorboard": True, # Open TensorBoard during training (defaults to False)
    }


    # define perfect heuristic
    def perfect_heuristic(observable_net, tokens_comb):
        """
        This heuristic function selects the best binding based on the task type and employee code.
        :param observable_net: the observable net.
        :param tokens_comb: the list of all possible bindings.
        :return: the best binding.
        """
        # The perfect heuristic is to always assign the task to the employee that can do it the fastest (so taks type 0 to resource 0 and task type 1 to resource 1)
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
            frozen_agency.length = 10 #TODO: parameterize in a better way
            frozen_agency.set_solver(RandomSolver())
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            for i in range(10):
                frozen_agency = copy.deepcopy(agency)
                res = frozen_agency.testing_run(length=10, solver=RandomSolver())
                random_reward.append(res)
                random_average += res
                random_std += res ** 2

            random_average /= 10
            random_std = (random_std / 10 - random_average ** 2) ** 0.5
            print(f"Random solver average reward: {random_average}, std: {random_std}")

        heuristic_average = 0
        heuristic_std = 0
        heuristic_reward = []
        for i in range(10):
            frozen_agency = copy.deepcopy(agency)
            solver = HeuristicSolver(perfect_heuristic)
            res = frozen_agency.testing_run(length=10, solver=solver)
            heuristic_reward.append(res)
            heuristic_average += res
            heuristic_std += res ** 2

        heuristic_average /= 10
        heuristic_std = (heuristic_std / 10 - heuristic_average ** 2) ** 0.5
        print(f"Heuristic solver average reward: {heuristic_average}, std: {heuristic_std}")



        ppo_average = 0
        ppo_std = 0
        ppo_reward = []

        if visualize_ppo:
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.set_solver(GymSolver(weights_path=weights_path, metadata=agency.make_metadata()))
            frozen_agency.length = 10
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            for i in range(10):
                frozen_agency = copy.deepcopy(agency)
                solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
                res = agency.testing_run(length=10, solver=solver)
                ppo_reward.append(res)
                ppo_average += res
                ppo_std += res ** 2

            ppo_average /= 10
            ppo_std = (ppo_std / 10 - ppo_average ** 2) ** 0.5
            print(f"DRL solver average reward: {ppo_average}, std: {ppo_std}")

        if not visualize_random and not visualize_ppo:
            #create a boxplot
            import matplotlib.pyplot as plt
            data = [random_reward, heuristic_reward, ppo_reward]  # List of lists for boxplot
            labels = ['Random', 'Heuristic', 'DRL']  # Labels for the solvers

            fig, ax = plt.subplots()
            ax.boxplot(data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))

            ax.set_ylabel('Reward')
            ax.set_title('Reward Distribution for Different Solvers')
            plt.show()

            print("Run finished. If you want to visualize the results, set visualize_random or visualize_ppo to True.")
