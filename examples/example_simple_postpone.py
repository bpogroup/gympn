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
from gympn.logging_utils import get_logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":

    # Initialize logger
    logger = get_logger(verbose=1)

    ###########################################################################
    # Run configurations
    train = True #set to False to test a trained model
    run_name = '2025-12-17-09-40-01_run'
    visualize_random = False  # Set to True to visualize the random solver
    visualize_heuristic = False # Set to True to visualize the heuristic solver
    visualize_ppo = False  # Set to True to visualize the PPO solver

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth") #customize if needed

    ###########################################################################

    # Instantiate a simulation problem.
    agency = GymProblem(allow_postpone=True, causal_rl=train)

    # Define cases.
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy = agency.add_var("busy", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 0})

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
        This function is called when a task is assigned to an resources.
        :param c: the task
        :param r: the resource
        :return: a list of SimTokens representing the task and the resource that were assigned to them
        """
        if r['code_employee']==1:
            return [SimToken((c, r), delay=20)]
        else:
            return [SimToken((c, r), delay=0.5)]


    agency.add_action([waiting, employee], [busy], behavior=start, name="start")

    def complete(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[1])]

    def r_function(x):

        if x[1]['code_employee'] == 1:
            return 1
        else:
            return 1

    agency.add_event([busy], [employee], complete, name='complete', reward_function=r_function)



    ###########################################################################
    # ALGORITHM SELECTION
    ###########################################################################
    # Choose which algorithm to use: "ppo-clip" (fast) or "dcl" (better planning)
    #ALGORITHM = "dcl"
    ALGORITHM = "ppo-clip"


    ###########################################################################
    # HYPERPARAMETER CONFIGURATIONS
    ###########################################################################

    # PPO-clip configuration: Fast training, standard policy gradient
    ppo_config = {
        "algorithm": "ppo-clip",
        "gam": 1,
        "lam": 0.99,
        "eps": 0.15,             # Tighter PPO clipping for stability
        "c": 0.2,
        "ent_bonus": 0.005,      # Lower entropy to reduce exploration noise
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [128, 64]},  # Larger network for PPO
        "policy_lr": 8e-4,       # More aggressive learning for PPO
        "policy_updates": 5,     # More optimization per epoch
        "policy_kld_limit": 0.1,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [128, 64]},  # Larger network for better estimates
        "value_lr": 8e-4,        # Match policy learning rate
        "value_updates": 10,     # More value training per epoch
        "value_weights": "",


        # Training Parameters - OPTIMIZED FOR PPO
        "episodes": 64,          # Proper batch alignment (64 episodes / 64 batch = 1 batch)
        "epochs": 100,           # Full training schedule
        "max_episode_length": None,
        "batch_size": 64,        # Perfect alignment
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        # Normalization Parameters (optional, defaults to True) CURRENTLY IMPLEMENTED INTERNALLY
        #"normalize_returns": True,  # Normalize value targets to improve value learning
        #"lr_schedule": True,        # Use cosine annealing learning rate scheduling

        # Saving Parameters
        "name": "run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,
    }

    # DCL configuration: Slower training, but uses planning for better action selection
    dcl_config = {
        "algorithm": "dcl",
        "gam": 1,
        "lam": 0.99,
        "eps": 0.15,             # Tighter clipping
        "c": 0.2,
        "ent_bonus": 0.005,      # Lower entropy to reduce exploration noise
        "agent_seed": None,

        # Policy Model - OPTIMIZED FOR DCL
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [64, 32]},
        "policy_lr": 5e-4,       # More aggressive than before
        "policy_updates": 3,     # Increased from 2
        "policy_kld_limit": 0.1,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model - OPTIMIZED FOR DCL
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [64, 32]},
        "value_lr": 5e-4,        # More aggressive learning
        "value_updates": 5,      # Increased from 3
        "value_weights": "",


        # Training Parameters - OPTIMIZED FOR DCL
        "episodes": 32,          # Fewer episodes due to planning overhead
        "epochs": 25,            # Shorter schedule
        "max_episode_length": None,
        "batch_size": 32,
        "sort_states": False,
        "use_gpu": False,
        "load_policy_network": False,
        "verbose": 1,

        # Normalization Parameters (optional, defaults to True)
        "normalize_returns": True,  # Normalize value targets to improve value learning
        "lr_schedule": True,        # Use cosine annealing learning rate scheduling

        # Saving Parameters
        "name": "run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,
    }

    # Select configuration based on algorithm choice
    default_args = ppo_config if ALGORITHM == "ppo-clip" else dcl_config
    # Update algorithm field to match selection
    default_args["algorithm"] = ALGORITHM


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
                if resource['code_employee'] == 0:
                    return {k: binding}

        #if no optimal binding is found, postpone
        return 'postpone'


    if train:
        #training functions
        logger.debug("Starting training run with causal RL enabled")
        agency.training_run(length=10, args_dict=default_args)
        logger.debug("Training run completed")

    else:
        logger.debug("Starting evaluation runs")
        random_average = 0
        random_std = 0
        random_reward = []
        if visualize_random:
            logger.debug("Visualizing random solver")
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.length = 10 #TODO: parameterize in a better way
            frozen_agency.set_solver(RandomSolver())
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            logger.debug("Evaluating random solver (10 runs)")
            for i in range(10):
                frozen_agency = copy.deepcopy(agency)
                res = frozen_agency.testing_run(length=10, solver=RandomSolver())
                random_reward.append(res)
                random_average += res
                random_std += res ** 2

            random_average /= 10
            random_std = (random_std / 10 - random_average ** 2) ** 0.5
            logger.info(f"Random solver: μ={random_average:.3f}, σ={random_std:.3f}")
            print(f"Random solver average reward: {random_average}, std: {random_std}")

        heuristic_average = 0
        heuristic_std = 0
        heuristic_reward = []
        logger.debug("Evaluating heuristic solver (10 runs)")
        for i in range(10):
            frozen_agency = copy.deepcopy(agency)
            solver = HeuristicSolver(perfect_heuristic)
            res = frozen_agency.testing_run(length=10, solver=solver)
            heuristic_reward.append(res)
            heuristic_average += res
            heuristic_std += res ** 2

        heuristic_average /= 10
        heuristic_std = (heuristic_std / 10 - heuristic_average ** 2) ** 0.5
        logger.info(f"Heuristic solver: μ={heuristic_average:.3f}, σ={heuristic_std:.3f}")
        print(f"Heuristic solver average reward: {heuristic_average}, std: {heuristic_std}")



        ppo_average = 0
        ppo_std = 0
        ppo_reward = []

        if visualize_ppo:
            logger.debug("Visualizing DRL solver")
            frozen_agency = copy.deepcopy(agency)
            frozen_agency.set_solver(GymSolver(weights_path=weights_path, metadata=agency.make_metadata()))
            frozen_agency.length = 10
            visual = Visualisation(frozen_agency)
            visual.show()
        else:
            logger.debug("Evaluating DRL solver (10 runs)")
            for i in range(10):
                frozen_agency = copy.deepcopy(agency)
                solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
                res = agency.testing_run(length=10, solver=solver)
                ppo_reward.append(res)
                ppo_average += res
                ppo_std += res ** 2

            ppo_average /= 10
            ppo_std = (ppo_std / 10 - ppo_average ** 2) ** 0.5
            logger.info(f"DRL solver: μ={ppo_average:.3f}, σ={ppo_std:.3f}")
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
