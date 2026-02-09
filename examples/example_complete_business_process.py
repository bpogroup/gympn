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
    run_name = '2026-02-02-18-31-18_run'#'2025-10-02-11-14-25_run' #specify the run name to load the weights from
    num_experiments = 5 #number of test experiments to run (if train=False)
    visualize_random = False  # Set to True to visualize the random solver
    visualize_ppo = True  # Set to True to visualize the PPO solver

    weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth") #customize if needed

    ###########################################################################

    # Instantiate a simulation problem.
    agency = GymProblem(causal_rl=train, allow_postpone=True)

    # Define cases with two task types to create meaningful routing decisions
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting = agency.add_var("waiting", var_attributes=['task_type'])
    busy_register = agency.add_var("busy_register", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 1})  # Two different task types

    # Processing stage - CRITICAL CHOICE POINT
    # Task type 0: Simple (should go to junior)
    # Task type 1: Complex (should go to senior)
    waiting_process = agency.add_var("waiting_process", var_attributes=['task_type'])

    # Junior path: fast but poor quality for complex tasks
    busy_junior_simple = agency.add_var("busy_junior_simple", var_attributes=['task_type', 'resource_id'])
    busy_junior_complex = agency.add_var("busy_junior_complex", var_attributes=['task_type', 'resource_id'])

    # Senior path: slower but high quality for complex tasks
    busy_senior_simple = agency.add_var("busy_senior_simple", var_attributes=['task_type', 'resource_id'])
    busy_senior_complex = agency.add_var("busy_senior_complex", var_attributes=['task_type', 'resource_id'])

    # Quality check stages (only for complex tasks)
    quality_check = agency.add_var("quality_check", var_attributes=['task_type', 'resource_id', 'quality_score'])
    rework_stage = agency.add_var("rework_stage", var_attributes=['task_type'])

    # Completed tasks
    completed = agency.add_var("completed", var_attributes=['task_type'])

    # Define resources.
    junior_employee = agency.add_var("junior_employee", var_attributes=['code_employee'])
    junior_employee.put({'code_employee': 0})
    junior_employee.put({'code_employee': 1})

    senior_employee = agency.add_var("senior_employee", var_attributes=['code_employee'])
    senior_employee.put({'code_employee': 0})


    # Define events.
    def arrive(a):
        return [SimToken(a, delay=1/15), SimToken(a)]

    agency.add_event([arrival], [arrival, waiting], arrive)

    # Registration event - all tasks must be registered
    def register(c, r):
        return [SimToken((c, r), delay=1 / 20)]

    agency.add_event([waiting, junior_employee], [busy_register], behavior=register, name="register")

    def complete_register(b):
        return [SimToken(b[1]), SimToken(b[0])]

    agency.add_event([busy_register], [junior_employee, waiting_process], complete_register, name='complete_register')

    # ===== CRITICAL CHOICE POINT: Route to junior or senior =====
    # Simple task (type 0) to junior: 0.2 delay (fast, good)
    # Simple task (type 0) to senior: 0.5 delay (slow, wasteful)
    # Complex task (type 1) to junior: 0.8 delay (fast, poor quality!)
    # Complex task (type 1) to senior: 0.4 delay (slower, good quality)

    def route_to_junior_simple(c, r):
        return [SimToken((c, r), delay=0.2)]

    def route_to_senior_simple(c, r):
        return [SimToken((c, r), delay=0.5)]

    def route_to_junior_complex(c, r):
        return [SimToken((c, r), delay=0.8)]

    def route_to_senior_complex(c, r):
        return [SimToken((c, r), delay=0.4)]

    agency.add_action([waiting_process, junior_employee], [busy_junior_simple],
                      behavior=route_to_junior_simple, name="route_junior_simple")
    agency.add_action([waiting_process, senior_employee], [busy_senior_simple],
                      behavior=route_to_senior_simple, name="route_senior_simple")
    agency.add_action([waiting_process, junior_employee], [busy_junior_complex],
                      behavior=route_to_junior_complex, name="route_junior_complex")
    agency.add_action([waiting_process, senior_employee], [busy_senior_complex],
                      behavior=route_to_senior_complex, name="route_senior_complex")

    # Completion events for simple tasks
    def complete_simple_junior(task_employee_tuple):
        # task_employee_tuple is (task, employee)
        task, employee = task_employee_tuple
        return [SimToken(employee), SimToken(task)]

    def complete_simple_senior(task_employee_tuple):
        # task_employee_tuple is (task, employee)
        task, employee = task_employee_tuple
        return [SimToken(employee), SimToken(task)]

    agency.add_event([busy_junior_simple], [junior_employee, completed], complete_simple_junior,
                     name='complete_junior_simple', reward_function=lambda x: 1)
    agency.add_event([busy_senior_simple], [senior_employee, completed], complete_simple_senior,
                     name='complete_senior_simple', reward_function=lambda x: 1)

    # Complex tasks must pass quality check
    def check_quality_junior(task_employee_tuple):
        # task_employee_tuple is the value of the token from busy_junior_complex
        # which is a tuple (task, employee) created by route_to_junior_complex
        task, employee = task_employee_tuple
        prob = random.uniform(0, 1)
        if prob > 0.3:  # 70% failure rate for junior on complex tasks
            quality_result = {'quality': 0, 'task_type': task.get('task_type', 1)}
            return [SimToken(quality_result, delay=0)]  # Failed quality check
        else:
            quality_result = {'quality': 1, 'task_type': task.get('task_type', 1)}
            return [SimToken(quality_result, delay=0)]  # Passed quality check

    def check_quality_senior(task_employee_tuple):
        # task_employee_tuple is the value of the token from busy_senior_complex
        # which is a tuple (task, employee) created by route_to_senior_complex
        task, employee = task_employee_tuple
        prob = random.uniform(0, 1)
        if prob > 0.1:  # 10% failure rate for senior on complex tasks
            quality_result = {'quality': 1, 'task_type': task.get('task_type', 1)}
            return [SimToken(quality_result, delay=0)]  # Passed quality check
        else:
            quality_result = {'quality': 0, 'task_type': task.get('task_type', 1)}
            return [SimToken(quality_result, delay=0)]  # Failed quality check

    agency.add_event([busy_junior_complex], [quality_check], check_quality_junior,
                     name='check_junior_complex')
    agency.add_event([busy_senior_complex], [quality_check], check_quality_senior,
                     name='check_senior_complex')

    # Handle quality check results
    def handle_quality_pass(q):
        # q is a dictionary: {'quality': quality_score, 'task_type': task_type}
        # Return completed task
        return [SimToken({'task_type': q.get('task_type', 0)})]

    def handle_quality_fail(q):
        # q is a dictionary: {'quality': quality_score, 'task_type': task_type}
        # Return task for rework
        return [SimToken({'task_type': q.get('task_type', 0)})]

    # Quality pass event: reward given when quality=1
    agency.add_event([quality_check], [completed],
                     behavior=handle_quality_pass,
                     name='quality_pass',
                     guard=lambda q: isinstance(q, dict) and q.get('quality', 0) == 1,
                     reward_function=lambda q: 1)

    # Quality fail event: task goes to rework when quality=0
    agency.add_event([quality_check], [rework_stage],
                     behavior=handle_quality_fail,
                     name='quality_fail',
                     guard=lambda q: isinstance(q, dict) and q.get('quality', 0) == 0,
                     reward_function=lambda q: -1)

    # Rework event - takes task from rework_stage and puts it back for reprocessing
    def rework(task):
        # task comes from rework_stage
        return [SimToken(task)]

    agency.add_event([rework_stage], [waiting_process], rework, name='rework_requeue')



    ###########################################################################

    # Heuristic function for intelligent task routing
    def task_routing_heuristic(pn, actions_dict, bindings=None):
        """
        Intelligent heuristic that routes tasks based on type:
        - Simple tasks (type 0) → Junior employees (fast)
        - Complex tasks (type 1) → Senior employees (better quality)

        This creates a clear advantage over random routing because:
        - Routing simple to junior: 0.2 delay (good)
        - Routing simple to senior: 0.5 delay (waste of senior time)
        - Routing complex to junior: 0.8 delay + 70% rework (very bad)
        - Routing complex to senior: 0.4 delay + 10% rework (good)
        """

        waiting_tokens = HeuristicSolver.get_place_tokens('waiting_process', pn)
        quality_check_tokens = HeuristicSolver.get_place_tokens('quality_check', pn)
        rework_tokens = HeuristicSolver.get_place_tokens('rework_stage', pn)

        junior_available = len(HeuristicSolver.get_place_tokens('junior_employee', pn))
        senior_available = len(HeuristicSolver.get_place_tokens('senior_employee', pn))

        # PRIORITY 1: Handle rework (failed complex tasks) - MUST use senior
        if rework_tokens and senior_available > 0:
            if 'rework' in actions_dict and actions_dict['rework']:
                return {'rework': actions_dict['rework'][0]}

        # PRIORITY 2: Handle quality check results
        if quality_check_tokens:
            if 'quality_pass' in actions_dict and actions_dict['quality_pass']:
                return {'quality_pass': actions_dict['quality_pass'][0]}

        # PRIORITY 3: Route waiting tasks based on TYPE
        if waiting_tokens:
            for token in waiting_tokens:
                task_type = token.var_name if hasattr(token, 'var_name') else None
                # Try to get task type from token attributes
                if hasattr(token, 'attributes'):
                    task_type = token.attributes.get('task_type', None)

                # If we can determine it's a simple task, prefer junior
                if task_type == 0:  # Simple task
                    if junior_available > 0 and 'route_junior_simple' in actions_dict and actions_dict['route_junior_simple']:
                        return {'route_junior_simple': actions_dict['route_junior_simple'][0]}
                    elif senior_available > 0 and 'route_senior_simple' in actions_dict and actions_dict['route_senior_simple']:
                        return {'route_senior_simple': actions_dict['route_senior_simple'][0]}

                # If we can determine it's a complex task, prefer senior
                elif task_type == 1:  # Complex task
                    if senior_available > 0 and 'route_senior_complex' in actions_dict and actions_dict['route_senior_complex']:
                        return {'route_senior_complex': actions_dict['route_senior_complex'][0]}
                    elif junior_available > 0 and 'route_junior_complex' in actions_dict and actions_dict['route_junior_complex']:
                        return {'route_junior_complex': actions_dict['route_junior_complex'][0]}

                # If we can't determine type, use availability heuristic
                else:
                    if junior_available > 0 and 'route_junior_simple' in actions_dict and actions_dict['route_junior_simple']:
                        return {'route_junior_simple': actions_dict['route_junior_simple'][0]}
                    elif senior_available > 0 and 'route_senior_simple' in actions_dict and actions_dict['route_senior_simple']:
                        return {'route_senior_simple': actions_dict['route_senior_simple'][0]}

        # PRIORITY 4: Register new work if available
        if 'register' in actions_dict and actions_dict['register']:
            return {'register': actions_dict['register'][0]}

        # FALLBACK: Take any available action
        for action_name, assignments in actions_dict.items():
            if assignments:
                return {action_name: assignments[0]}

        return 'postpone' if pn.allow_postpone else None



    # Default training arguments (change them as needed)
    default_args = {
        # Algorithm Parameters
        "algorithm": "ppo-clip",
        "causal_rl": True,  # Enable causal RL for better credit assignment
        "gam": 0.99,
        "lam": 0.95,
        "eps": 0.15,
        "c": 0.1,
        "ent_bonus": 0.001,
        "agent_seed": None,

        # Policy Model
        "policy_model": "gnn",
        "policy_kwargs": {"hidden_layers": [64, 32]},
        "policy_lr": 5e-4,
        "policy_updates": 4,
        "policy_kld_limit": 0.1,
        "policy_weights": "",
        "policy_network": "",
        "score": False,
        "score_weight": 1e-3,

        # Value Model
        "value_model": "gnn",
        "value_kwargs": {"hidden_layers": [64, 32]},
        "value_lr": 1e-3,
        "value_updates": 3,
        "value_weights": "",

        # Training Parameters
        "episodes": 20,
        "epochs": 100,
        "max_episode_length": None,
        "batch_size": 32,
        "sort_states": False,
        "use_gpu": True,
        "load_policy_network": False,
        "verbose": 1,

        # Saving Parameters
        "name": "run",
        "datetag": True,
        "logdir": "data/train",
        "save_freq": 1,
        "open_tensorboard": False,
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

        # Test heuristic solver
        solver_heuristic = HeuristicSolver(heuristic_function=task_routing_heuristic)
        heuristic_average, heuristic_std = run_experiments(agency, solver_heuristic, num_experiments, length=10)
        print(f"Heuristic solver average reward: {heuristic_average}, std: {heuristic_std}")

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

            print("\n" + "="*70)
            print("STATISTICAL SIGNIFICANCE TESTS")
            print("="*70)

            # DRL vs Random
            z, p_value, significant = check_statistical_significance(ppo_average, ppo_std, num_experiments, random_average, random_std, num_experiments)
            print(f"\nDRL vs Random:")
            print(f"  Z-score: {z:.4f}, P-value: {p_value:.6f}, Significant: {significant}")
            print(f"  Percentage difference: {100 * (ppo_average - random_average) / abs(random_average):+.2f}%")

            # DRL vs Heuristic
            z, p_value, significant = check_statistical_significance(ppo_average, ppo_std, num_experiments, heuristic_average, heuristic_std, num_experiments)
            print(f"\nDRL vs Heuristic:")
            print(f"  Z-score: {z:.4f}, P-value: {p_value:.6f}, Significant: {significant}")
            print(f"  Percentage difference: {100 * (ppo_average - heuristic_average) / abs(heuristic_average):+.2f}%")

            # Heuristic vs Random
            z, p_value, significant = check_statistical_significance(heuristic_average, heuristic_std, num_experiments, random_average, random_std, num_experiments)
            print(f"\nHeuristic vs Random:")
            print(f"  Z-score: {z:.4f}, P-value: {p_value:.6f}, Significant: {significant}")
            print(f"  Percentage difference: {100 * (heuristic_average - random_average) / abs(random_average):+.2f}%")

            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"Random solver:     {random_average:.4f} ± {random_std:.4f}")
            print(f"Heuristic solver:  {heuristic_average:.4f} ± {heuristic_std:.4f}")
            print(f"DRL solver:        {ppo_average:.4f} ± {ppo_std:.4f}")
            print("="*70)
