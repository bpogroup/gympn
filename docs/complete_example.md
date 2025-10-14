# A Complete Business Process Optimization Problem

## Introduction

This tutorial provides a comprehensive overview of how to model and solve a business process optimization problem using Action-Evolution Petri Nets (A-E PNs) within the `gympn` library.

## Motivation

In the previous examples, we explored basic concepts of A-E PNs. Now, we will tackle a more complex scenario that involves multiple resources, tasks, and constraints to demonstrate the practical application of these concepts in a business context.
In particular, we will model a simplified loan application at a bank. We will include stochastic arrivals, but we will keep the processing times deterministic for ease of comparison between policies.

## Problem Description

A bank processes loan applications through a series of tasks, each requiring specific resources. Resources are grouped in two pools: junior employees and senior employees. The bank has two employees in each pool.
Junior employees are responsible for registering new applications and drafting loan proposals.
When a new application is registered, it can be assigned to either a junior or a senior employee for providing advice to the customer regarding the best product for his case.
If an application is assigned to a junior employee, a simple financial product is proposed. Simple products are handled faster, but they have a higher risk of needing rework. 
Vice versa, if a case is assigned to a senior employee, a complex financial product is proposed. Complex products take longer to process, but they have a lower risk of needing rework.
The goal is to optimize the processing of these applications to minimize the total time taken from application submission to final decision.

We propose a BPMN model of the process, shown in the figure below.

![BPMN Model](images/bpmn_gympn_complete_example.png)

Below, we include the A-E PN model of the process, including decision points and the problem objective in the form of a reward function. Guard functions are omitted when not necessary for clarity. Similarly, transitions whose firing generates a reward of zero are not annotated. The token's colours are also omitted since they are not relevant for this example.

![AEPN Model](images/aepn_gympn_complete_example.png)

## GymPN Implementation

We will implement the above model using the `gympn` library. The implementation includes defining places, transitions, resources, and the reward function.

```python
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
import random

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

agency.training_run(length=10)
```

## Policy Evaluation

In this case, since the environment includes stochastic elements (e.g., arrival times and rework probabilities), we can evaluate different policies by running multiple simulations and comparing their performance based on the total time taken to process applications.
In the following code, we run the simulation multiple times and calculate the average reward (which corresponds to the number of completed applications) for the DRL policy and the random policy.
Then, we proceed to evaluate the performance of different policies by running multiple simulations and comparing their average rewards. We also propose a statistical test to determine if the differences in performance are significant.

```python
import copy
import numpy as np
from gympn.solvers import RandomSolver, GymSolver
from math import sqrt
from scipy.stats import norm

num_experiments = 1000

def run_experiments(problem, solver, num_experiments, reporter=None, length=None):
    rewards = []
    for i in range(num_experiments):
        # Create a fresh copy of the problem
        problem_copy = copy.deepcopy(problem)

        # Run the experiment
        reward = problem_copy.testing_run(solver, reporter=reporter, length=length)
        rewards.append(reward)
    return np.mean(rewards), np.std(rewards)


def check_statistical_significance(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    # Calculate the z-score
    z = (mean1 - mean2) / sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

    # Calculate the p-value (two-tailed test)
    p_value = 2 * norm.sf(abs(z))

    # Check significance
    significant = p_value < alpha

    return z, p_value, significant


drl_solver = GymSolver(weights_path='path_to_weights.pt', metadata=agency.make_metadata())
ppo_average, ppo_std = run_experiments(agency, drl_solver, num_experiments, length=10)


random_solver = RandomSolver()
random_average, random_std = run_experiments(agency, random_solver, num_experiments, length=10)

z, p_value, significant = check_statistical_significance(
    ppo_average, ppo_std, num_experiments,
    random_average, random_std, num_experiments
)
print(f"PPO Average Reward: {ppo_average}, Std: {ppo_std}")
print(f"Random Average Reward: {random_average}, Std: {random_std}")
print(f"Z-score: {z}, P-value: {p_value}, Significant: {significant}")
```