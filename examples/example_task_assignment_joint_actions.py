"""
This file implements a task assignment problem with two employees and a queue of tasks.
The goal is to assign tasks to employees in a way that minimizes the total time taken to complete all tasks.
The simulation uses a heuristic solver to assign tasks to employees based on their availability.
"""
import copy
import os
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver
from gympn.visualisation import Visualisation

if __name__ == "__main__":

    train=True #set to False to test a trained model
    #test_model_num = 99 #the number of the model to test (only if train=False)
    run_name = "2025-05-12-13-02-47_run"

    # Instantiate a simulation problem.
    agency = GymProblem()

    # Define cases.
    arrival = agency.add_var("arrival", var_attributes=['task_type'])
    waiting1 = agency.add_var("waiting1", var_attributes=['task_type'])
    busy1 = agency.add_var("busy1", var_attributes=['task_type', 'code_employee'])
    waiting2 = agency.add_var("waiting2", var_attributes=['task_type'])
    busy2 = agency.add_var("busy2", var_attributes=['task_type', 'code_employee'])

    arrival.put({'task_type': 0}) #TODO: add a check in put to see if the the keys of the tokens put into places correspond with its var_attributes
    arrival.put({'task_type': 1})

    # Define resources.
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})
    employee.put({'code_employee': 1})
    employee.put({'code_employee': 2})

    # Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    #agency.set_unobservable(simvars=['arrival', 'in_transit'], token_attrs={'waiting': ['id']})

    # Define events.
    def arrive(a):
        if a['task_type'] == 0:
            return [SimToken(a, delay=1), SimToken(a), None]
        elif a['task_type'] == 1:
            return [SimToken(a, delay=1), None, SimToken(a)]

    agency.add_event([arrival], [arrival, waiting1, waiting2], arrive)

    def start(c, r):
        """
        Resource 0 takes 1 time unit to finish tasks of type 0, and 2 time units to finish tasks of type 1.
        Resource 1 takes 2 time units to finish tasks of type 0, and 1 time unit to finish tasks of type 1.
        Resource 2 takes 3 time units to finish any task.
        :param c: the task
        :param r: the resource
        :return: a list of SimTokens representing the task and the resource that were assigned to them
        """
        if c['task_type'] == 0 and r['code_employee'] == 0 or c['task_type'] == 1 and r['code_employee'] == 1:
            return [SimToken((c, r), delay=1)]
        elif c['task_type'] == 0 and r['code_employee'] == 1 or c['task_type'] == 1 and r['code_employee'] == 0:
            return [SimToken((c, r), delay=2)]
        else:
            return [SimToken((c, r), delay=3)]


    agency.add_action([waiting1, employee], [busy1], behavior=start, name="start1")

    def complete1(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[-1])]

    agency.add_event([busy1], [employee], complete1, name='done1')

    agency.add_action([waiting2, employee], [busy2], behavior=start, name="start2")

    def complete2(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[-1])]

    agency.add_event([busy2], [employee], complete2, name='done2', reward_function=lambda x: 1)



    if train:
        #training functions
        agency.training_run(length=10)
        #training functions
        agency.training_run(length=10)
    else:
        solver = RandomSolver()
        agency.set_solver(solver)
        visual = Visualisation(agency)
        visual.show()

        frozen_agency = copy.deepcopy(agency)
        weights_path = os.path.join(os.getcwd(), "data", "train", run_name, f"best_policy.pth")
        solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
        agency.testing_run(length=10, solver=solver)
        random_solver = RandomSolver()
        frozen_agency.testing_run(length=10, solver=random_solver)
