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

if __name__ == "__main__":

    train=True #set to False to test a trained model
    test_model_num = 40 #the number of the model to test (only if train=False)

    # Instantiate a simulation problem.
    agency = GymProblem()

    # Define cases.
    arrival = agency.add_var("arrival", var_attributes=['task_type', 'id']) #TODO: maybe better to define task
    in_transit = agency.add_var("in_transit", var_attributes=['task_type', 'id'])
    waiting = agency.add_var("waiting", var_attributes=['task_type', 'id'])
    busy = agency.add_var("busy", var_attributes=['task_type', 'id', 'code_employee'])
    arrival.put({'task_type': 0, 'id': 0}) #TODO: add a check in put to see if the the keys of the tokens put into places correspond with its var_attributes
    arrival.put({'task_type': 1, 'id': 0})

    # Define resources.
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})
    employee.put({'code_employee': 1})
    employee.put({'code_employee': 2})

    # Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    agency.set_unobservable(simvars=['arrival', 'in_transit'], token_attrs={'waiting': ['id']})

    # Define events.
    def arrive(a):
        a['id'] += 1
        return [SimToken(a, delay=1), SimToken(a)]
    agency.add_event([arrival], [arrival, in_transit], arrive)
    agency.add_event([in_transit], [waiting], lambda x: [SimToken(x, delay=0)], name="dispatch")

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


    agency.add_action([waiting, employee], [busy], behavior=start, name="start")

    def complete(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[-1])]

    agency.add_event([busy], [employee], complete, name='done', reward_function=lambda x: 1)

    if train:
        #training functions
        agency.training_run(length=10)
        #training functions
        agency.training_run(length=10)
    else:
        frozen_agency = copy.deepcopy(agency)
        weights_path = os.path.join(os.getcwd(), "data", "train", "run", f"network-{test_model_num}.pth")
        solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
        agency.testing_run(length=30, solver=solver)
        random_solver = RandomSolver()
        frozen_agency.testing_run(length=30, solver=random_solver)
