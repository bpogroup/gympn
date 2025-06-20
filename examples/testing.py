"""
This file implements a task assignment problem with two employees and a queue of tasks.
The goal is to assign tasks to employees in a way that minimizes the total time taken to complete all tasks.
The simulation uses a heuristic solver to assign tasks to employees based on their availability.
"""
import copy
import os
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import GymSolver, RandomSolver, HeuristicSolver
from gympn.visualisation import Visualisation

if __name__ == "__main__":

    train=False #set to False to test a trained model
    test_model_num = 49 #the number of the model to test (only if train=False)
    folder_name = '2025-05-01-15-45-48_run'

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
    employee1 = agency.add_var("employee1", var_attributes=['code_employee'])
    employee1.put({'code_employee': 0})
    employee1.put({'code_employee': 1})
    #employee1.put({'code_employee': 2})

    employee2 = agency.add_var("employee2", var_attributes=['code_employee'])
    employee2.put({'code_employee': 0})
    employee2.put({'code_employee': 1})
    #employee2.put({'code_employee': 2})

    # Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    #agency.set_unobservable(simvars=['arrival', 'in_transit'], token_attrs={'waiting': ['id']})

    # Define events.
    def arrive(a):
        return [SimToken(a, delay=1), SimToken(a)]

    agency.add_event([arrival], [arrival, waiting1], arrive)

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


    agency.add_action([waiting1, employee1], [busy1], behavior=start, name="start1")

    def complete1(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[-1]), SimToken(b[0])]

    agency.add_event([busy1], [employee1, waiting2], complete1, name='done1')

    agency.add_action([waiting2, employee2], [busy2], behavior=start, name="start2")

    def complete2(b):
        """
        This function is called when a task is completed.
        It returns a list of SimTokens representing the task that was completed.
        :param b: the tuple (task, resource)
        :return: a list of SimTokens representing the resource that has completed a task
        """
        return [SimToken(b[-1])]

    agency.add_event([busy2], [employee2], complete2, name='done2', reward_function=lambda x: 1)


    #define perfect heuristic
    def perfect_heuristic(observable_net, tokens_comb):
        """
        This heuristic function selects the best binding based on the task type and employee code.
        :param observable_net: the observable net.
        :param tokens_comb: the list of all possible bindings.
        :return: the best binding.
        """
        #The perfect heuristic is to always assign the task to the employee that can do it the fastest (so taks type 0 to resource 0 and task type 1 to resource 1)
        for k, el in tokens_comb.items():
            for binding in el:
                task = binding[0][1].value
                resource = binding[1][1].value
                if task['task_type'] == 0 and resource['code_employee'] == 0 or task['task_type'] == 1 and resource['code_employee'] == 1:
                    return {k : binding}


    if train:
        #training functions
        agency.training_run(length=10)
        #training functions
        agency.training_run(length=10)
    else:
        #solver = RandomSolver()
        #agency.set_solver(solver)
        #visual = Visualisation(agency)
        #visual.show()

        frozen_agency = copy.deepcopy(agency)
        weights_path = os.path.join(os.getcwd(), "data", "train", folder_name, f"network-{test_model_num}.pth")
        solver = GymSolver(weights_path=weights_path, metadata=agency.make_metadata())
        agency.testing_run(length=10, solver=solver)
        #random_solver = RandomSolver()
        #frozen_agency2 = copy.deepcopy(frozen_agency)
        #frozen_agency.testing_run(length=10, solver=random_solver)

        #perfect_heuristic_solver = HeuristicSolver(perfect_heuristic)
        #frozen_agency2.testing_run(length=10, solver=perfect_heuristic_solver)
