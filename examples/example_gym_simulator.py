"""
Tests for the gym simulator.
"""
from simpn.simulator import SimToken

from gympn.simulator import GymProblem
from random import uniform
from random import expovariate as exp
from simpn.reporters import SimpleReporter
from gympn.solvers import HeuristicSolver
from gympn.utils import sim_tokens_values_from_bindings, binding_from_tokens_values

def test_gym_simulator():
    """
    Test the gym simulator.
    """
    # Instantiate a simulation problem.
    agency = GymProblem()

    # Define queues and other 'places' in the process.
    arrival = agency.add_var("arrival")
    waiting = agency.add_var("waiting")
    busy = agency.add_var("busy")

    arrival.put(1)

    # Define resources.
    employee = agency.add_var("employee")
    employee.put("e1")
    employee.put("e2")

    # Define events.

    def start(c, r):
        return [SimToken((c, r), delay=uniform(10, 15))]

    def assign_e1_if_available(token_combinations, problem):
        # here problem is not used, but it may be useful in more complex scenarios

        # sim_tokens_values_from_bindings returns a list of token values (i.e., strings) from a list of bindings
        assignable = sim_tokens_values_from_bindings(token_combinations)

        assignable = [el for el in assignable if el[0] == "e1" or el[1] == "e1"]
        if len(assignable):
            return binding_from_tokens_values(assignable[0], token_combinations)
        else:
            print("Resource e1 not available, returning a random assignment")
            return token_combinations[0]

    agency.add_action([waiting, employee], [busy], behavior=start, solver=HeuristicSolver(assign_e1_if_available))

    def complete(b):
        return [SimToken(b[1])]

    agency.add_event([busy], [employee], complete)

    def arrive(a):
        return [SimToken(a + 1, delay=exp(4) * 60), SimToken('c' + str(a))]


    agency.add_event([arrival], [arrival, waiting], arrive, reward_function=lambda x :  1)

    # Run the simulation.
    agency.simulate(60, SimpleReporter())

if __name__ == "__main__":
    test_gym_simulator()