"""
Tests for the gym simulator.
"""
import copy

from simpn.simulator import SimToken

from gympn.simulator import GymProblem
from simpn.reporters import SimpleReporter
from gympn.solvers import HeuristicSolver, GymSolver, RandomSolver
from gympn.utils import sim_tokens_values_from_bindings, binding_from_tokens_values

def test_training():
    """
    Test a training in the Gym Simulator.
    """

    train = False #if True, train a model, else test the trained model

    # Instantiate a simulation problem.
    agency = GymProblem()

    # Define queues and other 'places' in the process.
    arrival = agency.add_var("arrival", var_attributes=['id'])
    in_transit = agency.add_var("in_transit", var_attributes=[
        'id'])  # orders stay are dispatched and their existence can be observed (for now)

    waiting = agency.add_var("waiting", var_attributes=['id'])

    busy = agency.add_var("busy", var_attributes=['id', 'code_employee'])

    arrival.put({'id': 0})

    # Define resources.
    employee = agency.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 1})
    employee.put({'code_employee': 2})

    # Define which variables are observable (by default, every SimVar is observable, only unobservable if specified)
    agency.set_unobservable(['in_transit'])

    # Define events.

    def start(c, r):
        return [SimToken((c, r), delay=1)]#delay=uniform(1, 2))]

    def assign_e1_if_available(observable_net, token_combinations):
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

    #one action sends an employee out (which is desired for employee 1)
    unavailable = agency.add_var("unavailable", var_attributes=['code_employee'])

    def get_out(x):
        return [SimToken(x, delay=1)]

    agency.add_action([employee], [unavailable], behavior=get_out, name="get_out", solver=None)

    agency.add_event([unavailable], [employee], behavior=lambda x: [SimToken(x)], name="return")

    def complete(b):
        return [SimToken(b[1])]

    def get_reward(b):
        if b[1]['code_employee'] == 1:
            return -1
        else:
            return 10

    agency.add_event([busy], [employee], complete, name='complete', reward_function=get_reward)

    def arrive(a):
        return [SimToken({'id': a['id'] + 1}, delay=1), SimToken({'id': a['id']})]

    agency.add_event([arrival], [arrival, in_transit], arrive)

    agency.add_event([in_transit], [waiting], lambda x: [SimToken(x, delay=1)], name="dispatch")

    # Run the simulation.
    #agency.simulate(60, SimpleReporter())

    #train_aepn(agency)

    frozen_pn = copy.deepcopy(agency)

    tot = 0
    for i in range(10):
        agency = copy.deepcopy(frozen_pn)
        w_p = 'C:/Users/20215143/tue_repos/gympn/examples/data/train/2025-04-14-18-05-45_run/network-20.pth' #32 is quite good
        solver = GymSolver(weights_path=w_p, metadata=agency.make_metadata())
        res=agency.testing_run(solver)
        tot+=res
#
    #tot_r = 0
    #for i in range(10):
    #    agency = copy.deepcopy(frozen_pn)
    #    solver = RandomSolver()
    #    res = agency.testing_run(solver)
    #    tot_r += res

    tot_h = 0
    solver = HeuristicSolver(assign_e1_if_available)
    for i in range(10):
        agency = copy.deepcopy(frozen_pn)
        res = agency.testing_run(solver)
        tot_h += res

    print(f'Average PPO policy: {tot/10}')
    #print(f'Average random policy: {tot_r/10}')
    print(f'Average heuristic policy: {tot_h/10}')

if __name__ == "__main__":
    test_training()
