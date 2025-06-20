"""
Tests for the gym simulator.
"""
import copy

from simpn.simulator import SimToken

from gympn.simulator import GymProblem
from simpn.reporters import SimpleReporter
from gympn.solvers import HeuristicSolver, GymSolver, RandomSolver
from gympn.utils import sim_tokens_values_from_bindings, binding_from_tokens_values
from gympn.visualisation import Visualisation

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

    def assign_e1_if_available(pn, actions_dict):
        # here pn is not used, but it may be useful in more complex scenarios
        #if there is a combination in get_out containing a token whose code_employee is 1, fire it
        if 'get_out' in actions_dict.keys():
            assignable = actions_dict['get_out']
            for i in assignable:
                if i[0][1].value['code_employee'] == 1:
                    return {'get_out': i}
        if 'start' in actions_dict.keys():
            assignable = actions_dict['start']
            for i in assignable:
                if i[0][0]._id == 'employee':
                   if i[0][1].value['code_employee'] == 2:
                        return {'start': i}
                else:
                    if i[1][1].value['code_employee'] == 2:
                        return {'start': i}
        #otherwise, return a random assignment
        return {list(actions_dict.keys())[0]: list(actions_dict.values())[0][0]}



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
    #for i in range(10):
    #    agency = copy.deepcopy(frozen_pn)
    #    w_p = 'C:/Users/20215143/tue_repos/gympn/tests/data/train/run/network-80.pth' #32 is quite good
    #    solver = GymSolver(weights_path=w_p, metadata=agency.make_metadata())
    #    res=agency.testing_run(solver)
    #    tot+=res

    tot_r = 0
    #for i in range(10):
    #    agency = copy.deepcopy(frozen_pn)
    #    solver = RandomSolver()
    #    res = agency.testing_run(solver)
    #    tot_r += res

    tot_h = 0
    for i in range(10):
        agency = copy.deepcopy(frozen_pn)
        solver = HeuristicSolver(assign_e1_if_available)
        agency.set_solver(solver)
        #visual = Visualisation(agency)
        #visual.show()
        res = agency.testing_run(solver, length=10, reporter=SimpleReporter(), visualize=True)

    print(f'Average PPO policy: {tot/10}')
    print(f'Average random policy: {tot_r/10}')
    print(f'Average heuristic policy: {tot_h/10}')

if __name__ == "__main__":
    test_training()
