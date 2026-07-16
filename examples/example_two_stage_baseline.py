"""
Baseline comparison for the two-stage assignment problem.

Runs RandomSolver and the perfect heuristic on the same GymProblem to estimate
expected returns and provide a target for RL training.
"""
import copy
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.solvers import RandomSolver, HeuristicSolver


def perfect_heuristic(observable_net, tokens_comb):
    fallback = None
    for k, bindings in tokens_comb.items():
        for binding in bindings:
            res = binding[1][1].value
            if 'start_A' in k and res['resource_id'] == 0:
                return {k: binding}
            if 'start_B' in k and res['resource_id'] == 1:
                return {k: binding}
            if fallback is None:
                fallback = {k: binding}
    return fallback


def make_problem():
    agency = GymProblem(allow_postpone=True, causal_rl=False)

    arrival = agency.add_var("arrival", var_attributes=['case_id'])
    waiting_A = agency.add_var("waiting_A", var_attributes=['case_id'])
    busy_A = agency.add_var("busy_A", var_attributes=['case_id', 'resource_id'])
    waiting_B = agency.add_var("waiting_B", var_attributes=['case_id'])
    busy_B = agency.add_var("busy_B", var_attributes=['case_id', 'resource_id'])

    arrival.put({'case_id': 0})
    arrival.put({'case_id': 1})

    resource = agency.add_var("resource", var_attributes=['resource_id'])
    resource.put({'resource_id': 0})
    resource.put({'resource_id': 1})

    def arrive(a):
        next_case = {'case_id': a['case_id'] + 2}
        return [SimToken(next_case, delay=3), SimToken(a)]
    agency.add_event([arrival], [arrival, waiting_A], arrive)

    def start_A(c, r):
        delay = 1 if r['resource_id'] == 0 else 5
        return [SimToken((c, r), delay=delay)]
    agency.add_action([waiting_A, resource], [busy_A], behavior=start_A, name="start_A")

    def complete_A(b):
        case, res = b
        return [SimToken(res), SimToken(case)]
    agency.add_event([busy_A], [resource, waiting_B], complete_A, name='complete_A')

    def start_B(c, r):
        delay = 1 if r['resource_id'] == 1 else 5
        return [SimToken((c, r), delay=delay)]
    agency.add_action([waiting_B, resource], [busy_B], behavior=start_B, name="start_B")

    def complete_B(b):
        _, res = b
        return [SimToken(res)]
    agency.add_event([busy_B], [resource], complete_B, name='complete_B', reward_function=lambda x: 1)

    return agency


if __name__ == '__main__':
    n_runs = 50
    length = 20

    random_rewards = []
    heuristic_rewards = []

    for _ in range(n_runs):
        ag = make_problem()
        random_rewards.append(ag.testing_run(length=length, solver=RandomSolver()))

    for _ in range(n_runs):
        ag = make_problem()
        heuristic_rewards.append(ag.testing_run(length=length, solver=HeuristicSolver(perfect_heuristic)))

    import math
    def stats(lst):
        avg = sum(lst) / len(lst)
        std = math.sqrt(sum((x - avg) ** 2 for x in lst) / len(lst))
        return avg, std

    r_avg, r_std = stats(random_rewards)
    h_avg, h_std = stats(heuristic_rewards)

    print(f"Random solver — avg reward: {r_avg:.2f}, std: {r_std:.2f}")
    print(f"Heuristic solver — avg reward: {h_avg:.2f}, std: {h_std:.2f}")

