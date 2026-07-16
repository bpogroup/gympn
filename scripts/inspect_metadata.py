import os
import pprint
from gympn.simulator import GymProblem


def build_two_stage_problem():
    agency = GymProblem(allow_postpone=True, causal_rl=False)
    arrival = agency.add_var("arrival", var_attributes=['case_id'])
    waiting_A = agency.add_var("waiting_A", var_attributes=['case_id'])
    busy_A = agency.add_var("busy_A", var_attributes=['case_id', 'resource_id'])
    waiting_B = agency.add_var("waiting_B", var_attributes=['case_id'])
    busy_B = agency.add_var("busy_B", var_attributes=['case_id', 'resource_id'])

    arrival.put({'case_id': 0})
    arrival.put({'case_id': 0})

    resource = agency.add_var("resource", var_attributes=['resource_id'])
    resource.put({'resource_id': 0})
    resource.put({'resource_id': 1})

    def arrive(a):
        next_case = {'case_id': a['case_id']}
        from simpn.simulator import SimToken
        return [SimToken(next_case, delay=1), SimToken(a)]

    agency.add_event([arrival], [arrival, waiting_A], arrive)

    def start_A(c, r):
        from simpn.simulator import SimToken
        delay = 1 if r['resource_id'] == 0 else 2
        return [SimToken((c, r), delay=delay)]

    agency.add_action([waiting_A, resource], [busy_A], behavior=start_A, name="start_A")

    def complete_A(b):
        case, res = b
        from simpn.simulator import SimToken
        return [SimToken(res), SimToken(case)]

    agency.add_event([busy_A], [resource, waiting_B], complete_A, name='complete_A')

    def start_B(c, r):
        from simpn.simulator import SimToken
        delay = 1 if r['resource_id'] == 1 else 2
        return [SimToken((c, r), delay=delay)]

    agency.add_action([waiting_B, resource], [busy_B], behavior=start_B, name="start_B")

    def complete_B(b):
        _, res = b
        from simpn.simulator import SimToken
        return [SimToken(res)]

    agency.add_event([busy_B], [resource], complete_B, name='complete_B',
                     reward_function=lambda x: 1)

    return agency


if __name__ == '__main__':
    agency = build_two_stage_problem()
    cases = [
        {'add_self_loops': True, 'add_action_to_action': True, 'add_reverse_edges': False, 'add_all_to_action': False, 'add_reverse_for_extra': False},
        {'add_self_loops': True, 'add_action_to_action': True, 'add_reverse_edges': True, 'add_all_to_action': False, 'add_reverse_for_extra': False},
        {'add_self_loops': True, 'add_action_to_action': True, 'add_reverse_edges': True, 'add_all_to_action': True, 'add_reverse_for_extra': True},
    ]

    for cfg in cases:
        print('\n=== metadata with flags:')
        pprint.pprint(cfg)
        meta = agency.make_metadata(**cfg)
        print('\nNode types:')
        pprint.pprint(meta[0])
        print('\nEdge types:')
        pprint.pprint(meta[1])

