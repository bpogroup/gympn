"""
Deep diagnostic: Trace exact token IDs through the causal chain for postpone actions.
Verify that postpone's output tokens appear in the causal chain of reward transitions.
"""
import uuid
import types
from simpn.simulator import SimToken
from gympn.simulator import GymProblem
from gympn.environment import AEPN_Env
from gympn.causal_traces import CausalTraces


def create_env(causal_rl=True):
    env = GymProblem(allow_postpone=True, causal_rl=causal_rl)
    arrival = env.add_var("arrival", var_attributes=['task_type'])
    waiting = env.add_var("waiting", var_attributes=['task_type'])
    busy = env.add_var("busy", var_attributes=['task_type', 'resource_id'])
    arrival.put({'task_type': 0})
    arrival.put({'task_type': 0})
    employee = env.add_var("employee", var_attributes=['code_employee'])
    employee.put({'code_employee': 0})
    employee.put({'code_employee': 1})

    def arrive(a):
        return [SimToken(a, delay=1), SimToken(a)]
    env.add_event([arrival], [arrival, waiting], arrive)

    def start(c, r):
        if r['code_employee'] == 1:
            return [SimToken((c, r), delay=20)]
        else:
            return [SimToken((c, r), delay=0.5)]
    env.add_action([waiting, employee], [busy], behavior=start, name="start")

    def complete(b):
        return [SimToken(b[1])]

    def r_function(x):
        return 1

    env.add_event([busy], [employee], complete, name='complete', reward_function=r_function)
    return env


def optimal_policy(pn):
    """Only assign to emp0, postpone otherwise."""
    actions = pn.pn_actions
    for i, a in enumerate(actions):
        if isinstance(a[0], list) and a[0] == ['postpone']:
            continue
        binding, time, transition = a
        for place, token in binding:
            if place._id == 'employee' and token.value.get('code_employee') == 0:
                return i, "emp0"
    for i, a in enumerate(actions):
        if isinstance(a[0], list) and a[0] == ['postpone']:
            return i, "postpone"
    return 0, "default"


if __name__ == "__main__":
    pn = create_env(causal_rl=True)
    pn.length = 10

    # Assign initial token IDs
    for place in pn.places:
        for token in place.marking:
            setattr(token, '_id', str(uuid.uuid4()))

    gym_env = AEPN_Env(pn)
    state = gym_env.reset()
    pn = gym_env.pn

    done = False
    step = 0

    while not done and step < 10:  # Only first 10 steps for clarity
        action_idx, desc = optimal_policy(pn)
        
        # Print state BEFORE action
        ct = pn.causal_trace
        n_act = ct.transition_history.get_action_transitions_len()
        
        next_state, reward, terminated, truncated, info = gym_env.step(action_idx)
        pn = gym_env.pn
        step += 1
        
        ct = pn.causal_trace
        n_act_after = ct.transition_history.get_action_transitions_len()
        
        print(f"\nStep {step}: clock={pn.clock:.1f}, action={desc}, "
              f"act_trans: {n_act}→{n_act_after}, pn_reward={pn.reward}", flush=True)
        
        # Print the last registered action transition
        action_trans = ct.transition_history.get_action_transitions()
        if action_trans:
            last_at = action_trans[-1]
            print(f"  Last action trans: time={last_at['time']}, "
                  f"in={[tid[:8] for tid in last_at['input_tokens']]}, "
                  f"out={[tid[:8] for tid in last_at['output_tokens']]}", flush=True)
        
        done = terminated

    # Now analyze the full causal chain
    ct = pn.causal_trace
    print("\n" + "=" * 70)
    print("TOKEN_TO_ACTION MAPPING")
    print("=" * 70)
    
    action_transitions = ct.transition_history.get_action_transitions()
    token_to_action = {}
    for idx, act in enumerate(action_transitions):
        for tid in act.get("output_tokens", []):
            token_to_action[tid] = (idx, act.get("time"))
            print(f"  token {tid[:8]}... → action {idx} at time {act.get('time')}")
    
    print(f"\n  Total tokens in mapping: {len(token_to_action)}")
    
    # Check reward transitions and their causal chains
    print("\n" + "=" * 70)
    print("REWARD TRANSITION ANALYSIS")
    print("=" * 70)
    
    for i, tr in enumerate(ct.transition_history.transitions):
        reward = tr.get('reward', 0.0)
        if reward == 0.0:
            continue
        
        transition_time = tr.get('time')
        input_token_ids = tr.get('input_tokens', [])
        
        print(f"\n  Reward transition at time={transition_time}, reward={reward}")
        print(f"    Input tokens: {[tid[:8] for tid in input_token_ids]}")
        
        # Trace each input token
        for token_id in input_token_ids:
            direct = token_id in token_to_action
            if direct:
                action_idx, action_time = token_to_action[token_id]
                print(f"    Direct match: {token_id[:8]}... → action {action_idx} (time={action_time})")
            
            # Trace causal chain
            chain = ct.token_history.get_causal_chain(token_id)
            print(f"    Causal chain ({len(chain)} layers):")
            for depth, layer in enumerate(chain):
                for tid in layer:
                    in_map = tid in token_to_action
                    marker = " *** ACTION ***" if in_map else ""
                    if in_map:
                        aidx, atime = token_to_action[tid]
                        marker += f" (action {aidx}, time={atime})"
                    token_info = ct.token_history.get_token(tid)
                    event_id = token_info.get('event', '?')[:20] if token_info else '?'
                    print(f"      depth={depth}: {tid[:8]}... event={event_id}{marker}")
    
    # Now compute credits
    credits = ct.redistribute_rewards(gamma=1, scheme="hybrid")
    print("\n" + "=" * 70)
    print("CREDITS")
    print("=" * 70)
    
    for i, (at, cr) in enumerate(zip(action_transitions[:10], credits[:10])):
        is_postpone = "postpone" in str(at.get('transition', ''))
        print(f"  Action {i}: time={at.get('time')}, credit={cr:.4f}"
              f"{'  [POSTPONE]' if is_postpone else ''}")

