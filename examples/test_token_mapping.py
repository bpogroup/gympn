"""
Quick smoke test for token-wise redistribution and TrajectoryBuffer mapping.
"""
from types import SimpleNamespace
from gympn.causal_traces import CausalTraces
from gympn.data import TrajectoryBuffer

# Build causal traces
ct = CausalTraces()
# Create an action transition and a token produced by it
act = SimpleNamespace(_id='act_1')
tok1 = SimpleNamespace(_id='tok1')
# Register transition (action)
ct.transition_history.add_transition(act, input_tokens=[], output_tokens=[tok1], is_action=True, created_by=None, reward=0.0, time=1)
# Register token with creating transition
ct.token_history.add_token(tok1, transition=act, parent_tokens=[], created_by=0, time=1)
# Create an event transition that consumes tok1 and produces reward
ev = SimpleNamespace(_id='ev_1')
ct.transition_history.add_transition(ev, input_tokens=[tok1], output_tokens=[], is_action=False, created_by=None, reward=1.0, time=2)

# Now run redistribution
act_array, token_map = ct.redistribute_rewards_tokenwise(gamma=0.99, scheme='flow')
print('action_array:', act_array)
print('token_map:', token_map)

# Now test TrajectoryBuffer mapping
buf = TrajectoryBuffer(gam=0.99, lam=0.95)
# create dummy state
state = {'graph': {}}
# store one step with token_ids referencing 'tok1'
buf.store(state=state, action=0, action_index=0, reward=0.0, logprob=0.0, value=0.0, logpis=None, token_ids=[tok1._id])
# prepare to finish using causal traces
buf.finish(credits=ct, mode='replace')
print('returns_:', buf.returns_)
print('advantages_:', buf.advantages_)
print('credits_vec sum (should equal reward 1.0):', buf.returns_.sum().item())




