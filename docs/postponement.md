# Postponement and Net-Factored Credit

Postponement lets an agent defer a decision instead of firing an enabled action: for example, keeping a slow resource idle rather than giving it a job a faster resource will soon handle. Net-factored credit (NF-GAE) lets an agent learn efficiently when the net contains several independent parts.

## The Postponement Feature

### How Postponement Works

Postponement is enabled at environment creation:

```python
agency = GymProblem(allow_postpone=True)
```

When enabled, the agent receives an additional action: **postpone**. At any action decision point, the agent can choose to:
- Fire one of the available action transitions with a specific binding
- **Postpone** the current decision

When postpone is selected:
1. The network tag switches to **E** (evolution) without advancing the clock
2. The environment evolves non-deterministically
3. When the next action decision point is reached, the agent makes a new decision
4. This may occur immediately or after evolution transitions fire

### Implementation Example

Let's extend our task assignment example with postponement:

```python
from gympn.simulator import GymProblem
from simpn.simulator import SimToken

# Enable postponement
agency = GymProblem(allow_postpone=True)

# Define places and transitions...
arrival = agency.add_var("arrival", var_attributes=['task_type'])
waiting = agency.add_var("waiting", var_attributes=['task_type'])
busy = agency.add_var("busy", var_attributes=['task_type', 'resource_id'])
employee = agency.add_var("employee", var_attributes=['code_employee'])

# Define transitions...
def arrive(a):
    return [SimToken(a, delay=1), SimToken(a)]

agency.add_event([arrival], [arrival, waiting], arrive)

def start(c, r):
    if r['code_employee'] == 0:
        return [SimToken((c, r), delay=0.5)]  # Fast processor
    else:
        return [SimToken((c, r), delay=5.0)]  # Slow processor

agency.add_action([waiting, employee], [busy], behavior=start, name="start")

def complete(b):
    return [SimToken(b[1])]

agency.add_event([busy], [employee], complete, name='complete', 
                 reward_function=lambda x: 1)

# Now the agent can:
# 1. Choose to assign to Employee 0 or Employee 1
# 2. Choose to POSTPONE and wait for more tasks to arrive
```

### When Should Agents Postpone?

Agents learn to postpone when:

1. **Waiting for resource availability**: When no resources are currently available
2. **Gathering information**: When more tasks will arrive soon, making better decisions possible
3. **Reducing switching costs**: When postponing avoids expensive context switches
4. **Load balancing**: When deferring work allows for better distribution across resources

The agent learns these patterns through exploration and the reward signal.


## Postponement Per Net Component

A net often contains parts that share no place: separate processes, each with
its own queue and resources. `GymProblem.net_partition()` returns these
connected components of the place-transition graph (transitions that can never
fire are pruned first). Two transitions that share any place, such as a queue
or a resource pool, are in the same component.

With the default global postponement, choosing postpone freezes every
component until the next event. With component-scoped postponement, every
component that has an enabled action gets its own postpone option, and
choosing it blocks only that component until one of its own events fires; the
other components keep deciding:

```python
agency = GymProblem(allow_postpone=True)
# ... define the net ...
agency.postpone_scope = 'component'
```

With a single component the two scopes are identical.

## Net-Factored Credit (NF-GAE)

When the net has several components, the reward of one component cannot be
influenced by the decisions of another. Net-factored GAE uses this: each
decision is credited only with the rewards of its own component, on that
component's own decision clock, and the critic estimates the value of the
deciding component. With a single component it is exactly PPO with SMDP-GAE.

```python
agency.training_run(length=20, args_dict={
    "algorithm": "ppo-clip",
    "smdp_discount": True,   # discount the continuation by exp(-beta * tau)
    "beta": 0.5,
    "nfgae": True,
})
```

NF-GAE needs `allow_postpone=False` or `postpone_scope='component'`: a global
postpone couples the components. With `local_obs=True` (component turns,
work-conserving nets only) each decision observes and encodes only its own
component.
