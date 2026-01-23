# Postponing and Causal Reinforcement Learning

## Introduction

While Action-Evolution Petri Nets (A-E PNs) provide a powerful framework for modeling decision-making problems, two advanced features significantly enhance their applicability to complex real-world scenarios: **postponement** and **causal reinforcement learning (CRL)**.

Postponement allows agents to defer decisions when the environment is not ready, while causal reinforcement learning enables agents to trace reward origins back to the decisions that caused them. Together, these features address critical challenges in temporal credit assignment and delayed consequences in complex systems.

## Motivation

### The Postponement Problem

In many real-world processes, making an immediate decision is not always optimal or even possible. Consider a manufacturing system where an operator must decide when to process a batch of items:

1. **Premature decisions**: Acting before sufficient data arrives may lead to suboptimal choices.
2. **Resource constraints**: Resources may be unavailable at the current decision point.
3. **Sequential dependencies**: Some decisions depend on the outcomes of evolution transitions that haven't fired yet.

Traditional A-E PN frameworks require the agent to make a decision whenever an action transition is enabled. This can force suboptimal actions or require complex workarounds to model delayed decision-making.

**Postponement** addresses this by allowing agents to skip the current decision point and let the environment evolve naturally, deferring the choice until a more opportune moment.

### The Credit Assignment Problem

In A-E PN environments with non-immediate rewards, standard reinforcement learning struggles with **credit assignment**: determining which actions deserve credit for delayed rewards.

Consider this scenario in our task assignment example:
- At time 0: Agent assigns Task A to Employee 0 (fast processor)
- At time 1: Task A completes and generates a reward
- At time 2-5: Multiple other decisions are made
- At time 6: A side effect of Task A's completion triggers a cascading benefit

In this case, which actions should receive credit for the time-6 reward?

**Causal Reinforcement Learning** solves this by tracking the **causal chain** of events: which tokens were produced by which actions, and which rewards resulted from using those tokens. By tracing these dependencies backward, we can assign credit to the decisions that truly caused the reward.

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
agency = GymProblem(allow_postpone=True, causal_rl=False)

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

## Causal Reinforcement Learning

### The Core Concept

Causal Reinforcement Learning tracks **token lineage**: the ancestry of tokens back through the transitions that created them. When a reward is generated, CRL traces back through this lineage to identify which initial actions were responsible.

Key concepts:

- **Token IDs**: Every token produced by action transitions receives a unique identifier
- **Token History**: A record of which tokens produced which new tokens
- **Reward Redistribution**: Rewards are traced backward through token dependencies to credit the original actions

### How Rewards Are Distributed

Consider this example:

**Timeline:**
- Time 0: Action A produces Token T1
- Time 1: Action B uses Token T1 to produce Token T2
- Time 2: Action C uses Token T2 to produce Token T3
- Time 3: Transition fires using Token T3 to generate Reward = 10

**Without CRL:**
- The reward at time 3 has no clear connection to the action at time 0
- Standard GAE/advantage estimation may misattribute credit

**With CRL:**
1. Trace Reward backward: which tokens were used? → Token T3
2. Trace Token T3: which action produced it? → Action C (at time 2)
3. Trace Token T3's parents: Token T2 was used
4. Trace Token T2: which action produced it? → Action B (at time 1)
5. Trace Token T2's parents: Token T1 was used
6. Trace Token T1: which action produced it? → Action A (at time 0)

The reward is distributed:
- Action A receives credit: ~5.0 (originated the causal chain)
- Action B receives credit: ~3.0 (continued the chain)
- Action C receives credit: ~2.0 (completed the chain)

### Implementation Example

Enable causal RL at environment creation:

```python
# Enable both postponement and causal RL
agency = GymProblem(allow_postpone=True, causal_rl=True)
```

When causal RL is enabled:

1. **Token Registration**: Every token produced by action transitions automatically receives a unique ID
2. **Lineage Tracking**: The system records which tokens were used to produce new tokens
3. **Reward Redistribution**: After each episode, rewards are traced backward and redistributed to actions

### The Benefit for Delayed Rewards

Consider a manufacturing scenario:

```
Decision Point (t=0):
- Assign Job A to Machine X

Job Processing (t=1-5):
- Machine X processes Job A

Side Effect (t=6):
- Job A completion enables Job B to start

Downstream Reward (t=10):
- Job B completes successfully, generating reward = 100
```

**Standard RL Problem:**
- Gap between action (t=0) and reward (t=10) is 10 steps
- Other decisions made in between make credit assignment ambiguous

**CRL Solution:**
- At t=10, identify tokens involved in reward
- Trace backward: tokens from Job B completion
- Trace further: Job B started due to Job A completion
- Trace even further: Job A was assigned at t=0
- Assign credit to the t=0 decision directly

### Implementation Details

In `gympn`, causal tracking is implemented through:

1. **Token History** (`causal_traces.py`):
   - Records the lineage of every token
   - Tracks which tokens consumed which predecessors

2. **Eligibility Credits** (`data.py`):
   - Computes the credit each action receives
   - Accounts for both direct and indirect contributions

3. **Reward Redistribution** (`agents.py`):
   - Uses eligibility credits to reshape the reward signal
   - Distributes delayed rewards to earlier actions proportionally

## Combining Postponement and Causal RL

The true power emerges when combining both features:

### Scenario: Intelligent Batch Processing

```python
agency = GymProblem(allow_postpone=True, causal_rl=True)

# Setup task assignment with postponement and causal tracking...

def heuristic_policy(observable_net, tokens_comb):
    """
    Example heuristic that demonstrates postponement strategy.
    """
    # Try to find an optimal assignment
    for k, el in tokens_comb.items():
        for binding in el:
            task = binding[0][1].value
            resource = binding[1][1].value
            
            # Only assign if resource matches task type
            if resource['code_employee'] == task['task_type']:
                return {k: binding}
    
    # If no good match found, POSTPONE and wait for better options
    return 'postpone'
```

### How CRL Enhances Postponement

1. **Learning postpone value**: CRL correctly attributes rewards to postpone actions, helping the agent learn when deferring decisions is beneficial

2. **Valuing patience**: The causal chain shows that postponing to wait for a better resource pairing eventually leads to faster completion

3. **Chained decisions**: When one postponement leads to better future decisions, the entire chain receives appropriate credit

## Complete Example

Here's a complete example using both features:

```python
from gympn.simulator import GymProblem
from simpn.simulator import SimToken

# Enable both postponement and causal RL
agency = GymProblem(allow_postpone=True, causal_rl=True)

# Define environment
arrival = agency.add_var("arrival", var_attributes=['task_type'])
waiting = agency.add_var("waiting", var_attributes=['task_type'])
busy = agency.add_var("busy", var_attributes=['task_type', 'code_employee'])

arrival.put({'task_type': 0})
arrival.put({'task_type': 0})

employee = agency.add_var("employee", var_attributes=['code_employee'])
employee.put({'code_employee': 0})
employee.put({'code_employee': 1})

def arrive(a):
    return [SimToken(a, delay=1), SimToken(a)]

agency.add_event([arrival], [arrival, waiting], arrive)

def start(c, r):
    processing_time = 0.5 if r['code_employee'] == 0 else 2.0
    return [SimToken((c, r), delay=processing_time)]

agency.add_action([waiting, employee], [busy], behavior=start, name="start")

def complete(b):
    return [SimToken(b[1])]

agency.add_event([busy], [employee], complete, name='complete',
                 reward_function=lambda x: 1)

# Configure training with causal RL
training_config = {
    "algorithm": "ppo-clip",
    "gam": 1.0,
    "lam": 0.99,
    "episodes": 64,
    "epochs": 100,
    "batch_size": 64,
    "policy_lr": 8e-4,
    "value_lr": 8e-4,
    "policy_model": "gnn",
    "value_model": "gnn",
}

agency.training_run(length=10, args_dict=training_config)
```

## Key Takeaways

1. **Postponement** enables agents to defer decisions, which is crucial for:
   - Waiting for optimal conditions
   - Avoiding premature commitments
   - Learning patience and strategic deferral

2. **Causal RL** solves credit assignment by:
   - Tracing token lineage
   - Identifying causal chains from actions to rewards
   - Distributing credit fairly across the decision chain

3. **Combined power**: Using both features allows agents to:
   - Learn when to postpone intelligently
   - Understand that deferring decisions can lead to better long-term outcomes
   - Correctly attribute credit even with long delays and complex dependencies

4. **Practical benefits**:
   - Better convergence in complex environments
   - More interpretable credit assignment
   - Alignment with human intuition about decision-making

