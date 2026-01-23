# Advanced Features

## Graph Neural Networks for Observations

### Overview

GymPN uses Graph Neural Networks (GNNs) to process Petri Net observations. The network structure directly represents the PN structure, enabling agents to reason about state transitions and dependencies.

### Graph Representation

Each observation is represented as a heterogeneous graph with multiple node and edge types:

**Node Types:**
- `place`: Represents places/variables in the PN
- `a_transition`: Represents action transitions
- `e_transition`: Represents evolution transitions (events)
- `token`: Represents tokens currently in the system
- `postpone`: Special node for postponement action (if enabled)

**Edge Types:**
- `input`: Token to transition (prerequisite)
- `output`: Transition to token (produces)
- `reset`: Transition resets token

### Available Network Architectures

#### GNNPolicyNetwork

Used for learning action probabilities:

```python
from gympn.networks import GNNPolicyNetwork

policy_net = GNNPolicyNetwork(
    input_dim=64,           # Feature dimension (from metadata)
    hidden_dim=128,         # Hidden layer dimension
    output_dim=num_actions, # Number of actions
    num_layers=2,           # Graph convolution layers
    dropout=0.0             # Dropout rate
)
```

**Architecture:**
1. Input layer: `input_dim` → `hidden_dim`
2. Graph convolutions: `hidden_dim` → `hidden_dim` (repeated `num_layers` times)
3. Output layer: `hidden_dim` → `output_dim`

#### GNNValueNetwork

Used for estimating expected returns:

```python
from gympn.networks import GNNValueNetwork

value_net = GNNValueNetwork(
    input_dim=64,
    hidden_dim=128,
    num_layers=2,
    dropout=0.0
)
```

**Output:** Single scalar value per graph

#### Custom Network Design

You can design custom networks for your specific problem:

```python
import torch
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData

class CustomGNNPolicy(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                HeteroConv({
                    ('place', 'input', 'a_transition'): GCNConv(in_dim, hidden_dim),
                    ('place', 'output', 'a_transition'): GCNConv(in_dim, hidden_dim),
                    # ... more edge types
                }, aggr='add')
            )
        
        self.head = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Aggregate action node features
        a_trans_features = x_dict['a_transition']
        action_logits = self.head(a_trans_features)
        return action_logits
```

### Getting Feature Dimensions

Use `make_metadata()` to get required dimensions:

```python
metadata = env.make_metadata()

print(f"Node types: {metadata['node_types']}")
print(f"Feature dim: {metadata['features']}")
print(f"Num actions: {metadata['num_actions']}")
print(f"Has postpone: {metadata['has_postpone']}")
```

## Deep Causal Learning (DCL)

### How DCL Works

DCL improves upon standard PPO by incorporating a planning phase:

**Standard PPO:**
```
Observe state → Policy(state) → Action → Execute → Collect reward
```

**DCL with Planning:**
```
Observe state → Policy(state) → Sample rollouts → Evaluate → Improved policy → Train to match → Action → Execute
```

### Planning Algorithm

The DCL planner works in three steps:

#### 1. Rollout Generation

For each possible action, generate `rollouts_per_action` trajectories:

```python
# Pseudocode
for each action_binding:
    trajectories = []
    for i in range(rollouts_per_action):
        # Sample a trajectory under current policy + noise
        trajectory = simulate_forward(
            state, 
            action_binding,
            horizon=horizon,
            temperature=temperature  # Softmax temperature for action sampling
        )
        trajectories.append(trajectory)
```

#### 2. Trajectory Evaluation

Evaluate each trajectory using the current value function:

```python
def evaluate_trajectory(trajectory, value_net):
    """Compute cumulative discounted reward."""
    value = 0
    for t, transition in enumerate(trajectory):
        reward = transition.reward
        discount = gamma ** t
        value += discount * reward
    return value
```

#### 3. Policy Improvement

Compute improved policy probabilities based on trajectory values:

```python
# Get returns for all trajectories per action
returns_per_action = [
    mean_return(trajectories_for_action)
    for action in all_actions
]

# Softmax improvement
improved_pi = softmax(returns_per_action / temperature)

# Train policy to match improved distribution
policy_loss = cross_entropy(policy_logits, improved_pi)
```

### PlannerConfig Options

```python
from gympn.dcl_planner import PlannerConfig

config = PlannerConfig(
    # Lookahead
    horizon=5,                    # How many steps to look ahead
    
    # Sampling
    rollouts_per_action=32,       # Trajectories per action
    temperature=1.0,              # Softmax temperature (higher = more uniform)
    
    # Variance reduction
    use_crn=True,                # Common Random Numbers for stable comparison
    use_lineage=True,            # Track token lineage for causal RL
    
    # Discounting
    gamma=1.0,                   # Discount factor (1.0 for episodic)
    
    # Numerical stability
    min_prob=1e-10,              # Minimum probability for log
    max_log_prob=10.0            # Clamp log probabilities
)
```

### Tuning DCL

**Horizon too short (≤2):**
- Agent only sees immediate consequences
- May miss important causal chains

**Horizon too long (≥10):**
- Exponentially increases computation
- Sampling noise dominates

**Recommendation:** Start with 5, adjust based on problem temporal scale

**Few rollouts (≤16):**
- Noisy action value estimates
- Poor planning quality

**Many rollouts (≥64):**
- Accurate estimates but slow
- Training may become infeasible

**Recommendation:** Start with 32, increase for complex problems

**Temperature effects:**
- Temperature = 0.1: Sharp distribution (exploitation)
- Temperature = 1.0: Balanced
- Temperature = 10.0: Uniform (exploration)

**Recommendation:** 1.0 is usually best

## Token Lineage and Causal Tracking

### How Causal Tracking Works

When `causal_rl=True`, the system tracks:

1. **Token Identities:** Every token produced by actions gets a unique ID
2. **Parent-Child Relationships:** Which tokens were used to create new tokens
3. **Action-Token Mapping:** Which action produced which tokens
4. **Reward Causality:** Which tokens contributed to which rewards

### Implementation Details

```python
# In your environment
env = GymProblem(allow_postpone=True, causal_rl=True)

# Now every token has metadata:
# {
#     'value': {...},           # Original token attributes
#     'token_id': 'uuid-...',   # Unique identifier
#     'produced_by_action': 3,  # Which action created this
#     'produced_at_time': 1.5   # When it was created
# }
```

### Accessing Causal Information

During training, you can access causal information:

```python
# In episode callbacks
def log_causal_info(episode_data):
    # episode_data contains:
    # - actions_taken: list of actions
    # - token_ids: list of token IDs produced
    # - rewards: raw rewards
    # - credits: causal credits per action
    
    for action_idx, credit in enumerate(episode_data['credits']):
        print(f"Action {action_idx} received credit: {credit}")
```

### Reward Redistribution Algorithm

The causal RL system uses this algorithm to distribute rewards:

```
For each reward event:
    1. Identify tokens used in the event
    2. Trace backward through parent relationships
    3. Find all actions that contributed to this causal chain
    4. Distribute reward proportionally based on contribution depth
    5. Update action credits

Action credit = sum of all redistributed rewards from its tokens
```

### When to Use Causal RL

**Beneficial when:**
- Rewards are delayed relative to actions
- Complex chains of events between action and reward
- Need interpretability (which actions caused rewards?)
- Using postponement strategically

**Example scenario:**
```
Time 0: Action A produces Token T1
Time 1: Action B uses T1 to produce T2
Time 2: Action C uses T2 to produce T3
Time 5: T3 used to fire transition, generate reward = 100

Without causal RL:
- T3 sees reward at time 5
- T2 gets no credit
- T1 gets no credit
- No clear learning signal for A, B

With causal RL:
- Reward traced backward: T3 ← T2 ← T1
- T3, T2, T1 all get credit
- Actions A, B, C all learn from reward
```

### Visualizing Causal Chains

```python
# After training, inspect causal structure
from gympn.visualisation import visualize_causal_chain

# Get a specific episode
episode_data = env.get_last_episode()

# Visualize which actions contributed to a reward
visualize_causal_chain(
    episode_data,
    reward_event_id=5,
    show_token_genealogy=True
)
```

## Distributed Training (Parallelization)

### Using Multiprocessing

GymPN automatically parallelizes episode collection:

```python
agent.train(
    env,
    episodes=64,
    epochs=100,
    num_workers=4  # Collect episodes in parallel
)
```

**How it works:**
1. Main process orchestrates training
2. Worker processes collect episodes independently
3. Episodes gathered into batch
4. Main process updates networks
5. Repeat

### Performance Considerations

```
Speedup roughly scales with number of workers:
- 1 worker: baseline
- 4 workers: ~3.5x speedup
- 8 workers: ~7x speedup
- Beyond 8: diminishing returns (GIL, communication overhead)
```

**Recommended settings:**
```python
# For CPU-bound (most cases)
num_workers = min(4, cpu_count() - 1)

# For GPU (set use_gpu=True)
num_workers = min(2, cpu_count() - 1)  # Smaller to avoid memory issues
```

### Memory Efficiency

To reduce memory per worker:

```python
config = {
    "num_workers": 4,
    "episodes": 64,           # Divide by number of workers for same total
    "batch_size": 64,         # Can be larger than episodes if distributed
    "use_gpu": False,         # CPU is more memory efficient for workers
}
```

## Graph Visualization

### Visualizing Network Structure

```python
from gympn.visualisation import Visualisation

vis = Visualisation(env)

# Show network structure
vis.plot_net()

# Show a specific marking
vis.plot_marking(observable_net)

# Animate episode
vis.animate_episode(
    env,
    solver,
    max_steps=100,
    save_path='episode.gif'
)
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
import networkx as nx

# Get network structure
places = [v.name for v in env.variables]
transitions = [t.name for t in env.events + env.actions]

# Build graph
G = nx.DiGraph()
G.add_nodes_from(places, node_type='place')
G.add_nodes_from(transitions, node_type='transition')

# Add edges based on preconditions/postconditions
for action in env.actions:
    for precond in action.preconditions:
        G.add_edge(precond.name, action.name)
    for postcond in action.postconditions:
        G.add_edge(action.name, postcond.name)

# Draw
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()
```

## Custom Reward Functions

### Complex Reward Design

```python
def complex_reward(token_binding):
    """
    Example: Multi-objective reward function
    Rewards completion but penalizes lateness
    """
    task, resource = token_binding
    
    # Base completion reward
    reward = 1.0
    
    # Penalty for using slow resource
    if resource['code_employee'] == 1:  # Slow employee
        reward -= 0.3
    
    # Bonus for fast completion
    if resource['code_employee'] == 0:  # Fast employee
        reward += 0.2
    
    return reward

agency.add_event(
    [busy], [employee],
    complete,
    name='complete',
    reward_function=complex_reward
)
```

### Reward Shaping

To guide agent learning:

```python
def shaped_reward(token_binding):
    """Reward with potential-based shaping."""
    task, resource = token_binding
    
    base_reward = 1.0
    
    # Potential-based shaping (safe for convergence)
    phi_current = compute_potential(task, resource)
    phi_next = 0  # Terminal state
    
    shaping = 0.99 * phi_next - phi_current
    
    return base_reward + shaping

def compute_potential(task, resource):
    """Compute potential based on task characteristics."""
    # Lower potential for harder tasks
    if task['task_type'] == 0:
        return 0.5  # Simple task
    else:
        return 1.0  # Complex task
```

## Observation Filtering and Normalization

### Custom Observations

```python
class CustomObservationProcessor:
    def __init__(self, env):
        self.env = env
    
    def process(self, observable_net):
        """Convert observable net to custom representation."""
        # Get graph representation
        graph = self.env.make_observation(observable_net)
        
        # Add custom features
        graph.x_dict['place'] = self._add_place_features(
            graph.x_dict['place'],
            observable_net
        )
        
        return graph
    
    def _add_place_features(self, features, observable_net):
        """Add domain-specific features."""
        # Example: Add queue length as feature
        for i, place in enumerate(self.env.variables):
            tokens = observable_net.marking[place.name]
            queue_length = len(tokens)
            features[i, 0] = queue_length  # Feature slot 0 = queue length
        
        return features
```


