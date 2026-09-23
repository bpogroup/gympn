# Agents Guide

## Overview

GymPN provides one main agent for training in Action-Evolution Petri Net environments:

1. **PPOAgent** - Proximal Policy Optimization

Additionally, you can create custom solvers for evaluation and heuristic strategies.

## PPOAgent

### What it does

PPOAgent is a policy gradient agent that uses PPO to learn an optimal policy. It maintains two neural networks:
- **Policy network**: Maps observations to action probabilities
- **Value network**: Estimates the expected return from a state

### Supported PPO Variants

```python
from gympn.agents import PPOAgent

# PPO with clipping (recommended for most cases)
agent = PPOAgent(
    policy_network=policy_net,
    value_network=value_net,
    method='clip',  # or 'penalty'
    eps=0.15,       # Clipping range (for clip method)
    c=0.2,          # Penalty coefficient (for penalty method)
    policy_lr=8e-4,
    policy_updates=5,
    value_lr=8e-4,
    value_updates=10,
    gam=1.0,        # Discount factor
    lam=0.99,       # GAE lambda
    kld_limit=0.1,  # KL divergence limit (for penalty method)
    ent_bonus=0.005 # Entropy bonus
)
```

### Key Parameters

| Parameter | Default | Range | Meaning |
|-----------|---------|-------|---------|
| `method` | 'clip' | 'clip', 'penalty' | PPO variant |
| `eps` | 0.15 | 0.05-0.2 | Clipping range (clip) |
| `c` | 0.2 | 0.1-0.5 | Penalty weight (penalty) |
| `policy_lr` | 5e-4 | 1e-5 to 1e-2 | Policy learning rate |
| `policy_updates` | 5 | 1-20 | Updates per batch |
| `value_lr` | 5e-4 | 1e-5 to 1e-2 | Value learning rate |
| `value_updates` | 10 | 3-30 | Value updates per batch |
| `gam` | 0.99 | 0.9-1.0 | Discount factor |
| `lam` | 0.95 | 0.9-0.99 | GAE lambda |
| `kld_limit` | 0.1 | 0.01-0.5 | Max KL divergence (penalty) |
| `ent_bonus` | 0.0 | 0.0-0.1 | Entropy regularization |

### Methods

#### `train()`
Main training loop that runs multiple episodes and epochs.

```python
agent.train(
    env,
    episodes=64,           # Episodes per epoch
    epochs=100,            # Number of epochs
    batch_size=64,         # Training batch size
    max_episode_length=None,
    test_env=None,         # Optional test environment
    test_freq=10,          # Test every N epochs
    logdir='data/train',
    verbose=1
)
```

#### `run_episode()`
Runs a single episode and collects experience.

```python
reward, length = agent.run_episode(
    env,
    max_episode_length=1000,
    store=True  # Store in buffer for training
)
```

#### `run_episodes()`
Parallel episode collection using multiprocessing.

```python
returns, lengths = agent.run_episodes(
    env,
    episodes=64,
    max_episode_length=1000,
    num_workers=4
)
```

### Example: Training with PPOAgent

```python
from gympn.agents import PPOAgent
from gympn.networks import GNNPolicyNetwork, GNNValueNetwork
from gympn.simulator import GymProblem

# Create environment
env = GymProblem(allow_postpone=True, causal_rl=True)
# ... define places, transitions, events ...

# Create networks
policy_net = GNNPolicyNetwork(
    input_dim=env.make_metadata()['features'],
    hidden_dim=128,
    output_dim=len(env.action_transitions)
)

value_net = GNNValueNetwork(
    input_dim=env.make_metadata()['features'],
    hidden_dim=128
)

# Create agent
agent = PPOAgent(
    policy_network=policy_net,
    value_network=value_net,
    method='clip',
    eps=0.15,
    policy_lr=8e-4,
    value_lr=8e-4,
    gam=1.0,
    lam=0.99
)

# Train
agent.train(
    env,
    episodes=64,
    epochs=100,
    batch_size=64,
    logdir='data/train'
)
```

## Custom Solvers

### Creating a Custom Solver

You can create custom solvers for evaluation, heuristics, or baselines:

```python
from gympn.solvers import Solver

class MyCustomSolver(Solver):
    def __init__(self, params=None):
        super().__init__(params)
    
    def solve(self, observable_net, tokens_comb):
        """
        Args:
            observable_net: The observable marking of the net
            tokens_comb: Available token combinations for actions
        
        Returns:
            dict: Selected action binding, or 'postpone'
        """
        # Your logic here
        for action_id, bindings in tokens_comb.items():
            for binding in bindings:
                # Implement your strategy
                return {action_id: binding}
        
        return 'postpone'
```

### Built-in Solvers

#### RandomSolver
Selects actions uniformly at random.

```python
from gympn.solvers import RandomSolver

solver = RandomSolver(seed=42)
reward, length = env.testing_run(solver)
```

#### HeuristicSolver
Uses a provided heuristic function.

```python
from gympn.solvers import HeuristicSolver

def my_heuristic(observable_net, tokens_comb):
    # Your logic
    for action_id, bindings in tokens_comb.items():
        # Select based on criteria
        if suitable_binding(bindings[0]):
            return {action_id: bindings[0]}
    return 'postpone'

solver = HeuristicSolver(heuristic_func=my_heuristic)
reward, length = env.testing_run(solver)
```

#### GymSolver
Uses a trained neural network policy.

```python
from gympn.solvers import GymSolver

solver = GymSolver(
    weights_path='data/train/run_id/best_policy.pth',
    metadata=env.make_metadata(),
    deterministic=True
)
reward, length = env.testing_run(solver)
```

### Example: Custom Heuristic for Task Assignment

```python
def task_assignment_heuristic(observable_net, tokens_comb):
    """
    Assign tasks to employees based on task type preference.
    """
    for action_id, bindings in tokens_comb.items():
        best_binding = None
        best_score = -1
        
        for binding in bindings:
            task = binding[0][1].value  # First token is the task
            employee = binding[1][1].value  # Second token is employee
            
            # Prefer matching task type to employee code
            score = 1 if task['task_type'] == employee['code_employee'] else 0
            
            if score > best_score:
                best_score = score
                best_binding = binding
        
        if best_binding is not None:
            return {action_id: best_binding}
    
    return 'postpone'
```

## Comparing Agents

### Evaluation Framework

```python
import copy
import numpy as np

def evaluate_agent(problem, agent_or_solver, num_episodes=100):
    """Evaluate an agent/solver over multiple episodes."""
    returns = []
    for _ in range(num_episodes):
        problem_copy = copy.deepcopy(problem)
        reward = problem_copy.testing_run(agent_or_solver)
        returns.append(reward)
    
    return {
        'mean': np.mean(returns),
        'std': np.std(returns),
        'min': np.min(returns),
        'max': np.max(returns)
    }

# Compare multiple approaches
env = GymProblem(allow_postpone=True, causal_rl=True)
# ... setup environment ...

results = {}

# Random baseline
from gympn.solvers import RandomSolver
results['random'] = evaluate_agent(env, RandomSolver(), 100)

# Heuristic
from gympn.solvers import HeuristicSolver
results['heuristic'] = evaluate_agent(
    env, 
    HeuristicSolver(heuristic_func=my_heuristic),
    100
)

# Trained agent
from gympn.solvers import GymSolver
results['learned'] = evaluate_agent(
    env,
    GymSolver(weights_path='...', metadata=env.make_metadata()),
    100
)

# Print comparison
for name, stats in results.items():
    print(f"{name}: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

### Performance Comparison Example

```
Random:    12.5 ± 3.2
Heuristic: 18.3 ± 2.1
PPO:       22.1 ± 1.8
```

## Troubleshooting Agents

### Issue: Agent isn't learning

**Symptoms:** Loss doesn't decrease, return stays flat

**Solutions:**
1. Increase policy_lr (5e-4 → 8e-4)
2. Increase entropy bonus (0.005 → 0.01)
3. Increase policy_updates (3 → 8)
4. Check that rewards are being generated
5. Visualize episodes to see if agent is exploring

### Issue: Training is unstable

**Symptoms:** Loss oscillates, returns vary wildly

**Solutions:**
1. Decrease policy_lr (8e-4 → 5e-4)
2. Decrease eps (0.15 → 0.1) for PPO-Clip
3. Increase batch_size
4. Use PPO-Penalty instead of PPO-Clip
5. Reduce ent_bonus

### Issue: Agent overfits to training episodes

**Symptoms:** High training return, low test return

**Solutions:**
1. Increase entropy bonus
2. Decrease network hidden dimensions
3. Add test environment and monitor gap
4. Use dropout in networks
5. Increase epochs with fewer episodes per epoch


