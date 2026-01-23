# API Quick Reference

Quick lookup for commonly used classes and functions.

## Creating Environments

```python
from gympn.simulator import GymProblem

# Basic environment
env = GymProblem(
    allow_postpone=True,      # Enable postponement action
    causal_rl=True,           # Track causal relationships
    max_episode_length=1000   # Episode termination
)

# Add places (variables)
arrival = env.add_var(
    "arrival",
    var_attributes=['job_type', 'priority']
)

# Add events (transitions)
env.add_event(
    [arrival],                # Preconditions
    [arrival, waiting],       # Postconditions
    lambda token: [           # Behavior function
        SimToken(token, delay=1),
        SimToken(token)
    ],
    name='arrival_event',
    reward_function=lambda x: 0.0
)

# Add actions
env.add_action(
    [waiting, machine],       # Preconditions
    [processing],             # Postconditions
    behavior=lambda j, m: [SimToken((j, m), delay=1/m['speed'])],
    name='assign'
)
```

## Creating Networks

```python
from gympn.networks import GNNPolicyNetwork, GNNValueNetwork

metadata = env.make_metadata()

# Policy network
policy_net = GNNPolicyNetwork(
    input_dim=metadata['features'],
    hidden_dim=128,
    output_dim=metadata['num_actions'],
    num_layers=2,
    dropout=0.0
)

# Value network  
value_net = GNNValueNetwork(
    input_dim=metadata['features'],
    hidden_dim=128,
    num_layers=2,
    dropout=0.0
)
```

## Training Agents

### PPOAgent

```python
from gympn.agents import PPOAgent

# Create agent
agent = PPOAgent(
    policy_network=policy_net,
    value_network=value_net,
    method='clip',              # or 'penalty'
    eps=0.15,                   # PPO-Clip only
    c=0.2,                      # PPO-Penalty only
    policy_lr=8e-4,
    policy_updates=5,
    value_lr=8e-4,
    value_updates=10,
    gam=1.0,
    lam=0.99,
    ent_bonus=0.005
)

# Train
agent.train(
    env,
    episodes=64,
    epochs=100,
    batch_size=64,
    test_env=test_env,
    test_freq=10,
    logdir='data/train',
    num_workers=4,
    verbose=1
)

# Run single episode
reward, length = agent.run_episode(env, max_episode_length=1000)

# Run multiple episodes
returns, lengths = agent.run_episodes(
    env,
    episodes=64,
    num_workers=4
)
```

### DCLAgent

```python
from gympn.agents_dcl import DCLAgent
from gympn.dcl_planner import PlannerConfig

planner_cfg = PlannerConfig(
    horizon=5,
    rollouts_per_action=32,
    temperature=1.0,
    gamma=1.0,
    use_crn=True,
    use_lineage=True
)

agent = DCLAgent(
    policy_network=policy_net,
    value_network=value_net,
    planner_cfg=planner_cfg,
    policy_lr=5e-4,
    policy_updates=3,
    value_lr=5e-4,
    value_updates=5,
    gam=1.0,
    lam=0.99,
    ent_bonus=0.005
)

agent.train(env, episodes=32, epochs=25)
```

## Solvers and Evaluation

```python
from gympn.solvers import RandomSolver, HeuristicSolver, GymSolver

# Random baseline
random_solver = RandomSolver(seed=42)
reward, length = env.testing_run(random_solver)

# Heuristic solver
def my_heuristic(observable_net, tokens_comb):
    # Your strategy here
    for action_id, bindings in tokens_comb.items():
        return {action_id: bindings[0]}
    return 'postpone'

heuristic_solver = HeuristicSolver(heuristic_func=my_heuristic)
reward, length = env.testing_run(heuristic_solver)

# Learned policy
gym_solver = GymSolver(
    weights_path='data/train/run_id/best_policy.pth',
    metadata=env.make_metadata(),
    deterministic=True
)
reward, length = env.testing_run(gym_solver)

# Compare multiple
import numpy as np

results = {}
for name, solver in [
    ('random', random_solver),
    ('heuristic', heuristic_solver),
    ('learned', gym_solver)
]:
    returns = [env.testing_run(solver)[0] for _ in range(100)]
    results[name] = {
        'mean': np.mean(returns),
        'std': np.std(returns)
    }

for name, stats in results.items():
    print(f"{name}: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

## Logging and Visualization

```python
from gympn.logging_utils import setup_logging, get_logger
from gympn.visualisation import Visualisation

# Setup logging
setup_logging(
    logdir='data/train',
    level='INFO',
    use_tensorboard=True,
    use_wandb=False
)

logger = get_logger('training')
logger.info("Training started!")

# Visualization
vis = Visualisation(env)
vis.plot_net()                    # Network structure
vis.plot_marking(observable_net)  # Current state
vis.animate_episode(env, solver)  # Episode animation
```

## Common Hyperparameter Configs

### Fast Training
```python
config = {
    "algorithm": "ppo-clip",
    "episodes": 64,
    "epochs": 50,
    "batch_size": 64,
    "policy_lr": 8e-4,
    "value_lr": 8e-4,
    "policy_updates": 5,
    "value_updates": 10,
    "eps": 0.15,
    "ent_bonus": 0.005,
}
```

### Stable Training
```python
config = {
    "algorithm": "ppo-clip",
    "episodes": 128,
    "epochs": 100,
    "batch_size": 64,
    "policy_lr": 5e-4,
    "value_lr": 5e-4,
    "policy_updates": 8,
    "value_updates": 15,
    "eps": 0.1,
    "ent_bonus": 0.003,
}
```

### Planning-Based (DCL)
```python
config = {
    "algorithm": "dcl",
    "episodes": 32,
    "epochs": 25,
    "batch_size": 32,
    "policy_lr": 5e-4,
    "value_lr": 5e-4,
    "policy_updates": 3,
    "value_updates": 5,
    "dcl_horizon": 5,
    "dcl_rollouts": 32,
    "dcl_temp": 1.0,
    "ent_bonus": 0.005,
}
```

## Data Handling

```python
from gympn.data import ExperienceBuffer
import torch

# Create buffer
buffer = ExperienceBuffer(capacity=10000)

# Store experience
buffer.store(
    state=observation,
    action=action_id,
    reward=reward,
    logprob=log_prob,
    value=value_estimate,
    done=done
)

# Sample batch for training
batch = buffer.sample(batch_size=64)

# Get PyTorch tensors
states = torch.stack(batch['states'])
actions = torch.tensor(batch['actions'])
returns = torch.tensor(batch['returns'])
```

## Environment Inspection

```python
# Get environment metadata
metadata = env.make_metadata()
print(f"Features: {metadata['features']}")
print(f"Node types: {metadata['node_types']}")
print(f"Num actions: {metadata['num_actions']}")
print(f"Has postpone: {metadata['has_postpone']}")

# Get current observation
obs, info = env.reset()
print(f"Observation type: {type(obs)}")
print(f"Info: {info}")

# Get action space info
enabled_actions, tokens_comb = env.get_enabled_actions()
print(f"Enabled actions: {enabled_actions}")
print(f"Token bindings: {tokens_comb}")

# Get observable net
observable_net = env.observable_net
print(f"Observable marking: {observable_net.marking}")
```

## Metrics and Monitoring

```python
from gympn.logging_utils import TrainingMetrics

# Track metrics during training
metrics = TrainingMetrics()

metrics.log('episode_return', 25.5)
metrics.log('policy_loss', 0.045)
metrics.log('value_loss', 0.120)
metrics.log('kl_divergence', 0.023)
metrics.log('entropy', 0.314)

# Get statistics
print(metrics.get_stats('episode_return'))
# Output: {'mean': 24.2, 'std': 2.1, 'min': 18.5, 'max': 31.2}

# Reset for next epoch
metrics.reset()
```

## Common Patterns

### Training with Early Stopping

```python
best_return = -float('inf')
patience = 20
epochs_without_improvement = 0

for epoch in range(100):
    returns = agent.run_episodes(env, episodes=64)
    mean_return = np.mean(returns)
    
    if mean_return > best_return:
        best_return = mean_return
        epochs_without_improvement = 0
        agent.save_best_policy('best_policy.pth')
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stop at epoch {epoch}")
        break
```

### Cross-Validation

```python
import copy

def evaluate_with_seeds(env_factory, num_seeds=5):
    results = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = env_factory()
        agent = create_agent()
        agent.train(env, episodes=64, epochs=50)
        
        # Evaluate
        test_returns = []
        for _ in range(100):
            test_env = env_factory()
            r, _ = agent.run_episode(test_env)
            test_returns.append(r)
        
        results.append({
            'seed': seed,
            'mean': np.mean(test_returns),
            'std': np.std(test_returns)
        })
    
    return results
```

### Parallel Policy Evaluation

```python
from multiprocessing import Pool

def evaluate_policy(args):
    policy_path, num_episodes = args
    
    solver = GymSolver(weights_path=policy_path)
    env = GymProblem()
    
    returns = []
    for _ in range(num_episodes):
        r, _ = env.testing_run(solver)
        returns.append(r)
    
    return np.mean(returns), np.std(returns)

# Evaluate multiple policies in parallel
policies = [
    'policy_v1.pth',
    'policy_v2.pth',
    'policy_v3.pth'
]

with Pool(3) as pool:
    results = pool.map(
        evaluate_policy,
        [(p, 100) for p in policies]
    )

for policy, (mean, std) in zip(policies, results):
    print(f"{policy}: {mean:.2f} ± {std:.2f}")
```

## Command-Line Arguments Reference

All training arguments can be passed via command line or configuration dictionary. Use `make_parser()` to create an argument parser:

```python
from gympn.train import make_parser

parser = make_parser()
args = parser.parse_args()
```

### Environment Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--environment` | str | ActionEvolutionPetriNetEnv | Type of environment |
| `--env_seed` | int | 0 | Random seed for environment |

### Algorithm Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | ppo-clip | Training algorithm: ppo-clip, ppo-penalty, pg, or dcl |
| `--gam` | float | 1.0 | Discount factor γ (0-1) |
| `--lam` | float | 0.99 | GAE lambda parameter λ (0-1) |
| `--eps` | float | 0.2 | PPO clip range ε (ppo-clip only) |
| `--c` | float | 0.2 | KL penalty coefficient (ppo-penalty only) |
| `--ent_bonus` | float | 0.005 | Entropy regularization bonus |
| `--vf_coeff` | float | 0.5 | Value function loss weight |
| `--agent_seed` | int | 0 | Random seed for agent initialization |
| `--causal_rl` | bool | False | Enable causal credit redistribution |

### Policy Network Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--policy_model` | str | gnn | Policy network architecture (gnn) |
| `--policy_kwargs` | JSON | {"hidden_layers": [64]} | Network hyperparameters |
| `--policy_lr` | float | 3e-3 | Policy learning rate |
| `--policy_updates` | int | 10 | Policy optimization steps per epoch |
| `--policy_kld_limit` | float | 1.0 | KL divergence early stopping threshold |
| `--policy_weights` | str | "" | Legacy: initial policy weights file |
| `--policy_network` | str | "" | Path to policy checkpoint for resuming |
| `--score` | bool | False | Enable multi-objective training |
| `--score_weight` | float | 1e-3 | L2 regularization weight for multi-objective |

### Value Network Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--value_model` | str | gnn | Value network architecture (none or gnn) |
| `--value_kwargs` | JSON | {"hidden_layers": [64]} | Network hyperparameters |
| `--value_lr` | float | 3e-3 | Value learning rate |
| `--value_updates` | int | 40 | Value optimization steps per epoch |
| `--value_weights` | str | "" | Path to value network checkpoint |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 20 | Episodes collected per epoch |
| `--epochs` | int | 20 | Total number of training epochs |
| `--max_episode_length` | int/None | None | Max steps per episode (None = unlimited) |
| `--batch_size` | int/None | 64 | Mini-batch size for training |
| `--sort_states` | bool | False | Sort observations before batching |
| `--use_gpu` | bool | False | Use GPU if available |
| `--load_policy_network` | bool | False | Load previously trained policy as starting point |
| `--test_in_train` | bool | True | Evaluate agent during training |
| `--test_freq` | int | 1 | Test frequency (in epochs) |
| `--verbose` | int | 0 | Verbosity level (0=silent, 1=normal, 2=verbose) |
| `--num_workers` | int | 1 | **Number of workers for parallel episode collection** |

### Logging Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_wandb` | bool | True | Enable Weights & Biases logging |
| `--wandb_mode` | str | offline | W&B sync mode: online, offline, or disabled |
| `--wandb_project` | str | gympn-training | W&B project name |
| `--wandb_entity` | str | None | W&B entity (username/team name) |
| `--open_wandb` | bool | True | Auto-open W&B dashboard |

### DCL Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dcl_horizon` | int | 5 | Planning lookahead steps |
| `--dcl_rollouts` | int | 32 | Monte Carlo rollouts per action |
| `--dcl_temp` | float | 1.0 | Temperature for action sampling |

### Saving Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--name` | str | run | Run identifier/name |
| `--datetag` | bool | True | Append timestamp to run name |
| `--logdir` | str | data/train | Base directory for training runs |
| `--save_freq` | int | 1 | Model checkpoint frequency (in epochs) |
| `--open_tensorboard` | bool | False | Auto-launch TensorBoard dashboard |

### Command-Line Usage Examples

```bash
# Basic PPO-Clip training
python examples/example_simple_postpone.py \
  --algorithm ppo-clip \
  --episodes 64 \
  --epochs 100 \
  --policy_lr 5e-4 \
  --num_workers 4

# DCL agent with planning
python examples/example_simple_postpone.py \
  --algorithm dcl \
  --dcl_horizon 5 \
  --dcl_rollouts 32 \
  --num_workers 2

# Causal RL enabled
python examples/example_simple_postpone.py \
  --causal_rl true \
  --num_workers 4 \
  --wandb_mode online

# Resume from checkpoint
python examples/example_simple_postpone.py \
  --policy_network data/train/best_policy.pth \
  --num_workers 1
```

### Python Configuration Dictionary

Instead of command-line arguments, you can pass a configuration dictionary:

```python
config = {
    "algorithm": "ppo-clip",
    "episodes": 64,
    "epochs": 100,
    "policy_lr": 5e-4,
    "value_lr": 5e-4,
    "causal_rl": True,
    "num_workers": 4,  # Parallel episode collection
    "batch_size": 64,
    "use_wandb": True,
    "wandb_mode": "online",
}

agency = GymProblem(...)
agency.training_run(length=100, args_dict=config)
```
