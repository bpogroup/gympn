# Troubleshooting Guide

## Common Issues and Solutions

### Training Issues

#### Agent Not Learning

**Symptoms:**
- Loss stays constant or increases
- Return stays flat across epochs
- Policy doesn't change

**Diagnostics:**

```python
# Check 1: Are episodes being collected?
from gympn.logging_utils import get_logger
logger = get_logger('training')
# Should see episode rewards, lengths in logs

# Check 2: Are rewards being generated?
obs, info = env.reset()
done = False
while not done:
    action = agent.policy_net(obs)
    obs, reward, done, truncated, info = env.step(action)
    if reward > 0:
        print(f"✓ Reward: {reward}")

# Check 3: Are gradients flowing?
for param in agent.policy_net.parameters():
    if param.grad is None:
        print("✗ No gradient for parameter!")
```

**Solutions (in order of likelihood):**

1. **Learning rate too high** (most common)
   ```python
   # Try: Reduce by 50%
   policy_lr=5e-4,  # was 1e-3
   ```

2. **Learning rate too low**
   ```python
   # Symptoms: Very slow improvement
   # Try: Increase by 2x
   policy_lr=1e-3,  # was 5e-4
   ```

3. **Reward structure wrong**
   ```python
   # Check that rewards make sense
   def reward_func(token_binding):
       # Make sure this returns positive values for good outcomes
       return 1.0 if good else 0.0
   ```

4. **Network architecture mismatch**
   ```python
   # Make sure output dimension matches number of actions
   metadata = env.make_metadata()
   num_actions = metadata['num_actions']
   
   # Should have correct output dim
   policy_net = GNNPolicyNetwork(
       input_dim=metadata['features'],
       output_dim=num_actions  # ✓ Correct
   )
   ```

5. **Batch size too large**
   ```python
   # Batch larger than episodes causes issues
   # Solution: batch_size <= episodes
   config = {
       "episodes": 64,
       "batch_size": 64,  # ✓ Good
   }
   ```

6. **Use entropy bonus to force exploration**
   ```python
   config = {
       "ent_bonus": 0.01,  # Increased from 0.005
   }
   ```

#### Training is Unstable (High Variance)

**Symptoms:**
- Loss oscillates wildly
- Returns vary dramatically between epochs
- Occasional divergence

**Solutions:**

1. **Reduce learning rate**
   ```python
   policy_lr=3e-4,  # was 8e-4
   value_lr=3e-4,
   ```

2. **Reduce policy updates per batch**
   ```python
   policy_updates=3,  # was 5
   ```

3. **Reduce clipping range (PPO-Clip)**
   ```python
   eps=0.1,  # was 0.15
   ```

4. **Increase batch size**
   ```python
   batch_size=128,  # was 64
   ```

5. **Use PPO-Penalty instead of PPO-Clip**
   ```python
   method='penalty',
   c=0.2,
   kld_limit=0.05,
   ```

6. **Add gradient clipping**
   ```python
   # In training loop:
   torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
   ```

#### Training is Extremely Slow

**Symptoms:**
- Takes hours for 100 epochs
- 1 epoch takes >1 minute
- GPU is underutilized

**Solutions:**

1. **Reduce batch size if memory allows**
   ```python
   batch_size=32,  # was 64
   ```

2. **Use fewer policy/value updates**
   ```python
   policy_updates=2,  # was 5
   value_updates=5,   # was 10
   ```

3. **Enable parallelization**
   ```python
   agent.train(
       env,
       episodes=64,
       num_workers=4  # Parallel episode collection
   )
   ```

4. **Reduce network size**
   ```python
   policy_net = GNNPolicyNetwork(
       hidden_dim=64,  # was 128
   )
   ```

5. **Use PG instead of PPO**
   ```python
   # PG is fastest but less stable
   agent = PGAgent(policy_net, value_net)
   ```

6. **For DCL specifically: Reduce horizon or rollouts**
   ```python
   planner_cfg = PlannerConfig(
       horizon=3,              # was 5
       rollouts_per_action=16  # was 32
   )
   ```

---

### Environment Issues

#### Environment Crashes or Hangs

**Symptoms:**
- Episode collection never completes
- Memory usage grows indefinitely
- Process hangs

**Diagnostics:**

```python
# Add timeout to detect hangs
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Episode timeout!")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.sample_action(obs)
        obs, reward, done, truncated, info = env.step(action)
finally:
    signal.alarm(0)
```

**Solutions:**

1. **Add maximum episode length**
   ```python
   agent.run_episodes(
       env,
       max_episode_length=1000  # Prevent infinite loops
   )
   ```

2. **Check for infinite event loops**
   ```python
   # Make sure events don't create infinite tokens
   def event_behavior(token):
       # ✓ Correct: produces finite tokens
       return [SimToken(token, delay=1)]
       
       # ✗ Wrong: infinite production
       # while True:
       #     produce_token()
   ```

3. **Verify transitions have valid preconditions**
   ```python
   # All transitions should have preconditions
   # Otherwise they fire every step
   add_event(
       [place1],  # ✓ Has precondition
       [place2],
       behavior
   )
   ```

4. **Check for memory leaks in callbacks**
   ```python
   # Don't accumulate data in callbacks
   def callback(episode_data):
       # ✗ Wrong: grows without limit
       # global_list.append(episode_data)
       
       # ✓ Correct: process and forget
       process_episode(episode_data)
   ```

#### Wrong Observations

**Symptoms:**
- Network input shape mismatches
- Features are NaN or inf
- Graph has missing nodes/edges

**Diagnostics:**

```python
# Inspect observation structure
obs, info = env.reset()

if isinstance(obs, dict) and 'graph' in obs:
    graph = obs['graph']
    print(f"Node types: {graph.node_types}")
    for node_type in graph.node_types:
        print(f"  {node_type}: {graph[node_type].x.shape}")
    print(f"Edge types: {graph.edge_types}")
else:
    print(f"Observation type: {type(obs)}")
    print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

# Check for invalid values
if torch.isnan(obs['graph'].x_dict['place']).any():
    print("✗ NaN values in observations!")
if torch.isinf(obs['graph'].x_dict['place']).any():
    print("✗ Inf values in observations!")
```

**Solutions:**

1. **Ensure observation processor is set correctly**
   ```python
   env = GymProblem()
   metadata = env.make_metadata()
   print(f"Expected input: {metadata['features']}")
   
   # Network input should match
   policy_net = GNNPolicyNetwork(
       input_dim=metadata['features'],  # ✓ Must match
   )
   ```

2. **Normalize observations**
   ```python
   class NormalizedEnv:
       def __init__(self, env):
           self.env = env
           self.obs_mean = None
           self.obs_std = None
       
       def step(self, action):
           obs, reward, done, truncated, info = self.env.step(action)
           return self._normalize(obs), reward, done, truncated, info
       
       def _normalize(self, obs):
           # Normalize features to zero mean, unit variance
           if self.obs_mean is None:
               # Running statistics
               self.obs_mean = 0
               self.obs_std = 1
           
           obs['graph'].x_dict['place'] = (
               (obs['graph'].x_dict['place'] - self.obs_mean) / 
               (self.obs_std + 1e-8)
           )
           return obs
   ```

3. **Check variable initialization**
   ```python
   # Make sure places start with tokens if needed
   place = add_var("waiting")
   place.put({'value': 0})  # Initialize with default token
   ```

---

### Policy Issues

#### Policy Converges to Suboptimal Solution

**Symptoms:**
- Agent reaches ~50% of best reward
- Stagnates and won't improve further
- Different random seeds give different results

**Solutions:**

1. **Increase exploration**
   ```python
   ent_bonus=0.02,  # was 0.005
   ```

2. **Restart training with different seed**
   ```python
   import numpy as np
   import torch
   
   np.random.seed(42)
   torch.manual_seed(42)
   
   # Run multiple seeds
   results = []
   for seed in [1, 2, 3, 4, 5]:
       torch.manual_seed(seed)
       results.append(train_agent(env))
   
   best = max(results)
   ```

3. **Increase training duration**
   ```python
   epochs=200,  # was 100
   ```

4. **Use curriculum learning**
   ```python
   # Start with simpler rewards, gradually make harder
   epoch_rewards = []
   
   for epoch in range(100):
       # Difficulty increases with epoch
       difficulty = min(1.0, epoch / 50)
       agent.train(
           env,
           episodes=64,
           reward_scale=difficulty
       )
   ```

5. **Try DCL instead of PPO**
   ```python
   # DCL's planning may find better solutions
   agent = DCLAgent(
       policy_network=policy_net,
       value_network=value_net,
       planner_cfg=PlannerConfig(horizon=5, rollouts_per_action=32)
   )
   ```

#### Policy Overfits (High Train, Low Test)

**Symptoms:**
- Training return: 25
- Test return: 15
- Large train-test gap

**Solutions:**

1. **Add entropy bonus**
   ```python
   ent_bonus=0.02,
   ```

2. **Reduce network capacity**
   ```python
   policy_net = GNNPolicyNetwork(
       hidden_dim=64,  # was 128
   )
   ```

3. **Use early stopping**
   ```python
   agent.train(
       env,
       test_env=env,
       test_freq=10,
       early_stop_patience=20  # Stop if no improvement for 20 epochs
   )
   ```

4. **Add dropout**
   ```python
   policy_net = GNNPolicyNetwork(
       hidden_dim=128,
       dropout=0.2  # Enable dropout
   )
   ```

5. **Test with stochastic policy**
   ```python
   # Don't always use argmax
   action_probs = policy_net(obs)
   action = torch.multinomial(action_probs, 1)  # Sample instead of argmax
   ```

---

### Graph Issues

#### Graph Structure Warnings

**Symptoms:**
- "Heterograph has isolated nodes"
- "Edge index out of bounds"
- Graph connectivity warnings

**Diagnostics:**

```python
from torch_geometric.utils import degree

def check_graph_health(graph):
    """Diagnose graph structure issues."""
    
    for node_type in graph.node_types:
        x = graph[node_type].x
        print(f"{node_type}: {x.shape[0]} nodes, {x.shape[1]} features")
    
    for edge_type in graph.edge_types:
        edges = graph[edge_type].edge_index
        print(f"{edge_type}: {edges.shape[1]} edges")
        
        # Check for out-of-bounds indices
        src_nodes = edges[0]
        dst_nodes = edges[1]
        
        src_type, dst_type = edge_type[0], edge_type[2]
        n_src = graph[src_type].num_nodes
        n_dst = graph[dst_type].num_nodes
        
        if (src_nodes >= n_src).any():
            print(f"  ✗ Source node out of bounds!")
        if (dst_nodes >= n_dst).any():
            print(f"  ✗ Destination node out of bounds!")

# Run diagnostic
obs, info = env.reset()
check_graph_health(obs['graph'])
```

**Solutions:**

1. **Ensure all node types are created**
   ```python
   # Verify graph has required node types
   required = ['place', 'a_transition', 'e_transition']
   for node_type in required:
       if node_type not in obs['graph'].node_types:
           print(f"✗ Missing node type: {node_type}")
   ```

2. **Check edge index validity**
   ```python
   # All edge indices must point to valid nodes
   for edge_type, edges in graph.edge_items():
       if edges.edge_index.max() >= graph[edge_type[0]].num_nodes:
           print(f"✗ Invalid edge index in {edge_type}")
   ```

3. **Handle heterogeneous graphs carefully**
   ```python
   # Some operations don't work on hetero graphs
   # Use .to_homogeneous() if needed
   homo_graph = hetero_graph.to_homogeneous()
   ```

---

### Logging and Debugging

#### No Logs Being Generated

**Solutions:**

```python
# Ensure logging is initialized
from gympn.logging_utils import setup_logging, get_logger

setup_logging(
    logdir='data/train',
    level='INFO',
    use_tensorboard=True,
    use_wandb=False
)

logger = get_logger('training')
logger.info("This should appear!")
```

#### Logging to Multiple Destinations

```python
import logging

# Log to file and console
logger = logging.getLogger('gympn')
logger.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler('training.log')
fh.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
```

#### Profiling Training Speed

```python
import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()

# Your training code here
agent.train(env, episodes=64, epochs=10)

pr.disable()

# Print statistics
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(20)  # Top 20 functions
print(s.getvalue())
```

---

## Getting Help

If issues persist:

1. **Check the logs** for detailed error messages
2. **Run diagnostics** using code snippets above
3. **Simplify the problem** to identify root cause
4. **Consult examples** for similar problem types
5. **Open an issue** with reproducible example


