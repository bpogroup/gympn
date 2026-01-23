# Extending GymPN

This guide explains how to extend GymPN with custom components for your specific problems.

## Creating Custom Network Architectures

### Extending Base Network Classes

```python
import torch
from gympn.networks import GNNPolicyNetwork

class AttentionPolicyNetwork(GNNPolicyNetwork):
    """Policy network with attention mechanism."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__(input_dim, hidden_dim, output_dim)
        
        # Replace GCN with attention layers
        self.attention = torch.nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
    
    def forward(self, data):
        # Get base GNN features
        x = super().forward(data)
        
        # Apply attention
        attn_out, _ = self.attention(x, x, x)
        
        return attn_out
```

### Custom Feature Engineering

```python
class EnhancedGNNPolicy(torch.nn.Module):
    """Policy with custom feature extraction."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Domain-specific feature extractor
        self.feature_extractor = FeatureExtractor(input_dim)
        
        # Standard GNN
        self.gnn = torch.nn.ModuleList([
            # ... GNN layers ...
        ])
        
        self.head = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        # Extract features
        features = self.feature_extractor(data)
        
        # Process with GNN
        for layer in self.gnn:
            features = layer(features)
        
        # Output
        return self.head(features)

class FeatureExtractor(torch.nn.Module):
    def forward(self, data):
        # Extract queue lengths
        queue_features = extract_queue_lengths(data)
        
        # Extract resource utilization
        utilization = extract_resource_utilization(data)
        
        # Extract temporal features
        temporal = extract_temporal_features(data)
        
        return torch.cat([queue_features, utilization, temporal], dim=1)
```

## Creating Custom Solvers

### Implementing a Domain-Specific Solver

```python
from gympn.solvers import Solver

class LoadBalancingSolver(Solver):
    """
    Balances load across resources.
    
    Tries to assign tasks to the least-busy resource.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.max_utilization = {}
    
    def solve(self, observable_net, tokens_comb):
        """Load-balancing strategy."""
        
        # Get resource utilizations
        utilization = self._compute_utilization(observable_net)
        
        # Find best binding (least utilized resource)
        best_action = None
        best_binding = None
        best_score = float('inf')
        
        for action_id, bindings in tokens_comb.items():
            for binding in bindings:
                # Extract resource from binding
                resource = binding[1][1].value  # Second token is resource
                resource_id = resource['code_employee']
                
                # Score is current utilization (lower is better)
                score = utilization.get(resource_id, 0)
                
                if score < best_score:
                    best_score = score
                    best_action = action_id
                    best_binding = binding
        
        if best_binding is not None:
            return {best_action: best_binding}
        
        return 'postpone'
    
    def _compute_utilization(self, observable_net):
        """Compute how busy each resource is."""
        utilization = {}
        
        # Count busy tokens per resource
        for variable, tokens in observable_net.marking.items():
            if variable == 'busy':
                for token in tokens:
                    resource_id = token.value[1]['code_employee']
                    utilization[resource_id] = utilization.get(resource_id, 0) + 1
        
        return utilization
```

### Solver with Learning

```python
class AdaptiveSolver(Solver):
    """Solver that learns preferences over time."""
    
    def __init__(self, params=None):
        super().__init__(params)
        self.action_returns = {}  # Track average return per action
        self.action_counts = {}
    
    def solve(self, observable_net, tokens_comb):
        """Epsilon-greedy action selection based on learning."""
        
        epsilon = 0.1  # Exploration rate
        
        if np.random.random() < epsilon:
            # Explore: random action
            return self._random_action(tokens_comb)
        else:
            # Exploit: best learned action
            return self._best_action(tokens_comb)
    
    def update(self, action_id, reward):
        """Called after episode to update learning."""
        if action_id not in self.action_returns:
            self.action_returns[action_id] = 0
            self.action_counts[action_id] = 0
        
        # Running average
        n = self.action_counts[action_id]
        current_avg = self.action_returns[action_id]
        self.action_returns[action_id] = (
            (n * current_avg + reward) / (n + 1)
        )
        self.action_counts[action_id] += 1
    
    def _best_action(self, tokens_comb):
        """Return action with highest average return."""
        best_action = None
        best_return = -float('inf')
        
        for action_id in tokens_comb.keys():
            avg_return = self.action_returns.get(action_id, 0)
            if avg_return > best_return:
                best_return = avg_return
                best_action = action_id
        
        if best_action is not None:
            return {best_action: tokens_comb[best_action][0]}
        
        return 'postpone'
    
    def _random_action(self, tokens_comb):
        """Select random action."""
        action_id = list(tokens_comb.keys())[0]
        binding = np.random.choice(tokens_comb[action_id])
        return {action_id: binding}
```

## Creating Custom Environments

### Extending GymProblem

```python
from gympn.simulator import GymProblem
from simpn.simulator import SimToken

class ManufacturingProblem(GymProblem):
    """Custom problem: Job shop scheduling."""
    
    def __init__(self, num_machines=3, num_job_types=2):
        super().__init__(allow_postpone=True, causal_rl=True)
        
        self.num_machines = num_machines
        self.num_job_types = num_job_types
        
        self._create_environment()
    
    def _create_environment(self):
        """Define places, transitions, events, and actions."""
        
        # Places
        self.arrival = self.add_var(
            "arrival",
            var_attributes=['job_type', 'urgency']
        )
        self.waiting = self.add_var(
            "waiting",
            var_attributes=['job_type', 'urgency']
        )
        self.processing = self.add_var(
            "processing",
            var_attributes=['job_type', 'machine_id', 'urgency']
        )
        
        self.machine = self.add_var(
            "machine",
            var_attributes=['machine_id', 'speed']
        )
        
        # Initialize machines
        for i in range(self.num_machines):
            self.machine.put({
                'machine_id': i,
                'speed': 1.0 + 0.5 * i  # Slower machines
            })
        
        # Events
        def arrive(a):
            return [SimToken(a, delay=1), SimToken(a)]
        
        self.add_event(
            [self.arrival],
            [self.arrival, self.waiting],
            arrive
        )
        
        # Actions
        def start(job, machine):
            processing_time = 1.0 / machine['speed']
            return [SimToken((job, machine), delay=processing_time)]
        
        self.add_action(
            [self.waiting, self.machine],
            [self.processing],
            behavior=start,
            name="assign"
        )
        
        # Completion
        def complete(job_machine):
            job, machine = job_machine
            reward = self._compute_reward(job, machine)
            return [SimToken(machine[1])]
        
        self.add_event(
            [self.processing],
            [self.machine],
            complete,
            name='complete',
            reward_function=lambda x: 1.0  # Unit reward per job
        )
    
    def _compute_reward(self, job, machine):
        """Compute reward based on job characteristics."""
        if job['urgency'] > 0.5:
            return 2.0  # High priority gets bonus
        return 1.0
    
    def initialize(self):
        """Reset environment with random job arrivals."""
        super().initialize()
        
        # Add initial jobs
        for _ in range(3):
            job_type = np.random.randint(0, self.num_job_types)
            urgency = np.random.random()
            self.arrival.put({
                'job_type': job_type,
                'urgency': urgency
            })
```

## Creating Custom Reward Functions

### Dynamic Rewards Based on State

```python
class DynamicRewardFunction:
    """Reward function that changes based on environment state."""
    
    def __init__(self, env):
        self.env = env
        self.episode_start_time = None
        self.max_episode_length = 100
    
    def __call__(self, token_binding):
        """Compute reward dynamically."""
        
        # Get current time
        current_time = self.env.current_time
        
        # Time penalty: encourage finishing early
        time_penalty = -0.01 * current_time
        
        # Completion bonus
        completion_bonus = 1.0
        
        # Efficiency bonus: prefer faster machine
        job, machine = token_binding
        if machine['speed'] > 1.0:
            efficiency_bonus = 0.5
        else:
            efficiency_bonus = 0.0
        
        return completion_bonus + time_penalty + efficiency_bonus
```

### Hierarchical Rewards

```python
class HierarchicalRewardManager:
    """Multi-level reward structure."""
    
    def __init__(self):
        self.rewards = {
            'completion': {},      # Job completion rewards
            'efficiency': {},      # Efficiency metrics
            'fairness': {},        # Load balancing rewards
        }
    
    def compute_completion_reward(self, job, machine):
        """Reward for completing jobs."""
        base = 1.0
        if job['priority'] == 'high':
            return base * 2.0
        return base
    
    def compute_efficiency_reward(self, machine, completion_time):
        """Reward for efficient processing."""
        expected_time = 1.0 / machine['speed']
        if completion_time < expected_time:
            return 0.5  # Bonus for ahead of schedule
        elif completion_time > expected_time * 1.5:
            return -0.5  # Penalty for far behind
        return 0.0
    
    def compute_fairness_reward(self, machine_utilizations):
        """Reward for balanced load."""
        variance = np.var(list(machine_utilizations.values()))
        if variance < 0.5:
            return 0.2  # Bonus for balanced load
        return -0.1
    
    def total_reward(self, job, machine, completion_time, utilizations):
        """Combine all reward components."""
        return (
            self.compute_completion_reward(job, machine) +
            self.compute_efficiency_reward(machine, completion_time) +
            self.compute_fairness_reward(utilizations)
        )
```

## Creating Custom Heuristics

### Rule-Based Heuristic

```python
def rule_based_heuristic(observable_net, tokens_comb):
    """
    Heuristic based on domain-specific rules.
    
    Rules:
    1. High-priority jobs should use fast machines
    2. Avoid overloading any single machine
    3. Postpone if no good match available
    """
    
    # Rule 1: Identify high-priority jobs
    best_action = None
    best_binding = None
    best_score = -float('inf')
    
    for action_id, bindings in tokens_comb.items():
        for binding in bindings:
            job, machine = binding
            
            # Score based on priority and machine speed
            priority_bonus = 10.0 if job['urgency'] > 0.5 else 0.0
            speed_bonus = machine['speed']
            
            # Penalty for overloaded machines
            overload_penalty = count_busy_on_machine(observable_net, machine['id']) * 2.0
            
            score = priority_bonus + speed_bonus - overload_penalty
            
            if score > best_score:
                best_score = score
                best_action = action_id
                best_binding = binding
    
    if best_binding is not None and best_score > 0:
        return {best_action: best_binding}
    
    return 'postpone'
```

### Learned + Heuristic Hybrid

```python
def hybrid_solver(observable_net, tokens_comb, learned_policy_net):
    """
    Combines learned policy with hand-crafted heuristics.
    
    When heuristic is confident, use it. Otherwise defer to learned policy.
    """
    
    # Get heuristic's suggestion
    heuristic_action = rule_based_heuristic(observable_net, tokens_comb)
    
    # Get learned policy's suggestion
    graph_obs = make_graph_observation(observable_net)
    learned_logits = learned_policy_net(graph_obs)
    learned_action = argmax(learned_logits)
    
    # Compute heuristic confidence
    if has_high_priority_job(observable_net):
        # When stakes are high, trust heuristic
        return heuristic_action
    else:
        # Otherwise, use learned policy
        return learned_action

def count_busy_on_machine(observable_net, machine_id):
    """Count how many jobs are currently on this machine."""
    count = 0
    for job_machine in observable_net.marking['processing']:
        if job_machine[1]['machine_id'] == machine_id:
            count += 1
    return count

def has_high_priority_job(observable_net):
    """Check if any high-priority job is waiting."""
    for job in observable_net.marking['waiting']:
        if job['urgency'] > 0.7:
            return True
    return False
```

## Testing Custom Components

### Unit Testing Network

```python
import unittest
import torch

class TestCustomNetwork(unittest.TestCase):
    
    def setUp(self):
        self.network = AttentionPolicyNetwork(64, 128, 10)
    
    def test_forward_pass(self):
        """Test that network processes graphs correctly."""
        # Create dummy graph
        from torch_geometric.data import HeteroData
        
        data = HeteroData()
        data['place'].x = torch.randn(5, 64)
        data['a_transition'].x = torch.randn(3, 64)
        
        # Forward pass
        output = self.network(data)
        
        # Check output shape and values
        self.assertEqual(output.shape, (3, 10))
        self.assertFalse(torch.isnan(output).any())
    
    def test_gradient_flow(self):
        """Test that gradients flow through network."""
        data = HeteroData()
        data['place'].x = torch.randn(5, 64, requires_grad=True)
        data['a_transition'].x = torch.randn(3, 64, requires_grad=True)
        
        output = self.network(data)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        for param in self.network.parameters():
            self.assertIsNotNone(param.grad)
```

### Integration Testing Solver

```python
class TestLoadBalancingSolver(unittest.TestCase):
    
    def setUp(self):
        self.env = ManufacturingProblem(num_machines=3)
        self.solver = LoadBalancingSolver()
    
    def test_load_balancing(self):
        """Test that solver balances load across machines."""
        self.env.initialize()
        
        # Simulate many steps
        returns = []
        for _ in range(10):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.solver.solve(obs, get_tokens_comb(self.env))
                obs, reward, done, _, _ = self.env.step(action)
                total_reward += reward
            
            returns.append(total_reward)
        
        # Check performance is reasonable
        mean_return = np.mean(returns)
        self.assertGreater(mean_return, 5.0)
    
    def test_no_crashes(self):
        """Test that solver doesn't crash on edge cases."""
        # Empty tokens
        action = self.solver.solve(self.env.observable_net, {})
        self.assertEqual(action, 'postpone')
        
        # Single action
        tokens_comb = {0: [[('a', {}), ('b', {})]]}
        action = self.solver.solve(self.env.observable_net, tokens_comb)
        self.assertIsNotNone(action)
```

## Documenting Your Extensions

When creating custom components, document them properly:

```python
class MyCustomNetwork(torch.nn.Module):
    """
    Custom network architecture for specialized problems.
    
    This network combines graph convolution with attention mechanisms
    to handle complex dependencies in manufacturing environments.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (typically number of actions)
        num_heads (int): Number of attention heads (default: 4)
    
    Example:
        >>> net = MyCustomNetwork(64, 128, 10, num_heads=4)
        >>> output = net(graph_observation)
        >>> output.shape
        torch.Size([10])
    
    Notes:
        - Requires PyTorch Geometric for graph processing
        - Expects HeteroData format for observations
        - Suitable for problems with high-order dependencies
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        # Implementation...
        pass
    
    def forward(self, data):
        # Implementation...
        pass
```


