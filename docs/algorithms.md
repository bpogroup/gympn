# Training Algorithms in GymPN

## Overview

GymPN provides several reinforcement learning algorithms for training agents in Action-Evolution Petri Net environments. Each algorithm has different strengths, making them suitable for different problem characteristics.

## Available Algorithms

### 1. PPO (Proximal Policy Optimization)

**What it is:** A state-of-the-art policy gradient method that balances exploration and exploitation through clipping mechanisms.

**Two variants:**

#### PPO-Clip
Uses the clipping mechanism from the original PPO paper to constrain policy updates.

```python
config = {
    "algorithm": "ppo-clip",
    "eps": 0.15,              # Clipping range (0.1 to 0.2 typical)
    "policy_lr": 8e-4,        # Policy learning rate
    "policy_updates": 5,      # Updates per epoch
    "value_lr": 8e-4,         # Value network learning rate
    "value_updates": 10,      # Value updates per epoch
    "ent_bonus": 0.005,       # Entropy bonus for exploration
    "gam": 1.0,               # Discount factor
    "lam": 0.99,              # GAE lambda
}
```

**Best for:** General-purpose environments, fast training, stable convergence

**Speed:** Fast (baseline)

**Hyperparameters to tune:**
- `eps`: Larger values allow bigger policy changes (more exploration), smaller values are more conservative
- `policy_updates`: More updates lead to slower but potentially better convergence
- `ent_bonus`: Higher values encourage exploration, lower values encourage exploitation

#### PPO-Penalty
Uses a penalty term instead of clipping to constrain policy divergence.

```python
config = {
    "algorithm": "ppo-penalty",
    "c": 0.2,                 # Penalty coefficient
    "policy_kld_limit": 0.1,  # Maximum KL divergence allowed
    # ... other hyperparameters
}
```

**Best for:** When you want tighter control over policy divergence

**Speed:** Similar to PPO-Clip

**Key difference:** Better KL divergence control but slightly less stable

### 2. PG (Policy Gradient)

**What it is:** The classical REINFORCE algorithm with baseline (value function).

```python
config = {
    "algorithm": "pg",
    "policy_lr": 5e-4,
    "value_lr": 5e-4,
}
```

**Best for:** Simple problems, understanding baselines

**Speed:** Fastest

**Limitations:** Less stable than PPO, prone to high variance

## Algorithm Comparison

| Aspect | PPO-Clip | PPO-Penalty | PG |
|--------|----------|-------------|----|
| **Speed** | Fast | Fast | Fastest |
| **Stability** | Very High | High | Medium |
| **Sample Efficiency** | High | High | Low |
| **Complexity** | Medium | Medium | Low |
| **Best For** | General use | Strict KL control | Learning |
| **NF-GAE** | Compatible | Compatible | Compatible |
| **Postponement** | Compatible | Compatible | Compatible |

## Choosing an Algorithm

### Quick Decision Tree

```
Is training speed critical?
├─ YES: Use PPO-Clip
└─ NO:
    Do you need strict KL control?
    ├─ YES: Use PPO-Penalty
    └─ NO: Use PPO-Clip (default)
```

### Specific Scenarios

**Task Assignment (Simple)**
- Start with PPO-Clip
- Config: episodes=64, epochs=100, policy_lr=8e-4

**Business Process Optimization**
- Start with PPO-Clip
- Several independent processes in one net: add nfgae=True (see Postponement and Net-Factored Credit)

**Testing/Debugging**
- Use PG (simplest)
- Use small episodes and epochs
- Fast feedback for algorithm development

## Hyperparameter Tuning Guide

### Learning Rates
- **Policy LR:** Controls policy update magnitude
  - Too high: Divergence, instability
  - Too low: Slow learning
  - Start: 5e-4 to 8e-4
  - Adjust: -50% if diverging, +50% if learning is slow

- **Value LR:** Separate control for value function
  - Often set equal to policy_lr
  - Can be slightly higher (1.5x policy_lr) for better value estimates
  - Start: 5e-4 to 8e-4

### PPO-Specific

- **eps (Clipping Range):** 0.1-0.2
  - Higher: More aggressive updates, higher variance
  - Lower: Conservative updates, slower learning
  - Default: 0.15

- **policy_updates:** 3-10
  - Higher: Better optimization of each batch, slower training
  - Lower: Faster training, less stable
  - Default: 5

- **value_updates:** 5-20
  - Higher: Better value estimates, more computational cost
  - Lower: Faster training, potentially worse value estimates
  - Default: 10

### Entropy Bonus

```python
"ent_bonus": 0.001  # PPO and PG
```

- **Higher (0.01-0.05):** More exploration, may prevent convergence
- **Lower (0.001-0.005):** More exploitation, may get stuck in local optima
- **Rule of thumb:** Start with 0.005, adjust based on policy behavior

## Training Configuration Examples

### Template 1: Fast Training (PPO-Clip)
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

### Template 2: Stable Training (PPO-Clip)
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

### Template 3: Net-Factored Credit (PPO-Clip + NF-GAE)
```python
config = {
    "algorithm": "ppo-clip",
    "episodes": 64,
    "epochs": 100,
    "batch_size": 64,
    "policy_lr": 8e-4,
    "value_lr": 8e-4,
    "policy_updates": 5,
    "value_updates": 10,
    "eps": 0.15,
    "ent_bonus": 0.005,
    "smdp_discount": True,
    "beta": 0.5,
    "nfgae": True,
}
```

## Algorithm Selection by Problem Type

### If you have:

**Several independent processes in one net**
→ Use PPO-Clip with nfgae=True

**Many small independent tasks**
→ Use PPO-Clip with high ent_bonus
→ Fast episodes=64, epochs=50

**Sensitive decision points**
→ Use PPO-Penalty with low c=0.1
→ Strict control over policy divergence

**Stochastic environment**
→ Use PPO-Clip with many episodes
→ episodes=128, epochs=200

**Resource-constrained (slow training needed)**
→ Use PG
→ Simplest algorithm, fastest per step
→ Accept slower convergence

## Convergence Monitoring

Watch these metrics to diagnose algorithm issues:

```
KL Divergence
├─ Increasing: Policy diverging too fast → Reduce policy_lr or increase eps
└─ Near zero: Not updating → Increase policy_lr

Policy Loss
├─ Oscillating wildly: Instability → Reduce policy_lr
├─ Slowly improving: Normal → Continue
└─ Plateauing: Stuck → Increase ent_bonus

Value Loss
├─ Very high: Value network behind → Increase value_updates
├─ Slowly improving: Normal → Continue
└─ Constant: Value not learning → Increase value_lr

Entropy
├─ Decreasing too fast: Premature convergence → Increase ent_bonus
└─ Staying high: Good exploration → Normal
```

## Performance Optimization

### If training is too slow:

1. **Reduce batch size** (but keep multiple of episodes)
2. **Reduce policy_updates** and **value_updates**
3. **Reduce episode count** (but maintain statistical quality)

### If training is unstable:

1. **Reduce policy_lr** by 50%
2. **Reduce eps** (PPO-Clip) to 0.1
3. **Increase policy_updates** for better optimization
4. **Add entropy bonus** (0.01)

### If not converging:

1. **Increase entropy bonus** to 0.01-0.02
2. **Increase policy_updates** to 8-10
3. **Use larger batch_size** for gradient stability
4. **Enable nfgae if the net has independent parts**


