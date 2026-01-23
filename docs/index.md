# gympn

Welcome to the official documentation for `gympn`, a Python library for decision-making in Petri Nets.

## Overview

GymPN is a Python library designed for creating and training reinforcement learning (RL) agents in environments based on Action-Evolution Petri Nets (AEPN). It provides tools for defining simulation problems, environments, and agents, making it easier to experiment with RL algorithms in Petri Net-based systems.

- **Action-Evolution Petri Nets (AEPN):** Define and simulate A-E PN environments.
- **Customizable RL Agents:** Train agents using Proximal Policy Optimization (PPO).
- **Graph Observations:** Generate graph-based observations for RL agents using PyTorch Geometric.
- **Integration with Gymnasium:** Seamless integration with Gym environments.
- **Flexible Simulation Framework:** Define custom events, actions, and reward functions.

## Installation

To install the library, clone the repository and install the dependencies:

```bash
git clone https://github.com/bpogroup/gympn.git
cd gympn
pip install -r requirements.txt
```

## Documentation

### Start Here
- **[Documentation Overview](./DOCUMENTATION_OVERVIEW.md)** ⭐ **NEW** - Quick guide to find what you need based on your learning style and goals

### Getting Started
- **[Gentle Introduction](./gentle_introduction.md)** - Start here! Learn basic concepts and create your first environment
- **[Complete Example](./complete_example.md)** - Full end-to-end example with all components

### Core Concepts
- **[Algorithms](./algorithms.md)** - Overview of PPO, PPO-Clip, PPO-Penalty, PG, and DCL algorithms with hyperparameter tuning
- **[Agents Guide](./agents_guide.md)** - Detailed guide on PPOAgent, DCLAgent, and custom solvers
- **[Advanced Features](./advanced_features.md)** - GNNs, causal tracking, Deep Causal Learning, parallelization

### Advanced Topics
- **[Extending GymPN](./extending_gympn.md)** - Create custom networks, solvers, environments, and reward functions
- **[Causal RL and Postponement](./postpone_causal.md)** - Using causal rewards with strategic postponement
- **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions

### Reference
- **[API Quick Reference](./api_reference.md)** - Quick lookup for classes and functions
- **[Benchmarks](./benchmarks.md)** - Performance comparison and recommendations

### Contributing
- **[Contributing Guide](./contributing.md)** - How to contribute to GymPN
