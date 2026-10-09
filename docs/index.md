# gympn

Welcome to the official documentation for `gympn`, a Python library for decision-making in Petri Nets.

## Overview

GymPN is a Python library designed for creating and training reinforcement learning (RL) agents in environments based on Action-Evolution Petri Nets (AEPN). It provides tools for defining simulation problems, environments, and agents, making it easier to experiment with RL algorithms in Petri Net-based systems.

- **Action-Evolution Petri Nets (AEPN):** Define and simulate A-E PN environments.
- **Customizable RL Agents:** Train agents using Proximal Policy Optimization (PPO), with optional net-factored credit (NF-GAE) for nets with independent parts.
- **Postponement:** Let the agent wait, globally or per independent part of the net.
- **Graph Observations:** Generate graph-based observations for RL agents using PyTorch Geometric.
- **Integration with Gymnasium:** Seamless integration with Gym environments.
- **Flexible Simulation Framework:** Define custom events, actions, and reward functions.

## Installation

GymPN requires Python 3.10 or newer:

```bash
pip install gympn
```

Optional features are extras: `gympn[tensorboard]` for training curves in
TensorBoard, `gympn[wandb]` for Weights & Biases logging, `gympn[viz]` for
plotting the graph observations, and `gympn[all]` for all three. Training and
testing work without any of them.

PyTorch publishes CPU-only and CUDA-specific wheels on its own index; to get a
specific build, install `torch` first by following
<https://pytorch.org/get-started/locally/> and then run `pip install gympn`.

For development, clone the repository and install it in editable mode with
every extra (see [Contributing](./contributing.md)):

```bash
git clone https://github.com/bpogroup/gympn.git
cd gympn
pip install -e ".[dev]"
```

## Documentation

### Start Here
- **[Documentation Overview](./DOCUMENTATION_OVERVIEW.md)** ⭐ **NEW** - Quick guide to find what you need based on your learning style and goals

### Getting Started
- **[Gentle Introduction](./gentle_introduction.md)** - Start here! Learn basic concepts and create your first environment
- **[Complete Example](./complete_example.md)** - Full end-to-end example with all components

### Core Concepts
- **[Algorithms](./algorithms.md)** - Overview of PPO-Clip, PPO-Penalty and PG algorithms with hyperparameter tuning
- **[Agents Guide](./agents_guide.md)** - Detailed guide on PPOAgent and custom solvers
- **[Advanced Features](./advanced_features.md)** - GNNs, parallelization

### Advanced Topics
- **[Extending GymPN](./extending_gympn.md)** - Create custom networks, solvers, environments, and reward functions
- **[Postponement and Net-Factored Credit](./postponement.md)** - Strategic postponement and per-component credit
- **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions

### Reference
- **[API Quick Reference](./api_reference.md)** - Quick lookup for classes and functions
- **[Benchmarks](./benchmarks.md)** - Performance comparison and recommendations

### Contributing
- **[Contributing Guide](./contributing.md)** - How to contribute to GymPN
