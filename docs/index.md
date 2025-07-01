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

## Quick Start
To get started with GymPN, check our [gentle introduction](./gentle_introduction.md).