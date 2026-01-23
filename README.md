# GymPN Library

GymPN is a Python library designed for creating and training reinforcement learning (RL) agents in environments based on Action-Evolution Petri Nets (AEPN). It provides tools for defining simulation problems, environments, and agents, making it easier to experiment with RL algorithms in Petri Net-based systems.

## Features

- **Action-Evolution Petri Nets (AEPN):** Define and simulate A-E PN environments.
- **Customizable RL Agents:** Train agents using Proximal Policy Optimization (PPO) and Deep Causal Learning (DCL).
- **Graph Observations:** Generate graph-based observations for RL agents using PyTorch Geometric.
- **Causal Reinforcement Learning:** Track reward origins and assign credit through causal chains.
- **Strategic Postponement:** Defer decisions in complex temporal scenarios.
- **Integration with Gymnasium:** Seamless integration with Gym environments.
- **Flexible Simulation Framework:** Define custom events, actions, and reward functions.

## Quick Start

### Installation

To install the library and run a basic example, first clone the repository and install the dependencies (**important**: make sure you have Python 3.12 or higher):

```bash
git clone https://github.com/bpogroup/gympn.git
cd gympn
pip install -r requirements.txt
```

Then you can run a basic example to see how the library works:

```bash
python examples/example_minimal_task_assignment.py
```

### Learn GymPN

The complete documentation is available in the [docs directory](./docs/). Here's where to start:

1. **First time?** → Read the [Gentle Introduction](./docs/gentle_introduction.md) (30 minutes)
2. **Want a full example?** → See the [Complete Example](./docs/complete_example.md) (40 minutes)
3. **Need quick answers?** → Use the [Documentation Overview](./docs/DOCUMENTATION_OVERVIEW.md) to find what you need
4. **Prefer learning by doing?** → Check the [examples directory](./examples)
5. **Want to understand algorithms?** → Read the [Algorithms Guide](./docs/algorithms.md)
6. **Working with complex scenarios?** → See [Postponement and Causal RL](./docs/postpone_causal.md)

## Documentation

Complete documentation is available in the [docs directory](./docs/):

### Core Guides
- **[Gentle Introduction](./docs/gentle_introduction.md)** - Learn basic concepts and build your first environment
- **[Complete Example](./docs/complete_example.md)** - Full end-to-end example with all components
- **[Algorithms Guide](./docs/algorithms.md)** - Understand PPO, DCL, and other algorithms
- **[Agents Guide](./docs/agents_guide.md)** - Choose and configure the right agent

### Advanced Topics
- **[Advanced Features](./docs/advanced_features.md)** - GNNs, causal tracking, and optimization
- **[Postponement and Causal RL](./docs/postpone_causal.md)** - Strategic postponement and credit assignment
- **[Extending GymPN](./docs/extending_gympn.md)** - Create custom environments and solvers

### Reference
- **[API Quick Reference](./docs/api_reference.md)** - Quick lookup for classes and functions
- **[Benchmarks](./docs/benchmarks.md)** - Algorithm performance and recommendations
- **[Troubleshooting](./docs/troubleshooting.md)** - Common issues and solutions

### Navigation
- **[Documentation Overview](./docs/DOCUMENTATION_OVERVIEW.md)** - Guide to find the right documentation for your needs

## Examples

Several examples are available in the [examples directory](./examples):

- `example_minimal_task_assignment.py` - Minimal working example
- `example_complete_business_process.py` - Full business process example
- `example_simple_postpone.py` - Postponement feature example
- `SCG_extended_train.py` - Supply chain example with extended training
- And more!

## Contributing

Contributions are welcome! Please see the [Contributing Guide](./docs/contributing.md) for details on how to contribute to GymPN.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

