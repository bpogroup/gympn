# GymPN Library

GymPN is a Python library designed for creating and training reinforcement learning (RL) agents in environments based on Action-Evolution Petri Nets (AEPN). It provides tools for defining simulation problems, environments, and agents, making it easier to experiment with RL algorithms in Petri Net-based systems.

## Features

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

## Examples
For examples of how to use GymPN, check out the [examples directory](./examples). You can find various scripts demonstrating how to create environments, define agents, and run simulations.
Alternatively, a gentle introduction to the library is available in the [gentle introduction](./gentle_introduction.md).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have ideas for improvements or new features.  

## License
This project is licensed under the MIT License. See the LICENSE file for details.