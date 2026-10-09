# GymPN

GymPN is a Python library for creating and training reinforcement learning (RL) agents in environments based on Action-Evolution Petri Nets (A-E PN). It provides tools for defining simulation problems, environments, and agents, making it easier to experiment with RL algorithms on business processes and resource allocation problems modelled as Petri nets.

- **Documentation:** https://bpogroup.github.io/gympn
- **Source:** https://github.com/bpogroup/gympn
- **Issues:** https://github.com/bpogroup/gympn/issues

## Features

- **Action-Evolution Petri Nets (A-E PN):** Define and simulate A-E PN environments.
- **Customizable RL Agents:** Train agents using Proximal Policy Optimization (PPO), with optional net-factored credit (NF-GAE) for nets with independent parts.
- **Graph Observations:** Generate graph-based observations for RL agents using PyTorch Geometric.
- **Postponement:** Let the agent wait, globally or per independent part of the net.
- **Integration with Gymnasium:** The environment is a standard Gymnasium environment.
- **Flexible Simulation Framework:** Define custom events, actions, guards and reward functions.

## Installation

GymPN requires Python 3.10 or newer and is installed from PyPI:

```bash
pip install gympn
```

The core install pulls in `torch`, `torch-geometric`, `gymnasium` and `simpn`. PyTorch publishes CPU-only and CUDA-specific wheels on its own index; if `pip` picks a build you do not want, install PyTorch first by following https://pytorch.org/get-started/locally/ and then run `pip install gympn`.

Optional features are installed as extras:

| Extra | Adds | Enables |
| --- | --- | --- |
| `gympn[tensorboard]` | tensorboard | Training curves in TensorBoard (`open_tensorboard=True`) |
| `gympn[wandb]` | wandb | Weights & Biases logging (`use_wandb=True`) |
| `gympn[viz]` | matplotlib, networkx | Plotting the graph observations (`plot_observations=True`) |
| `gympn[all]` | all of the above | |

Training and testing work without any extra installed; a missing extra is reported with the `pip install` command that adds it.

For development (tests, linting, docs), clone the repository and install it in editable mode with every extra:

```bash
git clone https://github.com/bpogroup/gympn.git
cd gympn
pip install -e ".[dev]"
```

## Quick Start

The minimal example is a task assignment problem with two employees and two task types. Train a policy on it:

```python
from simpn.simulator import SimToken
from gympn import GymProblem, RandomSolver

agency = GymProblem()

# Places: tasks arrive, wait, and are processed by an employee.
arrival = agency.add_var("arrival", var_attributes=["task_type"])
waiting = agency.add_var("waiting", var_attributes=["task_type"])
busy = agency.add_var("busy", var_attributes=["task_type", "resource_id"])
employee = agency.add_var("employee", var_attributes=["code_employee"])
arrival.put({"task_type": 0})
arrival.put({"task_type": 1})
employee.put({"code_employee": 0})
employee.put({"code_employee": 1})

# Events evolve the net on their own; actions are the agent's decisions.
agency.add_event([arrival], [arrival, waiting], name="arrive",
                 behavior=lambda a: [SimToken(a, delay=1), SimToken(a)])
agency.add_action([waiting, employee], [busy], name="start",
                  behavior=lambda c, r: [SimToken((c, r), delay=1 if c["task_type"] == r["code_employee"] else 2)])
agency.add_event([busy], [employee], lambda b: [SimToken(b[1])], name="complete",
                 reward_function=lambda x: 1)

# Train a PPO agent; checkpoints go to data/train/<run name>/.
agency.training_run(length=10, args_dict={"epochs": 20, "episodes": 10, "logdir": "data/train", "name": "run"})

# Compare against a random policy on a fresh copy of the net.
import copy
print("random policy reward:", copy.deepcopy(agency).testing_run(RandomSolver(), length=10))
```

The [examples directory](https://github.com/bpogroup/gympn/tree/main/examples) contains complete scripts, including this one with a hand-written heuristic and a trained-policy evaluation (`example_minimal_task_assignment.py`; set `train = True` at the top to train before testing).

## Learn GymPN

1. **First time?** → Read the [Gentle Introduction](https://bpogroup.github.io/gympn/gentle_introduction/)
2. **Want a full example?** → See the [Complete Example](https://bpogroup.github.io/gympn/complete_example/)
3. **Need quick answers?** → Use the [Documentation Overview](https://bpogroup.github.io/gympn/DOCUMENTATION_OVERVIEW/)
4. **Want to understand algorithms?** → Read the [Algorithms Guide](https://bpogroup.github.io/gympn/algorithms/)
5. **Working with concurrent decisions?** → See [Postponement and Net-Factored Credit](https://bpogroup.github.io/gympn/postponement/)
6. **Looking up a class?** → Browse the [API Reference](https://bpogroup.github.io/gympn/reference/problem/)

Other guides: [Agents Guide](https://bpogroup.github.io/gympn/agents_guide/), [Advanced Features](https://bpogroup.github.io/gympn/advanced_features/), [Extending GymPN](https://bpogroup.github.io/gympn/extending_gympn/), [Benchmarks](https://bpogroup.github.io/gympn/benchmarks/), [Troubleshooting](https://bpogroup.github.io/gympn/troubleshooting/).

## Contributing

Contributions are welcome. See the [Contributing Guide](https://bpogroup.github.io/gympn/contributing/) for the development setup, the test suite and the release process.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
