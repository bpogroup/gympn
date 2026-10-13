# Changelog

All notable changes to gympn are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## Unreleased

## 0.1.0 - 2026-10-09

First release on PyPI.

### Added

- `GymProblem`: define a problem as an Action-Evolution Petri Net on top of
  `simpn`, with actions, events, guards, reward functions and tagged places.
- `AEPN_Env`: the Gymnasium environment built from a `GymProblem`, with
  heterogeneous-graph observations for PyTorch Geometric.
- `PPOAgent` and `PGAgent`, with PPO-Clip, PPO-Penalty and policy gradient,
  and optional net-factored credit assignment (NF-GAE) for nets with
  independent parts.
- `HeteroActor` and `HeteroCritic` graph policy and value networks.
- Postponement: the agent may wait, globally or per independent component.
- `GymSolver`, `HeuristicSolver` and `RandomSolver` for testing a trained
  policy, a hand-written heuristic or a random policy, and a `Visualisation`
  of a run.
- `seed_everything` and `seed_network_init` for reproducible runs.
- Optional extras: `tensorboard` (training curves), `wandb` (Weights & Biases
  logging) and `viz` (plotting the graph observations). The library trains and
  tests without any of them.
- Documentation site at <https://bpogroup.github.io/gympn> with an API
  reference generated from the docstrings.

### Notes

- Requires `simpn` 1.3 to 1.6. Later simpn releases depend on PyQt6, which
  cannot be imported on a headless Linux machine, and 1.8+ cannot deep-copy a
  problem.
