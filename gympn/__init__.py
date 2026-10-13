"""
gympn: reinforcement learning on Action-Evolution Petri Nets.

Main entry points:

- GymProblem: define a problem as an Action-Evolution Petri Net and train or
  test agents on it.
- AEPN_Env: the Gymnasium environment built from a GymProblem.
- PPOAgent, PGAgent: the training agents (PPO, optionally with net-factored
  credit, NF-GAE).
- HeteroActor, HeteroCritic: the graph policy and value networks.
- GymSolver, HeuristicSolver, RandomSolver: solvers for testing a trained
  policy, a heuristic or a random policy.
"""

__version__ = "0.1.0"

from .networks import HeteroActor, HeteroCritic
from .environment import AEPN_Env
from .simulator import GymProblem
from .agents import PGAgent, PPOAgent
from .solvers import BaseSolver, GymSolver, HeuristicSolver, RandomSolver
from .seeding import seed_everything, seed_network_init

__all__ = [
    "__version__",
    "GymProblem", "AEPN_Env",
    "PPOAgent", "PGAgent",
    "HeteroActor", "HeteroCritic",
    "BaseSolver", "GymSolver", "HeuristicSolver", "RandomSolver",
    "seed_everything", "seed_network_init",
]
