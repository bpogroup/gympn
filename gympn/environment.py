
import copy
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from gymnasium import spaces, Env


class AEPN_Env(Env):
    """
    Gym environment for training a Deep Reinforcement Learning agent on the AEPN simulator.
    """

    def __init__(self, aepn):
        """"Initialize the environment with a GymProblem instance."""
        super().__init__()
        self.pn = aepn
        self.frozen_pn = copy.deepcopy(self.pn)
        self.metadata = None

        # gym specific
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({'graph': spaces.Box(low=0, high=1, shape=(1,))})

        # mimic the network's organization
        self.run: List[Any] = []
        self.i: int = 0
        self.active_model: bool = True

        # debugging
        self.debug = False

    def step(self, action: int):
        """
        Execute one step in the environment.
        Returns (observation, reward, terminated, truncated, info)
        """
        old_rewards = self.pn.reward
        if action < 0 or action >= len(self.pn.pn_actions):
            valid_len = len(self.pn.pn_actions) - 1
            raise ValueError(f"Action {action} is not valid. Must be between 0 and {valid_len}")

        # handle postpone
        if action == len(self.pn.pn_actions) - 1 and self.pn.pn_actions[-1][0] == ['postpone']:
            self.pn.postpone()
            self.pn.just_postponed = True
            if self.debug:
                print("Postpone!")
        else:
            binding = self.pn.pn_actions[action]
            self.pn.just_postponed = False
            if self.debug:
                print(f"Action {action}: {binding} at time {self.pn.clock}")

            result_tokens = self.pn.fire(binding)
            self.pn.update_reward(binding, result_tokens)
            self.pn.bindings()  # updates the network tag if needed

        observation, terminated, self.i = self.pn.run_evolutions(self.run, self.i, self.active_model)

        if terminated:
            if self.debug:
                print(f'Terminated at time {self.pn.clock}')
            if self.pn.causal_rl:
                info = {'pn_reward': self.pn.reward, 'eligibility_credits': self.pn.causal_trace}
            else:
                info = {'pn_reward': self.pn.reward}
        else:
            info = {'pn_reward': self.pn.reward}

        if not self.pn.causal_rl:
            reward = (self.pn.reward - old_rewards)
        else:
            reward = 0.0  # episode-level credit assignment via causal traces

        return observation, reward, terminated, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment to its initial state."""
        if seed is not None:
            self.set_seed(seed)

        if self.debug:
            print(f"Entered reset with current reward for PN: {self.pn.reward} \n")
        self.pn = copy.deepcopy(self.frozen_pn)
        # ensure per-episode action index alignment with buffer indices
        self.pn._action_index = 0
        if self.pn.network_tag.is_evolution():
            self.pn.get_to_first_action()

        observation = self.pn.get_graph_observation()
        if self.metadata is None:
            self.metadata = self.pn.metadata
        return observation

    def render(self):
        """Render the current state of the environment (not implemented)."""
        pass

    def fork(self) -> "AEPN_Env":
        """
        Create a deep copy of the environment for parallel simulations.
        Prefer get_state()/set_state() for cheaper rollouts; fork() is a safe fallback.
        """
        return copy.deepcopy(self)

    # ------------------ NEW: State snapshot & RNG control ------------------

    def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of the full simulator + RNG states.
        This allows cheap replication via set_state() without deep-copying the whole env repeatedly.

        Returns
        -------
        snapshot : dict
            {
              'pn': deepcopy(self.pn),
              'run': deepcopy(self.run),
              'i': self.i,
              'active_model': self.active_model,
              'rng': {
                  'python': random.getstate(),
                  'numpy':  np.random.get_state(),
                  'torch_cpu': torch.get_rng_state(),
                  'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
              }
            }
        """
        snapshot = {
            "pn": copy.deepcopy(self.pn),
            "run": copy.deepcopy(self.run),
            "i": self.i,
            "active_model": self.active_model,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        return snapshot

    def set_state(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore a previously captured simulator + RNG state.

        Parameters
        ----------
        snapshot : dict
            A dictionary produced by get_state().
        """
        # Simulator state
        self.pn = copy.deepcopy(snapshot["pn"])
        self.run = copy.deepcopy(snapshot["run"])
        self.i = int(snapshot["i"])
        self.active_model = bool(snapshot["active_model"])

        # RNG states (best effort; guard against missing keys/devices)
        rng = snapshot.get("rng", {})
        try:
            py_state = rng.get("python", None)
            if py_state is not None:
                random.setstate(py_state)
        except Exception:
            pass

        try:
            np_state = rng.get("numpy", None)
            if np_state is not None:
                np.random.set_state(np_state)
        except Exception:
            pass

        try:
            cpu_state = rng.get("torch_cpu", None)
            if isinstance(cpu_state, torch.ByteTensor) or cpu_state is not None:
                torch.set_rng_state(cpu_state)
        except Exception:
            pass

        try:
            cuda_states = rng.get("torch_cuda", None)
            if cuda_states is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass

    def set_seed(self, seed: int) -> None:
        """
        Seed *all* stochastic elements (Python, NumPy, PyTorch CPU/GPU, and Petri net if available).

        Parameters
        ----------
        seed : int
            The seed to set.
        """
        # Python & NumPy
        random.seed(seed)
        np.random.seed(seed)

        # PyTorch CPU/GPU
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic settings (optional, might affect performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Petri net internal seeding, if provided by your implementation
        if hasattr(self.pn, "seed") and callable(getattr(self.pn, "seed")):
            try:
                self.pn.seed(seed)
            except Exception:
                print(f"Warning: pn.seed() failed for seed {seed}.")
                pass
        if hasattr(self.pn, "set_seed") and callable(getattr(self.pn, "set_seed")):
            try:
                self.pn.set_seed(seed)
            except Exception:
                pass

    def enabled_actions(self, state):
        # Return positions in the current enabled list (exact indices env.step expects)
        if isinstance(self.pn.pn_actions, list) and len(self.pn.pn_actions) > 0:
            return list(range(len(self.pn.pn_actions)))
        return []

