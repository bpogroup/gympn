
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

    # Use the lightweight PN snapshot in get_state/set_state (markings + scalars
    # + shared trace) instead of a full deepcopy of the net structure. Set False
    # to fall back to the deepcopy path (equivalence testing / safety).
    LIGHT_SNAPSHOT = True

    def __init__(self, aepn):
        """"Initialize the environment with a GymProblem instance."""
        super().__init__()
        self.pn = aepn
        self.frozen_pn = copy.deepcopy(self.pn)
        # Ensure frozen copy does not carry over causal traces from previous runs
        try:
            if hasattr(self.frozen_pn, 'causal_trace') and self.frozen_pn.causal_trace is not None:
                self.frozen_pn.causal_trace.flush()
        except Exception:
            pass
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

    def step(self, action: int, build_obs: bool = True):
        """
        Execute one step in the environment.
        Returns (observation, reward, terminated, truncated, info)

        ``build_obs=False`` skips constructing the (expensive) HeteroData graph
        observation and returns ``None`` in its place. Building the graph is
        ~80% of a step's cost (profiled), so planners that only need the reward,
        the ``done`` flag and the updated ``pn.pn_actions`` — e.g. MCTS descent
        and rollout steps that continue with a cheap policy — pass False and
        build the graph on demand only where a network forward is actually
        required (node expansion).
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

        observation, terminated, self.i = self.pn.run_evolutions(
            self.run, self.i, self.active_model, build_obs=build_obs)

        # Prepare info dict
        if self.pn.causal_rl:
            # Always include causal trace when causal_rl is enabled
            # It accumulates throughout the episode and is needed at episode end
            info = {'pn_reward': self.pn.reward, 'eligibility_credits': self.pn.causal_trace}
        else:
            info = {'pn_reward': self.pn.reward}

        # Always return the true per-step reward. Historically this was zeroed
        # in causal mode ("episode-level credit assignment via causal traces"),
        # which silently starved every consumer of rewards_raw: the causal-mu
        # hybrid never mixed raw-reward GAE (it actually mixed a value-TD
        # term), and the LCV control-variate base was pure value noise. The
        # causal credit paths ignore step rewards by construction
        # (finish(mode='replace')), so restoring them changes nothing for
        # lrq/lrq2/mc_q at mu=0.
        reward = (self.pn.reward - old_rewards)

        return observation, reward, terminated, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment to its initial state."""
        if seed is not None:
            self.set_seed(seed)

        if self.debug:
            print(f"Entered reset with current reward for PN: {self.pn.reward} \n")
        self.pn = copy.deepcopy(self.frozen_pn)
        # Ensure the active PN starts with a fresh causal trace
        try:
            if hasattr(self.pn, 'causal_trace') and self.pn.causal_trace is not None:
                self.pn.causal_trace.flush()
        except Exception:
            pass
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
            # Lightweight PN snapshot (markings + scalars + shared trace) instead
            # of a full deepcopy of the net structure — ~20% of a search, profiled.
            # LIGHT_SNAPSHOT=False falls back to the old deepcopy (used to verify
            # behavioural equivalence).
            "pn_state": (self.pn.save_state() if AEPN_Env.LIGHT_SNAPSHOT else None),
            "pn": (None if AEPN_Env.LIGHT_SNAPSHOT else copy.deepcopy(self.pn)),
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
        if snapshot.get("pn_state") is not None:
            self.pn.restore_state(snapshot["pn_state"])
        else:
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

    def set_seed(self, seed: int, *, deterministic: bool = True) -> None:
        """
        Seed *all* stochastic elements for a reproducible run: Python, NumPy,
        PyTorch (CPU/GPU), cuDNN, and the underlying Petri net if it exposes a
        seed hook. Delegates to :func:`gympn.seed_everything` so there is a
        single source of truth for seeding across the library.

        Parameters
        ----------
        seed : int
            The seed to set.
        deterministic : bool, default True
            Request deterministic PyTorch kernels (see ``seed_everything``).
        """
        from .seeding import seed_everything
        seed_everything(seed, deterministic=deterministic)

        # Petri net internal seeding, if the problem exposes one.
        for hook in ("set_seed", "seed"):
            fn = getattr(self.pn, hook, None)
            if callable(fn):
                try:
                    fn(seed)
                    break
                except Exception:
                    pass

    def enabled_actions(self, state):
        # Return positions in the current enabled list (exact indices env.step expects)
        if isinstance(self.pn.pn_actions, list) and len(self.pn.pn_actions) > 0:
            return list(range(len(self.pn.pn_actions)))
        return []

