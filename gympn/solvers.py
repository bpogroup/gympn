from abc import ABC, abstractmethod
from typing import Any, List, Dict
import torch
from gympn.utils import sim_tokens_values_from_bindings, binding_from_tokens_values
import random
import warnings


class BaseSolver(ABC):
    """
    Base class for solvers. Solvers are used to select a binding from a list of bindings according to a policy.
    """
    @abstractmethod
    def solve(self, observable_net, bindings: List) -> Any:
        """
        Select a binding from a list of bindings according to a policy.

        Parameters
        ----------
        :param observable_net: the GymProblem object representing the (observable) petri net.
        :param bindings: list of possible bindings to choose from.
        """

        pass

class GymSolver(BaseSolver):
    """
    Gym solvers are used to select a binding from a list of bindings according to a traind DRL policy.
    The policy is a neural network that takes as input the current state of the observable net and returns a binding.
    The weights of the policy are loaded from a file.

    Attributes
    ----------
    policy_model : torch.nn.Module
        The neural network model that represents the trained policy.
    """

    def __init__(self, weights_path: str, metadata: list):
        """
        Initialize the GymSolver with the path to the weights of the trained policy and the metadata of the assignment graph.

        Parameters
        ----------
        :param weights_path: Path to the weights of the trained policy.
        :param metadata: Metadata of the assignment graph that constitutes the observations for the trained policy. This is typically created via the get_metadata() method of the GymProblem.

        Returns
        ----------
        None
        """
        self.policy_model = torch.load(weights_path, weights_only=False)

    def solve(self, obs) -> Any:
        return self.policy_model.forward(obs)


class HeuristicSolver(BaseSolver):
    """
    Heuristic solver that can choose among REAL action bindings and,
    when exposed, a postpone pseudo-binding.

    Heuristic return formats supported:
      - 'postpone'
      - int (index into the provided 'bindings' list)
      - tuple (a binding tuple that must be present in 'bindings')
      - dict  {action_id: token_values} -> mapped to a timed binding
    """

    def __init__(self, heuristic_function: Any):
        self.heuristic_function = heuristic_function

    def solve(self, observable_net, bindings: List) -> Any:
        """
        Parameters
        ----------
        observable_net : GymProblem or observation-like object
            Used by the heuristic to inspect places/actions and flags.
        bindings : List
            List of candidates to choose from. Items are either:
              - REAL action timed bindings: ([ (place, token), ... ], time, transition)
              - postpone pseudo-binding:   (['postpone'], current_clock, None)

        Returns
        -------
        Either:
          - 'postpone'
          - a REAL timed binding (from 'bindings')
        """
        self.bindings = bindings

        # Identify real bindings vs. pseudo postpone entry
        def _is_real_binding(b) -> bool:
            return (
                isinstance(b, tuple) and len(b) >= 2 and
                isinstance(b[0], list) and len(b[0]) > 0 and isinstance(b[0][0], tuple)
            )

        real_bindings = [b for b in bindings if _is_real_binding(b)]
        postpone_present = any(isinstance(b, tuple) and b and b[0] == ['postpone'] for b in bindings)

        # Build token-combinations only for REAL bindings
        tokens_comb = sim_tokens_values_from_bindings(real_bindings)

        # Call heuristic, supporting both (obs, tokens) and (obs, tokens, bindings)
        try:
            selection = self.heuristic_function(observable_net, tokens_comb, bindings)
        except TypeError:
            selection = self.heuristic_function(observable_net, tokens_comb)

        # ---- Case A: explicit 'postpone'
        if selection == 'postpone':
            allow = getattr(observable_net, 'allow_postpone', False)
            # Only accept postpone if the environment currently allows postpone AND a postpone entry is present in 'bindings'
            if allow and postpone_present:
                return 'postpone'

            # Otherwise, gracefully fallback to a real binding when possible.
            # If a postpone entry exists but postpone is not allowed, prefer to fall back and warn the user.
            if postpone_present and not allow:
                warnings.warn("Heuristic requested 'postpone' but postpone is currently not allowed by the environment. Falling back to the first real binding.")
                if real_bindings:
                    return real_bindings[0]
                # No real bindings but postpone was present -> inconsistent state: raise informative error
                raise ValueError("Heuristic returned 'postpone' but postpone is not allowed and no real bindings are available.")

            # No postpone entry in bindings: fall back to first real binding if any, otherwise return the first binding available
            if real_bindings:
                return real_bindings[0]
            if bindings:
                # last resort: return whatever is present
                return bindings[0]
            raise ValueError("Heuristic returned 'postpone' but no bindings were provided.")

        # ---- Case B: integer index into 'bindings'
        if isinstance(selection, int):
            if selection < 0 or selection >= len(bindings):
                raise IndexError(f"Heuristic index {selection} out of range [0, {len(bindings)-1}]")
            chosen = bindings[selection]
            if isinstance(chosen, tuple) and chosen and chosen[0] == ['postpone']:
                # If heuristic returned the index of the postpone pseudo-binding, translate to 'postpone' iff allowed
                allow = getattr(observable_net, 'allow_postpone', False)
                if allow:
                    return 'postpone'
                # otherwise fall back to first real binding
                warnings.warn("Heuristic selected the postpone index but postpone is not allowed. Falling back to first real binding.")
                return real_bindings[0] if real_bindings else bindings[0]
            return chosen

        # ---- Case C: binding tuple
        if isinstance(selection, tuple):
            if selection and selection[0] == ['postpone']:
                allow = getattr(observable_net, 'allow_postpone', False)
                if allow and postpone_present:
                    return 'postpone'
                warnings.warn("Heuristic returned a postpone tuple but postpone is not allowed or not present. Falling back to first real binding.")
                return real_bindings[0] if real_bindings else bindings[0]
            if selection in bindings:
                return selection
            raise ValueError("Heuristic returned a binding tuple not present in 'bindings'.")

        # ---- Case D: {action_id: token_values}
        if isinstance(selection, dict):
            if len(selection) != 1:
                raise ValueError("Heuristic must return exactly one {action_id: token_values}.")
            action_id, token_values = next(iter(selection.items()))
            try:
                action_node = observable_net.id2node[action_id]
            except Exception:
                raise KeyError(f"Action id '{action_id}' not found in observable_net.id2node.")
            untimed = (action_node, token_values)
            return binding_from_tokens_values(untimed, real_bindings)

        raise TypeError("Heuristic must return one of: 'postpone' | int | binding tuple | {action_id: token_values}")


    def set_heuristic_function(self, heuristic_function: Any):
        """
        Set the heuristic function to be used by the solver.

        Parameters
        ----------
        :param heuristic_function: Heuristic function to be used by the solver.

        Returns
        ----------
        None

        """
        self.heuristic_function = heuristic_function

    #Helper functions to simplify the use of the heuristic function
    @staticmethod
    def get_place_tokens(place_id: str, pn) -> List:
        """
        Get the tokens of a place in the process network.

        Parameters
        ----------
        :param place_id: the id of the place.
        :param pn: the GymProblem object representing the (observable) petri net.

        Returns
        ----------
        :return: list of token values in the place.
        """
        return [t.value for p in pn.places if p._id == place_id for t in p.marking]  # (1)

    @staticmethod
    def get_place_tokens_by_type(place_id: str, desired_values: Dict, pn) -> List:
        """
        Get the tokens of a place in the process network.

        Parameters
        ----------
        :param place_id: the id of the place.
        :param desired_values: a dictionary having as key the attribute name and as value the desired value of the attribute (NOTE: this function assumes that token attributes are represented as dictionaries).
        :param pn: the GymProblem object representing the (observable) petri net.

        Returns
        ----------
        :return: list of token values in the place.
        """
        if not isinstance(desired_values, dict):
            raise TypeError("Desired_values must be a dictionary")
        if not all(isinstance(k, str) for k in desired_values.keys()):
            raise TypeError("Keys of desired_values must be strings")
        if len(desired_values) == 0:
            raise ValueError(
                "Desired_values must not be empty. If you want all tokens, use get_place_tokens(place_id, pn) instead.")

        ordered_token_values = HeuristicSolver.get_place_tokens(place_id, pn)
        if len(ordered_token_values) == 0:
            return []

        if not all(isinstance(t, dict) for t in ordered_token_values):
            raise TypeError("Tokens must be dictionaries")
        if not all(k in ordered_token_values[0] for k in desired_values.keys()):
            raise KeyError("Some keys of desired_values are not present in the tokens")

        # Corrected list comprehension
        return [t for t in ordered_token_values if all(t[k] == v for k, v in desired_values.items())]

    @staticmethod
    def get_action_available_assignments(action, actions_dict):
        """
        Get the available assignments for an action.

        Parameters
        ----------
        :param action: the action to be fired.
        :param actions_dict: the dictionary of actions and their available assignments.

        Returns
        -------
        :return: list of available assignments for the action.
        """
        if action in actions_dict.keys():
            return actions_dict[action]
        else:
            raise ValueError(f"Action {action} not found in actions_dict.")

    @staticmethod
    def get_attribute_based_assignments(action, attribute_dict, actions_dict):
        """
        Get the available assignments for an action based on a specific attribute.

        Parameters
        ----------
        :param action: the action to be fired.
        :param attribute_dict: the dictionary of attributes and their values.
        :param actions_dict: the dictionary of actions and their available assignments.

        Returns
        ----------
        :return: list of available assignments for the action based on the specified attribute.
        """
        available_assignments = HeuristicSolver.get_action_available_assignments(action, actions_dict)
        if attribute_dict is None:
            return available_assignments
        else:
            return [a for a in available_assignments if all(a[0][1].value[k] == v for k, v in attribute_dict.items())]

class RandomSolver(BaseSolver):
    def solve(self, observable_net, bindings: List) -> Any:
        """
        Select a binding randomly from a list of bindings.
        Parameters
        ----------
        :param observable_net: the GymProblem object representing the (observable) petri net.
        :param bindings: list of possible bindings to choose from.

        Returns
        ----------
        :return: a randomly selected binding from the list of bindings.
        """
        #TODO: Implement random selection with optional seeding
        # For now, just return a random binding
        b = random.choice(bindings)
        return b