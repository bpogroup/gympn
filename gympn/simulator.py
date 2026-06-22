import copy
import os
import itertools
import random
import inspect
import uuid
import copy
import subprocess

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from simpn.simulator import SimProblem, SimEvent, SimVar, SimVarQueue, SimToken

from gympn.causal_traces import CausalTraces
from gympn.environment import AEPN_Env
from gympn.solvers import BaseSolver, GymSolver
from gympn.train import make_agent, make_parser, make_logdir, launch_tensorboard
from gympn.plotter import GraphPlotter
from gympn.visualisation import Visualisation
from gympn.wandb_integration import init_wandb


from typing import List, Tuple

def _is_real_binding_tuple(b) -> bool:
    return (
        isinstance(b, tuple) and len(b) >= 2 and
        isinstance(b[0], list) and len(b[0]) > 0 and isinstance(b[0][0], tuple)
    )


class GymProblem(SimProblem):
    """
        A decision problem GymProblem, which consists of a collection of simulation variables SimVar, a collection of simulation events SimEvent, and a collection of actions SimAction.
        The simulation has a time and a marking of SimVar. A marking is an assignment of values to SimVar variables (also see SimVar).
        The difference between a normal Python variable and a SimVar is that: (1) a SimVar can have multiple values; and (2) these values have a simulation time
        at which they become available. A SimEvent is a function that takes SimVar values as input and produces SimVar values (also see SimEvent). The difference with normal
        Python functions is that events can happen whenever its incoming SimVar parameters have a value. If a SimVar parameter has multiple values,
        it can happen on any of these values.

        Additionally, the GymProblem has a collection of actions SimAction. An action is a function that takes SimVar values as input and produces SimVar values according to a policy.
        The policy is defined as part of the action. The difference with events is that actions are deterministic and can only happen when its incoming SimVar parameters have a value.

        To be DRL-ready, a Gym environment is instantiated for each SimAction present in the network. This means that
        every environment trains a separate agent that is solely responsible for one action. The environment is responsible for
        the interaction between the agent and the simulation model. The agent receives the state of the simulation model and
        returns an action. The environment then executes the action on the simulation model and returns the new state to the agent.

        Future work: SingleAgentGymProblem should train a single agent for all the actions in the network.

        :param debugging: if set to True, produces more information for debugging purposes (defaults to True).
    """

    def __init__(self, debugging=True, binding_priority=lambda bindings: bindings[0], tag='e', has_var_attrs=True, solver=None, plot_observations=False, allow_postpone=False, causal_rl=False, causal_postpone_tokenflow=False):
        super().__init__(debugging, binding_priority)

        self.network_tag = NetworkTag(tag)  # boolean to indicate if it is time to take action ('a') or evolutions (i.e. normal events, 'e')
        self.has_var_attrs = has_var_attrs  # boolean to indicate if the problem has variable attributes (necessary for DRL)
        self.allow_postpone = allow_postpone  # boolean to indicate if the problem allows postponing actions
        self.just_postponed = False  # boolean to indicate if the system just postponed
        # When True, postpone is recorded in the causal trace as a real
        # token-flow transition (consume-and-recreate the action-eligible tokens,
        # with the postpone sentinel as their producer) so it becomes a node in
        # the token DAG and can receive flow credit. When False (default) postpone
        # is a sink-less sentinel that occupies its step slot but gets zero credit
        # (credit flows *through* it). See CAUSAL_ADVANTAGE_TD0.md (Design C).
        self.causal_postpone_tokenflow = causal_postpone_tokenflow

        self.reward_functions = {} # reward functions are associated to events/actions through a dictionary for backward compatibility with simpn events
        self.var_attributes = {} # similarly, places need to be associated with the attributes of their tokens
        self.reward = 0 #total cumulated reward (for DRL)

        # helpers for expansion and translation to assignment graph
        # evolutions are events that happen independently of the policy (saved in self.events)
        self.actions = []  # actions are distinguished from events

        self.arcs = [] #helper parameter used to simplify the creation of the graph observation

        self.pn_actions = [] #used to store the

        #DRL training parameters
        self.length = 0 #total episode length
        self.metadata = None#self.make_metadata()

        #solver
        self.solver = solver
        self.plot_observations = plot_observations #extra debugging
        if self.plot_observations:
            self.plotter = GraphPlotter()
        else:
            self.plotter = None

        #unobservable simvars
        self.unobservable_simvars = []
        #unobservable token attributes
        self.unobservable_token_attrs = {} #this is a dictionary with the place._id as key and a list of unobservable token attrs for that place as values
        #categorical token attributes
        self.categorical_token_attrs = {}

        #helpers for causal traces
        self.causal_rl = causal_rl  # whether to use causal traces for reward assignment
        if self.causal_rl:
            self.causal_trace = CausalTraces()  # tune gamma/lam if needed
            # Single source of truth: redistribute_rewards() reads this to decide
            # whether postpone sentinels are treated as credit sinks.
            self.causal_trace.postpone_tokenflow = self.causal_postpone_tokenflow

        self._debugging = True

    def add_gym_var(self, name, attributes: dict, priority=lambda token: token.time):
        """
        Creates a new GymVar with the specified name as identifier and the specified dictionary of token attributes.
        Token attributes specify what is the expected structure of the tokens in the place, and are expected to have
        string keys and numerical values.
        Adds the GymVar to the problem and returns it.


        :param name: a name for the SimVar.
        :param attributes: a dictionary of token attributes
        :param priority: a function that takes a token as input and returns a value that is used to sort the tokens in the order in which they will be processed (lower values first). The default is processing in the order of the time of the token.
        :return: a SimVar with the specified name as identifier.
        """
        # Generate and add SimVar
        result = GymVar(name, attributes, priority=priority)
        self.add_prototype_var(result)
        return result


    def add_var(self, name, initial=None, var_attributes=None, categorical_values=None):
        """
        Creates a new SimVar with the specified name and initial value. Adds the SimVar to the problem and returns it.

        :param name: the identifier of the SimVar.
        :param initial: the initial value of the SimVar.
        :param var_attributes: the set of attributes that characterize the tokens in the Simvar
        :param categorical_values: a dictionary containing var attributes as names and a list [min, max, step] specifying the range and step (necessary for one-hot-encoding)
        :return: a SimVar with the specified parameters.
        """

        if categorical_values is None:
            categorical_values = {}
        ret = super().add_var(name, initial)


        if var_attributes is not None:
            self.var_attributes[name] = var_attributes
        elif self.has_var_attrs:
            raise ValueError("Var attributes must be provided.")
        else:
            print("Warning: no var attributes provided. This may cause problems with DRL solvers.")

        if categorical_values:
            self.categorical_token_attrs.update({ret._id: categorical_values})
        #TODO: handle exceptions (non-existing var name, min higher than max, more than three values...)

        return ret

    def add_event(self, inflow, outflow, behavior, name=None, guard=None, reward_function=None):
        #call the parent class method
        event = super().add_event(inflow, outflow, behavior, name, guard)

        t_name = name if name is not None else behavior.__name__

        if t_name is not None:
            if reward_function is not None:
                self.reward_functions[t_name] = reward_function

        if reward_function is not None:
            if not callable(reward_function):
                raise TypeError(
                    "Event " + t_name + ": the reward function must be a function. (Maybe you made it a function call, exclude the brackets.)")
            parameters = inspect.signature(reward_function).parameters
            num_mandatory_params = sum(
                1 for p in parameters.values()
                if p.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]
                # count all parameters which are not "*args" or "**kwargs"
            )
            if num_mandatory_params != len(inflow):
                raise TypeError(
                    "Event " + t_name + ": the reward function must take as many parameters as there are input variables.")

        return event



    def add_action(self, inflow, outflow, behavior, name=None, guard=None, reward_function=None, solver=None):
        """
                Creates a new SimEvent with the specified parameters (also see SimEvent). Adds the SimEvent to the problem and returns it.

                :param inflow: a list of incoming SimVar of the event.
                :param outflow: a list of outgoing SimVar of the event/
                :param behavior: a function that takes as many parameters as there are incoming SimVar. The function must return a list with as many elements as there are outgoing SimVar. When the event happens, the function is performed on the incoming SimVar and the result of the function is put on the outgoing SimVar.
                :param name: the identifier of the event.
                :param guard: a function that takes as many parameters as there are incoming SimVar. The function must evaluate to True or False for all possible values of SimVar. The event can only happen for values for which the guard function evaluates to True.
                :param reward_function: a function that takes as many parameters as there are incoming SimVar. The function must return a reward value for the event.
                :param solver: a solver that decides which specific binding to use for firing.
                :return: a SimAction with the specified parameters.
                """

        # Check name
        t_name = name
        if t_name is None:
            if behavior.__name__ == "<lambda>":
                raise TypeError("Event name must be set or procedure behavior function must be named.")
            else:
                t_name = behavior.__name__
        if t_name in self.id2node:
            raise TypeError("Event " + t_name + ": node with the same name already exists. Names must be unique.")

        # Check inflow
        c = 0
        for i in inflow:
            if not isinstance(i, SimVar):
                raise TypeError("Event " + t_name + ": inflow with index " + str(c) + " is not a SimVar.")
            c += 1

        # Check outflow
        c = 0
        for o in outflow:
            if not isinstance(o, SimVar):
                raise TypeError("Event " + t_name + ": outflow with index " + str(o) + " is not a SimVar.")
            c += 1

        # Check behavior function
        if not callable(behavior):
            raise TypeError(
                "Event " + t_name + ": the behavior must be a function. (Maybe you made it a function call, exclude the brackets.)")
        parameters = inspect.signature(behavior).parameters
        if len(parameters) != len(inflow):
            raise TypeError(
                "Event " + t_name + ": the behavior function must take as many parameters as there are input variables.")

        # Check constraint
        if guard is not None:
            if not callable(guard):
                raise TypeError(
                    "Event " + t_name + ": the constraint must be a function. (Maybe you made it a function call, exclude the brackets.)")
            parameters = inspect.signature(guard).parameters
            if len(parameters) != len(inflow):
                raise TypeError(
                    "Event " + t_name + ": the constraint function must take as many parameters as there are input variables.")

        # If the problem allows postpone, all SimActions also have a guard that checks if the action was just postponed



        if self.allow_postpone:
            if guard is not None:
                old_guard = guard
                sig = inspect.signature(old_guard)
                params = ', '.join(sig.parameters.keys())

                # Create the function code as a string
                func_code = f"""def new_guard({params}):
                if self.just_postponed:
                    return False
                return old_guard({params})
                """

                # Prepare the local namespace for exec
                local_vars = {'self': self, 'old_guard': old_guard}
                exec(func_code, local_vars)
                guard = local_vars['new_guard']
            else:
                sig = inspect.signature(behavior)
                params = ', '.join(sig.parameters.keys())

                # Create the function code as a string
                func_code = f"""def new_guard({params}):
                if self.just_postponed:
                    return False
                return True
                """

                # Prepare the local namespace for exec
                local_vars = {'self': self}
                exec(func_code, local_vars)
                guard = local_vars['new_guard']

        # Generate and add SimEvent
        result = SimAction(t_name, guard=guard, behavior=behavior, incoming=inflow, outgoing=outflow, solver=solver)

        self.actions.append(result)

        if reward_function is not None:
            self.reward_functions[t_name] = reward_function
        #else:
        #    print("Warning: no reward function provided. A reward of 0 is assumed.")
        #    self.reward_functions[t_name] = lambda x: 0
        self.id2node[t_name] = result

        return result


    def bindings(self):
        """
        Calculates the set of timed bindings enabled for all events and actions in the problem.

        The method determines which bindings are currently enabled based on the simulation clock and the network tag.
        It also updates the simulation clock and network tag as necessary.

        :return: A tuple containing:
            - A list of tuples ([(place, token), (place, token), ...], time, event) representing the enabled bindings.
            - A boolean indicating whether the model is still active.
        """
        #if self.just_postponed:
            #print("Just postponed, checking bindings...")

        timed_bindings_evo = []
        timed_bindings_act = []

        # Collect bindings from events and actions
        for t in self.events:
            for (binding, time) in self.event_bindings(t):
                timed_bindings_evo.append((binding, time, t))
        for a in self.actions:
            for (binding, time) in self.event_bindings(a):
                timed_bindings_act.append((binding, time, a))


        if len(timed_bindings_evo) == 0 and len(timed_bindings_act) == 0:
            return [], False

        if self.network_tag.is_evolution():
            timed_bindings_curr = timed_bindings_evo
            timed_bindings_other = timed_bindings_act
        else:
            timed_bindings_curr = timed_bindings_act
            timed_bindings_other = timed_bindings_evo

        timed_bindings_curr.sort(key=lambda b: b[1])
        timed_bindings_other.sort(key=lambda b: b[1])

        # Prevent advancing clock to action bindings if just postponed
        if self.allow_postpone and self.just_postponed and self.network_tag.is_action():
            return [], True

        # Check if the tag needs to be updated
        if len(timed_bindings_curr) > 0 and timed_bindings_curr[0][1] <= self.clock:
            pass
        elif len(timed_bindings_curr) > 0 and timed_bindings_curr[0][1] > self.clock:
            if len(timed_bindings_other) > 0:
                if timed_bindings_other[0][1] < timed_bindings_curr[0][1]: #switch tag only if the time that enables bindings in the other tag is strictly lower than the time that enables new bindings in the current tag
                    self.network_tag.update_tag()
                    if self.clock < timed_bindings_other[0][1]:
                        self.clock = timed_bindings_other[0][1]
                    timed_bindings_curr = timed_bindings_other
                else:
                    if self.clock < timed_bindings_curr[0][1]:
                        self.clock = timed_bindings_curr[0][1]
            else:
                if self.clock < timed_bindings_curr[0][1]:
                    self.clock = timed_bindings_curr[0][1]
        elif len(timed_bindings_curr) == 0 and len(timed_bindings_other) != 0:
            self.network_tag.update_tag()
            if self.clock < timed_bindings_other[0][1]:
                self.clock = timed_bindings_other[0][1]
            timed_bindings_curr = timed_bindings_other
        else:
            return [], False

        # CRITICAL FIX: Check just_postponed AGAIN after potential tag switch.
        # The tag may have been switched from 'e' to 'a' in the logic above
        # (e.g., when evolutions are in the future but actions are available sooner).
        # After a postpone, the agent must NOT be given action bindings —
        # instead, advance the clock to the next evolution event.
        if self.allow_postpone and self.just_postponed and self.network_tag.is_action():
            # Tag was switched to 'a' but we just postponed.
            # Switch back to 'e' and advance to the next evolution time.
            if len(timed_bindings_evo) > 0:
                self.network_tag.tag = 'e'
                if self.clock < timed_bindings_evo[0][1]:
                    self.clock = timed_bindings_evo[0][1]
                timed_bindings_curr = timed_bindings_evo
            else:
                # No evolution events at all — model is stuck
                return [], True

        # Return the timed bindings that have time <= clock
        bindings = [(binding, time, t) for (binding, time, t) in timed_bindings_curr if time <= self.clock]
        # The other bindings are useful to delay
        #time_other_bindings = [t[1] for t in timed_bindings_other][0] if len(timed_bindings_other) > 0 else None
        return bindings, True#, time_other_bindings

    def simulate(self, duration, reporter=None):
        """
        Executes a simulation run for the specified duration.

        The simulation runs events and actions until no more can occur or the specified duration is reached.
        If multiple events or actions can occur, one is selected based on the network tag and solver.

        :param duration: The maximum duration of the simulation.
        :param reporter: A reporter or list of reporters to log simulation events. Each reporter must implement a `callback` method.
        """
        active_model = True
        while self.clock <= duration and active_model:
            bindings, active_model = self.bindings()
            if len(bindings) > 0 and self.network_tag.is_evolution():
                timed_binding = bindings[0]
                self.fire(timed_binding)
                if reporter is not None:
                    # report changes in marking
                    self.print_report(reporter, timed_binding)
                #if timed_binding[-1]._id in self.reward_functions.keys():
                self.update_reward(timed_binding)
            elif len(bindings) > 0 and self.network_tag.is_action():
                for a in self.actions:
                    timed_binding = a.execute(bindings, self)
                    #timed_binding = [el for el in bindings if el[0] == chosen_binding][0]
                    #result_tokens = self.fire(timed_binding)
                    #if timed_binding[-1]._id in self.reward_functions.keys():
                    self.update_reward(timed_binding)
                    if reporter is not None:
                        #report changes in marking
                        self.print_report(reporter, timed_binding)

    def fire(self, timed_binding):
        """
        Fires the specified timed binding.
        Binding is a tuple ([(place, token), (place, token), ...], time, event)
        """
        try:
            (binding, time, event) = timed_binding
        except:
            raise TypeError("Binding " + str(timed_binding) + ": is not a valid timed binding.")

        if isinstance(event, SimEvent) and not isinstance(event, SimAction):
            self.just_postponed = False

        # process incoming places:
        variable_assignment = []
        for (place, token) in binding:
            # remove tokens from incoming places
            place.remove_token(token)
            # assign values to the variables on the arcs
            variable_assignment.append(token.value)

        # calculate the result of the behavior of the event
        try:
            result = event.behavior(*variable_assignment)
        except Exception as e:
            raise TypeError("Event " + str(event) + ": behavior function generates exception for values " + str(variable_assignment) + ".") from e
        if self._debugging:
            if type(result) != list:
                raise TypeError("Event " + str(event) + ": behavior function does not generate a list for values " + str(variable_assignment) + ".")
            if len(result) != len(event.outgoing):
                raise TypeError("Event " + str(event) + ": behavior function does not generate as many values as there are output variables for values " + str(variable_assignment) + ".")
            i = 0
            for r in result:
                if r is not None:
                    if isinstance(event.outgoing[i], SimVarQueue):
                        if not isinstance(r, list):
                            raise TypeError("Event " + str(event) + ": does not generate a queue for variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                    else:
                        if not isinstance(r, SimToken):
                            raise TypeError("Event " + str(event) + ": does not generate a token for variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                        if not (type(r.delay) is int or type(r.delay) is float):
                            raise TypeError("Event " + str(event) + ": does not generate a numeric value for the delay of variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                        if not (type(r.time) is int or type(r.time) is float):
                            raise TypeError("Event " + str(event) + ": does not generate a numeric value for the time of variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ".")
                i += 1

        new_tokens = []
        for i in range(len(result)):
            if result[i] is not None:
                if isinstance(event.outgoing[i], SimVarQueue):
                    event.outgoing[i].add_token(result[i])
                else:
                    if result[i].time > 0 and result[i].delay == 0:
                        raise TypeError("Deprecated functionality: Event " + str(event) + ": generates a token with a delay of 0, but a time > 0, for variable " + str(event.outgoing[i]) + " for values " + str(variable_assignment) + ". It seems you are using the time of the token to represent the delay.")
                    token = SimToken(result[i].value, time=self.clock + result[i].delay)
                    if self.causal_rl:
                        #tokens need to be marked with unique ids
                        setattr(token, '_id', str(uuid.uuid4()))
                    event.outgoing[i].add_token(token)
                    new_tokens.append(token)

        return new_tokens


    def print_report(self, reporter, timed_binding):
        """
        Calls the reporter's callback function to log the details of a binding.

        :param reporter: A reporter or list of reporters to log simulation events.
        :param timed_binding: The binding that occurred, represented as a tuple (binding, time, event).
        """
        if type(reporter) == list:
            for r in reporter:
                r.callback(timed_binding)
        else:
            reporter.callback(timed_binding)

    def get_heuristic_observation(self):
        """
        Creates a lean observation for heuristic solvers:
          - Includes only observable places and any transitions fully connected to those places.
          - Does NOT add a 'postpone' node (postpone is exposed via the pseudo-binding).
          - Propagates postpone-related flags for consistent logic in heuristics.
        """
        ret = GymProblem(debugging=self._debugging,
                         binding_priority=self.binding_priority,
                         tag=self.network_tag.tag,  # the tag value is not used by the heuristic itself
                         has_var_attrs=self.has_var_attrs,
                         solver=None,
                         plot_observations=False,
                         allow_postpone=self.allow_postpone,  # propagate flags
                         causal_rl=self.causal_rl,
                         causal_postpone_tokenflow=self.causal_postpone_tokenflow)

        # carry flags explicitly (in case constructor defaults differ)
        ret.allow_postpone = self.allow_postpone
        ret.just_postponed = self.just_postponed

        # copy places (excluding unobservable)
        # FIX: deepcopy places that need token attr removal to avoid mutating originals
        observable_places = [p for p in self.places if p._id not in self.unobservable_simvars]
        ret.places = []
        for p in observable_places:
            if self.unobservable_token_attrs and p._id in self.unobservable_token_attrs:
                p_copy = copy.deepcopy(p)
                # Strip unobservable attributes from token values in the copy
                for token in p_copy.marking:
                    if isinstance(token.value, dict):
                        for key in list(self.unobservable_token_attrs[p._id]):
                            token.value.pop(key, None)
                ret.places.append(p_copy)
            else:
                ret.places.append(p)
        for p in ret.places:
            ret.id2node[p._id] = p

        # events fully within observed places — compare by _id since we may have copies
        observable_place_ids = {p._id for p in ret.places}
        for e in self.events:
            if all(p._id in observable_place_ids for p in e.incoming) and \
               all(p._id in observable_place_ids for p in e.outgoing):
                ret.events.append(e)
                ret.id2node[e._id] = e

        # actions fully within observed places
        for a in self.actions:
            if all(p._id in observable_place_ids for p in a.incoming) and \
               all(p._id in observable_place_ids for p in a.outgoing):
                ret.actions.append(a)
                ret.id2node[a._id] = a

        ret.clock = self.clock
        return ret

    def get_graph_observation(self, minimal_obs=False, normalize=False, remove_empty_nodes=True, add_self_loops=True):
        """
        Generates a graph-based observation of the problem.

        The observation is represented as a heterogeneous graph, where nodes correspond to places and transitions,
        and edges represent the connections between them.

        :param normalize: Whether to normalize the node features (default is True).
        :param add_self_loops: Whether to add self-loops to action transition nodes (default is True).
        :return: A dictionary containing:
            - 'graph': The graph observation as a `HeteroData` object.
            - 'mask': A mask graph indicating valid transitions.
            - 'actions_dict': A mapping of transitions to their bindings.
        """

        #graph observation expects places to contain their color sets types (only the types of variables for each token)

        ret_graph = HeteroData()
        token_value = None  # used to store the token value of the current place
        new_values = None  # used to store the new values of the current place
        a_transition_dict = {}  # helper to keep track of the transitions indexes in the graph
        e_transition_dict = {}  # helper to keep track of the transitions indexes in the graph


        transition_binding_map = []
        node_types = [p._id for p in self.places if
                      p._id not in self.unobservable_simvars]  # [set(p.tokens_attributes.keys()) for p in self.places]
        node_types.append('a_transition')
        node_types.append('e_transition')

        self.expanded_pn, transition_binding_map_wrong = self.expand_no_future_tokens()  # expand the petri net into a new petri net with 1-bounded places

        # Update the HeteroData object with the places attributes and the transition nodes
        for n_t in node_types:
            if n_t != 'a_transition' and n_t != 'e_transition':
                p_nodes = []
                # Check if n_t is already a key in ret_graph
                for p in self.expanded_pn.places:
                    p_name_original = GymProblem._get_string_before_last_dot(p._id)
                    if p_name_original == n_t:

                        if p.marking:

                            for token in p.marking:  # will always be only a single token because places were expanded already
                                if isinstance(token.value, dict): #token attrs observability assumes tokens are dictionaries
                                    token_value = torch.tensor([float(value) for key, value in token.value.items() if key not in self.unobservable_token_attrs.get(p_name_original, []) and key not in self.categorical_token_attrs.get(p_name_original, {})]).type(
                                        torch.float32) #if the token has no attributes, we create an empty tensor
                                    if p_name_original in self.categorical_token_attrs.keys():
                                        #one hot encoding of categorical values
                                        for key, value in token.value.items():
                                            if key in self.categorical_token_attrs[p_name_original].keys():
                                                #one hot encoding
                                                one_hot = torch.zeros(len(self.categorical_token_attrs[p_name_original][key]), dtype=torch.float32)
                                                one_hot[value] = 1.0
                                                token_value = torch.cat((token_value, one_hot), dim=0)

                                elif isinstance(token.value, (tuple, list)):
                                    if all(isinstance(item, dict) for item in token.value):
                                        token_value = torch.tensor(
                                            [float(value) for d in token.value for value in d.values()]).type(torch.float32)
                                    else:
                                        token_value = torch.tensor([float(value) for value in token.value]).type(
                                            torch.float32) #floats are always observable
                                else:
                                    raise TypeError("Unsupported token value type.")

                            p_nodes.append(token_value)

                        else: #if no token is present in the marking, we need to include an empty node with size equal to var_attributes[node._id]
                            #if attributes are one_hot_encoded, we need to consider that in length
                            len_attrs = len(self.var_attributes[n_t]) if n_t not in self.unobservable_token_attrs.keys() else len(
                                [el for el in self.var_attributes[n_t] if el not in self.unobservable_token_attrs[n_t]])
                            if n_t in self.categorical_token_attrs.keys():
                                len_attrs += sum(len(v) - 1 for v in self.categorical_token_attrs[n_t].values())
                            p_nodes.append(torch.tensor([], dtype=torch.float32).reshape(0, len_attrs))

                nodes = p_nodes
            elif n_t == 'a_transition':
                t_nodes = []
                #t_nodes_mask = []
                a_transition_dict = {} #helper to keep track of the transitions indexes in the graph
                for i, t in enumerate(self.expanded_pn.actions):
                    if add_self_loops:
                        t_name = GymProblem._get_string_before_last_dot(t._id)
                        t_values = [1 if action._id == t_name else 0 for action in self.actions] #a_transitions know which transition type they are (useful with self loops and multiple actions)
                    else:
                        t_values = [] #otherwise they do not need to carry information

                    b_list = [el.marking[0] for el in t.incoming]
                    binds = [([place for place in self.places if place._id == GymProblem._get_string_before_last_dot(el._id)][0], el.marking[0]) for el in t.incoming]
                    b_time = max([t.time for t in b_list]) #changed min to max, it was an error!
                    original_transition = [action for action in self.actions if action._id == GymProblem._get_string_before_last_dot(t._id)][0]

                    if b_time <= self.clock:

                        t_nodes.append(torch.tensor(t_values).type(torch.float32))
                        transition_binding_map.append((binds, b_time, original_transition))  # include only valid bindings
                    #else:
                    #    t_nodes.append(torch.tensor(t_values).type(torch.float32))

                    #if b_time <= self.clock:
                    #    pn_bindings[i] = (b_list, b_time)
                    #    t_nodes.append(torch.tensor(t_values).type(torch.float32))
                    #    transition_binding_map.append((b_list, b_time, t))  # include only valid bindings
                    #else:
                    #    t_nodes.append(
                    #        torch.tensor(t_values).type(torch.float32))  # still add node, but not as valid action

                    a_transition_dict[t._id] = i

                nodes = t_nodes
            else:
                t_nodes = []

                e_transition_dict = {}  # helper to keep track of the transitions indexes in the graph

                for i, t in enumerate(self.expanded_pn.events):
                    t_name = GymProblem._get_string_before_last_dot(t._id)
                    t_values = [1 if e._id == t_name else 0 for e in self.events]
                    t_nodes.append(torch.tensor(t_values).type(torch.float32))
                    e_transition_dict[t._id] = i

                nodes = t_nodes


            if nodes:
                if n_t not in ret_graph.node_types:
                    # If not, create a new Data object and assign it to ret_graph[n_t]
                    new_nodes = torch.stack(nodes)
                    if new_nodes.numel() == 0 and n_t not in ['a_transition', 'e_transition']:
                        if n_t not in self.unobservable_token_attrs.keys():
                            ret_graph[n_t].x = torch.empty(0, len(self.var_attributes[n_t]))
                        else:
                            ret_graph[n_t].x = torch.empty(0, len(
                                [el for el in self.var_attributes[n_t] if el not in self.unobservable_token_attrs[n_t]]))
                    else:
                        ret_graph[n_t].x = new_nodes

                else:
                    # If yes, stack the new nodes to the existing ones
                    ret_graph[n_t].x = torch.cat((ret_graph[n_t].x, torch.stack(nodes)), dim=1)
            else:
                if n_t not in ['a_transition', 'e_transition']:
                    if n_t not in self.unobservable_token_attrs.keys():
                        ret_graph[n_t].x = torch.empty(0, len(self.var_attributes[n_t]))  # create empty placeholder of the right size
                    else:
                        ret_graph[n_t].x = torch.empty(0, len(
                            [el for el in self.var_attributes[n_t] if el not in self.unobservable_token_attrs[n_t]]))  # create empty placeholder of the right size


        # Update the HeteroData object with the arcs
        for a in self.expanded_pn.arcs:
            #a[0] is the source
            source_name = GymProblem._get_string_before_last_dot(a[0]._id)
            dest_name = GymProblem._get_string_before_last_dot(a[1]._id)
            key = None

            if type(a[0]) is SimVar and source_name not in self.unobservable_simvars:# and source_name in ret_graph: #if the place has tokens inside of it
                if type(a[1]) is SimAction:
                    #key = (source_name, 'edge', 'a_transition')
                    #different edge type for each place to transition connection
                    key = (source_name, 'edge_to_a_transition', 'a_transition')
                    if a[0].marking:
                        new_values = torch.tensor(
                            [[int(a[0]._id.split('.')[-1])],
                             [a_transition_dict[a[1]._id]]]).type(torch.int64)
                    else:
                        new_values = torch.empty([2, 0], dtype=torch.int64)
                else:
                    #key = (source_name, 'edge', 'e_transition')
                    key = (source_name, 'edge_to_e_transition', 'e_transition')
                    if a[0].marking:
                        new_values = torch.tensor(
                            [[int(a[0]._id.split('.')[-1])],
                             [e_transition_dict[a[1]._id]]]).type(torch.int64)
                    else:
                        new_values = torch.empty([2, 0], dtype=torch.int64)

            elif type(a[1]) is SimVar and dest_name not in self.unobservable_simvars:# and ret_graph[dest_name]: #if the place has tokens inside of it
                if type(a[0]) is SimAction:
                    #key = ('a_transition', 'edge', dest_name)
                    key = ('a_transition', 'edge_to_' + dest_name, dest_name)
                    if a[1].marking:
                        new_values = torch.tensor(
                            [[a_transition_dict[a[0]._id]],
                             [int(a[1]._id.split('.')[-1])]]).type(torch.int64)
                    else:
                        new_values = torch.empty([2, 0], dtype=torch.int64)
                else:
                    #key = ('e_transition', 'edge', a[1]._id.split('.')[0])
                    key = ('e_transition', 'edge_to_' + dest_name, dest_name)
                    if a[1].marking:
                        new_values = torch.tensor(
                            [[e_transition_dict[a[0]._id]],
                             [int(a[1]._id.split('.')[-1])]]).type(torch.int64)
                    else:
                        new_values = torch.empty([2, 0], dtype=torch.int64)


            if key is not None:
                if key not in ret_graph.edge_types:
                    ret_graph[key].edge_index = new_values
                else:
                    ret_graph[key].edge_index = torch.cat((ret_graph[key].edge_index, new_values), dim=1)

        if add_self_loops:
            #create self loops on 'a_transition' nodes
            if 'a_transition' in ret_graph.node_types:
                num_a_transitions = ret_graph['a_transition'].x.size(0)
                self_loop_edges = torch.stack([torch.arange(num_a_transitions), torch.arange(num_a_transitions)])
                #if ('a_transition', 'edge', 'a_transition') not in ret_graph.edge_types:
                #    ret_graph[('a_transition', 'edge', 'a_transition')].edge_index = self_loop_edges
                #else:
                #    ret_graph[('a_transition', 'edge', 'a_transition')].edge_index = torch.cat(
                #        (ret_graph[('a_transition', 'edge', 'a_transition')].edge_index, self_loop_edges), dim=1)
                if ('a_transition', 'self_loop', 'a_transition') not in ret_graph.edge_types:
                    ret_graph[('a_transition', 'self_loop', 'a_transition')].edge_index = self_loop_edges
                else:
                    ret_graph[('a_transition', 'self_loop', 'a_transition')].edge_index = torch.cat(
                        (ret_graph[('a_transition', 'self_loop', 'a_transition')].edge_index, self_loop_edges), dim=1)

        if remove_empty_nodes:
            #remove empty nodes connects all the incoming edges to the outgoing edges
            ret_graph = self.remove_empty_nodes(ret_graph)

        if minimal_obs:
            ret_graph = self.minimal_graph(ret_graph)

        if normalize:
            # Normalize the features
            transform = SafeNormalizeFeatures()
            #transform2 = remove_isolated_nodes.RemoveIsolatedNodes()
            #transform = Compose([transform1, transform2])
            ret_graph = transform(ret_graph)

        self.pn_actions = transition_binding_map

        if self.allow_postpone:
            # postpone is a new node type that allows to delay actions
            # it is connected to all nodes if not just_postponed, otherwise it is isolated
            #postpone_node_feature = torch.tensor([[1.0 if not self.just_postponed else 0.0]]).type(torch.float32)
            postpone_node_feature = torch.tensor([[0.0]]).type(torch.float32)
            ret_graph['postpone'].x = postpone_node_feature
            #create special edges from all nodes to postpone
            for n_t in ret_graph.node_types:
                if n_t != 'postpone':
                    num_nodes = ret_graph[n_t].x.size(0)
                    source_indices = torch.arange(num_nodes)
                    dest_indices = torch.zeros(num_nodes,
                                               dtype=torch.int64)  # all point to the single postpone node
                    edge_indices = torch.stack([source_indices, dest_indices])

                    # Use a source-specific edge type
                    #edge_type = (n_t, f'to_postpone_from_{n_t}', 'postpone')
                    edge_type = (n_t, 'to_postpone', 'postpone')
                    if edge_type not in ret_graph.edge_types:
                        ret_graph[edge_type].edge_index = edge_indices
                    else:
                        ret_graph[edge_type].edge_index = torch.cat((ret_graph[edge_type].edge_index, edge_indices),
                                                                    dim=1)

            #also include postpone in self.pn_actions with special binding
            transition_binding_map.append((['postpone'], self.clock, None)) #no binding, time is current clock


        if self.metadata is None: #TODO: handle case where postpone is initially not available
            self.metadata = ret_graph.metadata()

        if self.plot_observations:
            self.plotter.plot_side_by_side(self.expanded_pn, ret_graph)

        return {'graph': ret_graph, 'actions_dict': transition_binding_map}#{'graph': ret_graph, 'mask': mask_graph, 'actions_dict': transition_binding_map}


    def remove_empty_nodes(self, ret_graph):
        """
        Removes empty nodes (e.g. nodes that have numel()==0) from the graph observation by modifying the connectivity of the graph.
        If an empty node has incoming and outgoing edges, those will be removed.

        Empty nodes are those that do not have any tokens or attributes associated with them.

        :param ret_graph: The graph observation as a `HeteroData` object.
        :return: The updated `HeteroData` object with empty nodes removed.
        """
        # Create a copy of the graph to avoid modifying it while iterating
        updated_graph = ret_graph.clone()

        # Iterate over all node types in the graph
        for node_type in updated_graph.node_types:
            # Get the nodes of the current type
            nodes = updated_graph[node_type].x

            # Check if the nodes are empty (numel() == 0)
            if nodes.numel() == 0:
                # If empty, remove the edges connected to this node type
                for edge_type in updated_graph.edge_types:
                    if edge_type[0] == node_type or edge_type[2] == node_type:
                        updated_graph[edge_type].edge_index = torch.tensor([], dtype=torch.int64).reshape(2, 0)

                # Remove the empty node type from the graph
                #del updated_graph.node_types[node_type]

        return updated_graph

    def expand_no_future_tokens(self):
        """
        Expands the attributed A-E PN into a 1-bounded attributed A-E PN.
        Filters out tokens and bindings with time > self.clock.
        :return: A tuple containing:
        - The expanded `GymProblem` instance.
        - A mapping of transitions to their bindings in the expanded problem.
        """
        expanded_pn = GymProblem(has_var_attrs=True)
        expanded_pn.arcs = []

        #keep track of valid tokens in every place
        token_map = {}

        # Expand places, filtering out future tokens
        for p in self.places:
            valid_tokens = [t for t in p.marking if t.time <= self.clock]
            token_map[p._id] = valid_tokens
            if valid_tokens:
                for i, t in enumerate(valid_tokens):
                    new_place_id = f"{p._id}.{i}"
                    new_place = expanded_pn.add_var(name=new_place_id, var_attributes=self.var_attributes[p._id])
                    new_place.put(t.value, time=t.time)
                    if hasattr(t, '_id'):
                        #keep the same token id for causal rl
                        setattr(new_place.marking[0], '_id', t._id)
            else:
                new_place_id = f"{p._id}.0"
                expanded_pn.add_var(name=new_place_id, var_attributes=self.var_attributes[p._id])

        # Expand events (evolutions)
        for t in self.events:
            new_inflow = []
            for p in t.incoming:
                if isinstance(p, SimVar):
                    if len(token_map[p._id]):
                        for i in range(len(token_map[p._id])):
                            new_inflow.append(expanded_pn.id2node[f"{p._id}.{i}"])
                            expanded_pn.arcs.append((expanded_pn.id2node[f"{p._id}.{i}"], t))
                    else:
                        new_inflow.append(expanded_pn.id2node[f"{p._id}.0"])
                        expanded_pn.arcs.append((expanded_pn.id2node[f"{p._id}.0"], t))
                elif isinstance(p, SimVarQueue):
                    new_inflow.append(p)

            new_outflow = []
            for p in t.outgoing:
                if isinstance(p, SimVar):
                    if len(token_map[p._id]):
                        for i in range(len(token_map[p._id])):
                            new_outflow.append(expanded_pn.id2node[f"{p._id}.{i}"])
                            expanded_pn.arcs.append((t, expanded_pn.id2node[f"{p._id}.{i}"]))
                    else:
                        new_outflow.append(expanded_pn.id2node[f"{p._id}.0"])
                        expanded_pn.arcs.append((t, expanded_pn.id2node[f"{p._id}.0"]))
                elif isinstance(p, SimVarQueue):
                    new_outflow.append(p)

            expanded_pn.add_event(new_inflow, new_outflow, behavior=t.behavior, name=t._id)

        # Expand actions (transitions)
        transition_binding_map = []
        for t in self.actions:
            transition_bindings = self.event_bindings(t)
            valid_transition_bindings = [(binding, time) for (binding, time) in transition_bindings if time <= self.clock]
            if len(valid_transition_bindings) == 0:
                continue  # No valid bindings for this transition
            for binding, time in valid_transition_bindings:
                if time > self.clock:
                    print("Unexpected future binding found during expansion.")
                    continue  # Skip future bindings

                transition_binding_map.append([binding, time, t])
                new_inflows = {}
                new_outflows = {}

                for place, token in binding:
                    for new_place in [pl for pl in expanded_pn.places if
                                      self._get_string_before_last_dot(pl._id) == place._id]:
                        new_inflows.setdefault(place._id, []).append(new_place)

                for p in t.outgoing:
                    if len(token_map[p._id]):
                        for i in range(len(token_map[p._id])):
                            new_outflows.setdefault(p._id, []).append(expanded_pn.id2node[f"{p._id}.{i}"])
                    else:
                        new_outflows[p._id] = [expanded_pn.id2node[f"{p._id}.0"]]

                in_cartesian_product = [list(comb) for comb in itertools.product(*new_inflows.values())]
                out_cartesian_product = list(new_outflows.values())[0]

            for index_new_action, io in enumerate(in_cartesian_product):
                new_inflow = io
                new_outflow = out_cartesian_product
                #dummy behavior that does nothing, as the actual behavior is not relevant in the expanded net
                #dummy behavior takes as many arguments as there are input places of the new transition
                dummy_behavior = self._make_dummy_behavior(len(new_inflow))

                new_tr = expanded_pn.add_action(new_inflow, new_outflow, behavior=dummy_behavior, name=f"{t._id}.{index_new_action}")
                for el in new_tr.incoming:
                    expanded_pn.arcs.append((el, new_tr))
                for el in new_tr.outgoing:
                    expanded_pn.arcs.append((new_tr, el))

        return expanded_pn, transition_binding_map

    def _make_dummy_behavior(self, num_args):
        # Create a dummy function with the correct number of arguments
        arg_list = ', '.join([f'arg{i}' for i in range(num_args)])
        func_code = f"def dummy_behavior({arg_list}):\n    return [None] * {num_args}"
        local_vars = {}
        exec(func_code, {}, local_vars)
        return local_vars['dummy_behavior']

    def expand(self):
        """
        Expands the attributed A-E PN into a 1-bounded attributed A-E PN.

        Each token in the original problem is mapped to a place in the expanded problem, and transitions are expanded
        to account for all possible token combinations.

        :return: A tuple containing:
            - The expanded `GymProblem` instance.
            - A mapping of transitions to their bindings in the expanded problem.
        """
        expanded_pn = GymProblem(has_var_attrs=True)

        expanded_pn.arcs = [] # additional helper parameter to reconstruct the assignment graph edges
        in_cartesian_product = [] # stores the new transition inflow according to the a-e pn expansion mechanism
        out_cartesian_product = [] # stores the new transition outflow according to the a-e pn expansion mechanism

        for p in self.places:
            if len(p.marking):
                for i, t in enumerate(
                        p.marking):  # for each token in the place, create a 1-bounded place with the token inside
                    new_place_id = p._id + '.' + str(i)
                    new_place = expanded_pn.add_var(name=new_place_id, var_attributes=self.var_attributes[p._id])
                    new_place.put(t.value, time=t.time)
            else:  # if there is no token in the place, we keep the original place (necessary to respect the inflows and outflows)
                new_place_id = p._id + '.' + str(0)
                expanded_pn.add_var(name=new_place_id, var_attributes=self.var_attributes[p._id])

        for index, t in enumerate(self.events):
            new_inflow = []  # stores the new place inflow according to the a-e pn expansion mechanism
            for p in t.incoming:
                if type(p) is SimVar:
                    if len(p.marking):
                        for i in range(len(p.marking)):
                            new_inflow.append(expanded_pn.id2node[p._id + '.' + str(i)])
                            expanded_pn.arcs.append((expanded_pn.id2node[p._id + '.' + str(i)], t))
                    else:
                        new_inflow.append(expanded_pn.id2node[p._id + '.' + str(0)])
                        expanded_pn.arcs.append((expanded_pn.id2node[p._id + '.' + str(0)], t))
                elif type(p) is SimVarQueue: #SimVarQueues should remain the same (they are not used by the expansion)
                    new_inflow.append(p)

            new_outflow = []
            for p in t.outgoing:
                if type(p) is SimVar:
                    if len(p.marking):
                        for i in range(len(p.marking)):
                            new_outflow.append(expanded_pn.id2node[p._id + '.' + str(i)])
                            expanded_pn.arcs.append((t, expanded_pn.id2node[p._id + '.' + str(i)]))
                    else:
                        new_outflow.append(expanded_pn.id2node[p._id + '.' + str(0)])
                        expanded_pn.arcs.append((t, expanded_pn.id2node[p._id + '.' + str(0)]))
                elif type(p) is SimVarQueue:
                    new_outflow.append(p)

            # E transitions are not expanded (only the incoming/outgoing arcs are adapted to match the new network topology)
            expanded_pn.add_event(new_inflow, new_outflow, behavior=t.behavior, guard=t.guard, name=t._id)

        transition_binding_map = []
        for index_t, t in enumerate(self.actions):
            # action transitions are expanded into n transitions where n is the number of possible tokens' associations from connected places
            transition_bindings = self.event_bindings(t)  # operated on the non-expanded network

            if len(transition_bindings) == 0:
                continue
            for index_b, b in enumerate(
                    transition_bindings):  # a new transition is created for every combination in the bindings

                transition_binding_map.append([b[0], b[1], t])
                new_inflows = {}  # stores the new transition inflow according to the a-e pn expansion mechanism
                new_outflows = {}

                for index_c, c in enumerate(b[0]):  # c is a tuple (SimVar, SimToken)
                    original_place, original_token = c

                    # to get the new place id we check if the markings are the same (NOT ENOUGH: the same token can be in different places)
                    # instead we check if the place id is the same (i.e. the token is in the same place)

                    for new_place in [pl for pl in expanded_pn.places if
                                      GymProblem._get_string_before_last_dot(pl._id) == original_place._id]:
                        #for token_index, token in enumerate(new_place.marking):
                        if original_place._id not in new_inflows.keys():
                            new_inflows[original_place._id] = [new_place]
                        else:
                            new_inflows[original_place._id].append(new_place)

                for p in t.outgoing:
                    if len(p.marking):
                        for i in range(len(p.marking)):
                            if p._id not in new_outflows.keys():
                                new_outflows[p._id] = [expanded_pn.id2node[p._id + '.' + str(i)]]
                            else:
                                new_outflows[p._id].append(expanded_pn.id2node[p._id + '.' + str(i)])
                    else:
                        new_outflows[p._id] = [expanded_pn.id2node[p._id + '.' + str(0)]]

                #generate the combination of size len(b[0]) for the inflow and outflow
                in_cartesian_product = [list(combination) for combination in itertools.product(*new_inflows.values())]
                out_cartesian_product = list(new_outflows.values())[0]#[list(combination) for combination in itertools.product(*new_outflows.values())]

            #create the new transitions
            for index_new_action, io in enumerate(in_cartesian_product):
                new_inflow = io
                new_outflow = out_cartesian_product
                new_tr = expanded_pn.add_action(new_inflow, new_outflow, behavior=t.behavior, guard=t.guard,
                                               name=t._id + '.' + str(index_new_action))

                for el in new_tr.incoming:
                    expanded_pn.arcs.append((el, new_tr))
                for el in new_tr.outgoing:
                    expanded_pn.arcs.append((new_tr, el))

        return expanded_pn, transition_binding_map

    def minimal_graph(self, graph):
        """
        Generates a minimal graph observation by removing all nodes that are not attached to 'a_transition'.

        :param graph: The original graph observation as a `HeteroData` object.
        :return: The updated `HeteroData` object with only 'a_transition' nodes and those directly attached to them.
        """
        # Identify 'a_transition' nodes
        if 'a_transition' not in graph.node_types: #likely at the end of an episode
            #print("'a_transition' node type not found in the graph.")
            #The final state does not have to contain actions (the episode is over)
            return graph
        new_ret_graph = HeteroData()

        #a_transition_nodes = set(range(graph['a_transition'].x.size(0)))

        # Create a new graph with only 'a_transition'
        new_ret_graph['a_transition'].x = graph['a_transition'].x

        # Find nodes directly connected to 'a_transition'
        connected_nodes = list()
        for edge_type in graph.edge_types:
            src_type, _, dst_type = edge_type
            if dst_type == 'a_transition':
                new_ret_graph[edge_type].edge_index = graph[edge_type].edge_index

                if dst_type == 'a_transition':
                    connected_nodes.append(src_type)

        for n in connected_nodes:
            if n not in new_ret_graph.node_types:
                new_ret_graph[n].x = graph[n].x

        # Include 'a_transition' nodes in the connected nodes
        #connected_nodes.update(a_transition_nodes)

        return new_ret_graph


    @staticmethod
    def _get_string_before_last_dot(s: str) -> str:
        """
        Extracts the substring before the last dot in a string.

        :param s: The input string.
        :return: The substring before the last dot.
        """
        last_dot_index = s.rfind('.')
        if last_dot_index == -1:
            return s  # Return the whole string if no dot is found
        return s[:last_dot_index]

    @staticmethod
    def tokens_combinations(event):
        """
        Creates a list of token combinations that are available to the specified event.
        These are combinations of tokens that are on the incoming SimVar of the event.
        For example, if a event has incoming SimVar a and b with tokens 1@0 on a and 2@0, 3@0 on b,
        the possible combinations are [(a, 1@0), (b, 2@0)] and [(a, 1@0), (b, 3@0)]

        :param event: the event to return the token combinations for.
        :return: a list of lists, each of which is a token combination.
        """
        # create all possible combinations of incoming token values
        bindings = [[]]
        for place in event.incoming:
            new_bindings = []
            for token in place.marking:  # get set of colors in incoming place
                for binding in bindings:
                    new_binding = binding.copy()
                    new_binding.append((place, token))
                    new_bindings.append(new_binding)
            bindings = new_bindings
        return bindings

    def event_bindings(self, event):
        """
        Calculates the set of bindings that enables the given event.
        Each binding is a tuple ([(place, token), (place, token), ...], time) that represents a single enabling binding.
        A binding is
        a possible token combination (see token_combinations), for which the event's
        guard function evaluates to True. In case there is no guard function, any combination is also a binding.
        The time is the time at which the latest token is available.
        For example, if a event has incoming SimVar a and b marked with tokens [1@2] on a and [2@3, 3@1] on b,
        the possible bindings are ([(a, 1@2), (b, 2@3)], 3) and ([(a, 1@2), (b, 3@1)], 3)

        :param event: the event for which to calculate the enabling bindings.
        :return: list of tuples ([(place, token), (place, token), ...], time)
        """
        if len(event.incoming) == 0:
            raise Exception("Though it is strictly speaking possible, we do not allow events like '" + str(
                self) + "' without incoming arcs.")

        if self.allow_postpone:
            if type(event) is SimAction:
                if self.just_postponed:
                    return []  # cannot take any action right after postponing

        bindings = self.tokens_combinations(event)

        # a binding must have all incoming places
        nr_incoming_places = len(event.incoming)
        new_bindings = []
        for binding in bindings:
            if len(binding) == nr_incoming_places:
                new_bindings.append(binding)
        bindings = new_bindings

        # if a event has a guard, only bindings are enabled for which the guard evaluates to True
        result = []
        for binding in bindings:
            variable_values = []
            time = None
            for (place, token) in binding:
                variable_values.append(token.value)
                if time is None or token.time > time:
                    time = token.time
            enabled = True
            if event.guard is not None:
                try:
                    enabled = event.guard(*variable_values)
                except Exception as e:
                    raise TypeError("Event " + event + ": guard generates exception for values " + str(
                        variable_values) + ".") from e
                if self._debugging and not isinstance(enabled, bool):
                    raise TypeError("Event " + event + ": guard does evaluate to a Boolean for values " + str(
                        variable_values) + ".")
            if enabled:
                result.append((binding, time))
        return result

    def get_to_first_action(self):
        """
        Runs the petri net until it gets to 'a' tag (initialization for mdp environment)
        """
        #bindings, active_model = self.bindings()

        while self.network_tag.is_evolution():
            bindings, active_model = self.bindings()
            if self.network_tag.is_action(): #the bindings() function controls the evolution of self.tag, so we need to break the cicle as soon as tag == 'a'
                break
            #same as step() in SimProblem
            if len(bindings) > 0:
                timed_binding = self.binding_priority(bindings)
                result_tokens = self.fire(timed_binding)
                #if timed_binding[-1]._id in self.reward_functions.keys():
                self.update_reward(timed_binding, result_tokens)
                #return timed_binding
            else:
                raise Exception("Invalid initial state for the network")
        return self


    def run_evolutions(self, run, i, active_model):
        """
        Function invoked by Gym environment to let the network perform evolutions when no actions are required
        """

        while self.clock <= self.length and active_model:
            bindings, active_model = self.bindings()
            # Finite-horizon premise: NO transition may fire past self.length.
            # bindings() may advance the clock to the next scheduled event; if that
            # event lies beyond the horizon, the episode ends WITHOUT firing it
            # (work not completed by the deadline yields no reward). This keeps the
            # simulated return well-defined as "reward accrued within [0, length]".
            if self.clock > self.length:
                active_model = False
                break

            if len(bindings) > 0 and self.network_tag.is_evolution():
                self.just_postponed = False
                binding = random.choice(bindings)
                run.append(binding)
                result_tokens = self.fire(binding)
                self.update_reward(binding, result_tokens)
                i += 1

            elif len(bindings) > 0 and self.network_tag.is_action():  # hand control to the gym env
                if not self.allow_postpone:
                    condition = True if self.causal_rl else len(bindings) > 1
                    if condition:  # only consult the agent when there is a real choice
                        return self.get_graph_observation(), self.clock > self.length or not active_model, i
                    else:
                        binding = bindings[0]
                        run.append(binding)
                        result_tokens = self.fire(binding)
                        self.update_reward(binding, result_tokens)
                        i += 1
                else:
                    # allow_postpone: the agent may also choose to postpone.
                    return self.get_graph_observation(), self.clock > self.length or not active_model, i
            else:
                active_model = False

        return self.get_graph_observation(), self.clock > self.length or not active_model, i

    def update_reward(self, timed_binding, result_tokens=None):
        binding, time, transition = timed_binding
        variable_values = []

        if transition._id in self.reward_functions:
            for (place, token) in binding:
                variable_values.append(token.value)
            try:
                r_f = self.reward_functions[transition._id](*variable_values)
            except Exception as e:
                raise TypeError(
                    "Transition " + transition._id + ": reward function generates exception for values " + str(
                        variable_values) + ".") from e
            if self._debugging and not isinstance(r_f, (float, int)):
                raise TypeError(
                    "Transition " + transition._id + ": reward function evaluate to a non-numeric type for values " + str(
                        variable_values) + ".")
            self.reward += r_f

        else:
            r_f = 0

        #print(f"produced reward {r_f} with binding {binding}")
        #update causal reward buffer if enabled
        if self.causal_rl:
            self.update_causal_trace(binding, result_tokens, r_f, transition)


        return r_f, variable_values


    def make_metadata(self, add_self_loops=True, add_action_to_action=True, add_reverse_edges=False):
        """
        Construct PyG HeteroData metadata: (node_types, edge_type_tuples).

        New optional flags:
        - add_action_to_action: include a generic ('a_transition','a_to_a','a_transition') metapath
          that connects action nodes (useful to let the GNN pass messages directly between
          different action nodes such as stage A and stage B).
        - add_reverse_edges: add reverse edge types for each discovered edge type so message
          passing can flow bidirectionally (each (u,rel,v) adds (v, rel+'_rev', u)).
        """
        nodes_meta = ['e_transition', 'a_transition']
        edges_meta = []
        if add_self_loops:
            edges_meta.append(('a_transition', 'self_loop', 'a_transition'))

        for p in self.places:
            nodes_meta.append(p._id)
        for e in self.events:
            for inc in e.incoming:
                if type(inc) is SimVar:
                    e_m = (inc._id, 'edge_to_e_transition', 'e_transition')
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)
            for out in e.outgoing:
                if type(out) is SimVar:
                    e_m = ('e_transition', 'edge_to_' + out._id, out._id)
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)
        for a in self.actions:
            for inc in a.incoming:
                if type(inc) is SimVar:
                    e_m = (inc._id, 'edge_to_a_transition', 'a_transition')
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)
            for out in a.outgoing:
                if type(out) is SimVar:
                    e_m = ('a_transition', 'edge_to_' + out._id, out._id)
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)

        if self.allow_postpone:
            nodes_meta.append('postpone')
            for n in nodes_meta:
                if n != 'postpone':
                    edges_meta.append((n, 'to_postpone', 'postpone'))

        # Optionally add a generic action->action metapath to connect action nodes
        # (helps message passing across different action types such as start_A -> start_B)
        if add_action_to_action:
            a2a = ('a_transition', 'a_to_a', 'a_transition')
            if a2a not in edges_meta:
                edges_meta.append(a2a)

        # Optionally add reverse edge types for all discovered edge types so the GNN
        # can pass messages bidirectionally without relying on the conv implementation
        # to implicitly handle reverse relations.
        if add_reverse_edges:
            # Gather existing edge types snapshot and add reverse counterparts when missing
            existing = list(edges_meta)
            for (src, rel, dst) in existing:
                rev = (dst, rel + '_rev', src)
                if rev not in edges_meta:
                    edges_meta.append(rev)

        return tuple([nodes_meta, edges_meta])


    def old_make_metadata(self, add_self_loops=True):
        nodes_meta = ['e_transition', 'a_transition']
        edges_meta = []
        if add_self_loops:
            edges_meta.append(('a_transition', 'edge', 'a_transition'))

        for p in self.places:
            nodes_meta.append(p._id)
        for e in self.events:
            for inc in e.incoming:
                if type(inc) is SimVar:
                    e_m = (inc._id, 'edge', 'e_transition')
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)
            for out in e.outgoing:
                if type(out) is SimVar:
                    e_m = ('e_transition', 'edge', out._id)
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)
        for a in self.actions:
            for inc in a.incoming:
                if type(inc) is SimVar:
                    e_m = (inc._id, 'edge', 'a_transition')
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)
            for out in a.outgoing:
                if type(out) is SimVar:
                    e_m = ('a_transition', 'edge', out._id)
                    if e_m not in edges_meta:
                        edges_meta.append(e_m)

        if self.allow_postpone:
            nodes_meta.append('postpone')
            for n in nodes_meta:
                if n != 'postpone':
                    edges_meta.append((n, f'to_postpone_from_{n}', 'postpone'))

        return tuple([nodes_meta, edges_meta])

    def training_run(self, length = 10, args_dict=None):
        """
        Runs the petri net as a reinforcement learning environment.
        :param length: the length of the run.
        :param args_dict: a dictionary with the arguments to be used for the run. The possible keys are:

        Algorithm Parameters:
        - `algorithm` (str): The training algorithm to use. Choices are `['ppo-clip', 'ppo-penalty', 'pg']`. Default: `'ppo-clip'`.
        - `gam` (float): Discount rate. Default: `0.99`.
        - `lam` (float): Generalized advantage parameter. Default: `0.99`.
        - `eps` (float): Clip ratio for clipped PPO. Default: `0.2`.
        - `c` (float): KLD weight for penalty PPO. Default: `0.2`.
        - `ent_bonus` (float): Bonus factor for sampled policy entropy. Default: `0.0`.
        - `vf_coeff` (float): Value function loss coefficient in total loss. Default: `0.5`.
        - `agent_seed` (int or None): Seed for the agent. Default: `None`.

        Policy Model:
        - `policy_model` (str): The type of policy network. Choices are `['gnn']`. Default: `'gnn'`.
        - `policy_kwargs` (dict): Arguments to the policy model constructor, passed through `json.loads`. Default: `{"hidden_layers": [64]}`.
        - `policy_lr` (float): Policy model learning rate. Default: `3e-4`.
        - `policy_updates` (int): Number of policy model updates per epoch. Default: `2`.
        - `policy_kld_limit` (float or None): KL divergence limit for early stopping. Default: `None` (disabled — not recommended for variable action sets).
        - `policy_weights` (str): Filename for initial policy weights. Default: `""` (empty string).
        - `policy_network` (str): Filename for initial policy network. Default: `""` (empty string).
        - `score` (bool): Whether to have multi-objective training. Default: `False`.
        - `score_weight` (float): Weight gradients of L2 loss. Default: `1e-3`.

        Value Model:
        - `value_model` (str): The type of value network. Choices are `['none', 'gnn']`. Default: `'gnn'`.
        - `value_kwargs` (dict): Arguments to the value model constructor, passed through `json.loads`. Default: `{"hidden_layers": [64]}`.
        - `value_lr` (float): Value model learning rate. Default: `3e-4`.
        - `value_updates` (int): Number of value model updates per epoch. Default: `10`.
        - `value_weights` (str): Filename for initial value weights. Default: `""` (empty string)`.

        Training Parameters:
        - `episodes` (int): Number of episodes per epoch. Default: `10`.
        - `epochs` (int): Number of epochs. Default: `100`.
        - `max_episode_length` (int or None): Maximum number of interactions per episode. Default: `None`.
        - `batch_size` (int or None): Size of batches in training. Default: `64`.
        - `sort_states` (bool): Whether to sort the states before batching. Default: `False`.
        - `use_gpu` (bool): Whether to use a GPU if available. Default: `False`.
        - `load_policy_network` (bool): Whether to load a previously trained policy as a starting point for this run. Default: `False`.
        - `verbose` (int): How much information to print. Default: `0`.

        Saving Parameters:
        - `name` (str): Name of the training run. Default: `'run'`.
        - `datetag` (bool): Whether to append the current time to the run name. Default: `False`.
        - `logdir` (str): Base directory for training runs. Default: `'data/train'`.
        - `save_freq` (int): How often to save the models. Default: `1`.
        - 'use_wandb' (bool): Whether to use Weights & Biases for logging. Default: `True`.
        - 'wandb_mode' (str): W&B mode: 'online' (cloud), 'offline' (local), or 'disabled'. Default: `'offline'`.
        """
        self.length = length
        args = make_parser().parse_args()

        if args_dict is not None:
            try:
                for key, value in args_dict.items():
                    if hasattr(args, key):
                        setattr(args, key, value)
                    else:
                        raise Exception(f"Argument {key} not recognized")
            except Exception as e:
                raise Exception(f"The provided arguments are not valid: {e}")

        if not args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if args.agent_seed is not None:
            np.random.seed(args.agent_seed)
            random.seed(args.agent_seed)
            torch.manual_seed(args.agent_seed)
            torch.cuda.manual_seed(args.agent_seed)
            # TODO: two more lines for cuda

        #EXPERIMENTAL: if causal_rl, every token in the network is complemented with a unique identifier
        if self.causal_rl:
                new_unobservable_dict = {'places': [], }
                #TODO: update the unobservable elements (probably not necessary cause it is not registered in the place's variables) AND check that no token had _id in its attributes
                #self.set_unobservable(simvars=[], token_attrs={'EVERY_PLACE': ['_id']})
                for place in self.places:
                    #include _id in the set of unobservable features
                    for token in place.marking:
                        setattr(token, '_id', str(uuid.uuid4()))
                print("Causal RL enabled: each token has been assigned a unique identifier.")
                # Ensure causal trace starts empty at the beginning of training
                try:
                    self.causal_trace.flush()
                except Exception:
                    pass

                # Register initial tokens as roots in the causal trace.
                # This ensures the BFS in redistribute_rewards() can find them
                # as proper root nodes instead of encountering unknown token IDs.
                import types
                initial_sentinel = types.SimpleNamespace(_id="__initial__")
                for place in self.places:
                    for token in place.marking:
                        self.causal_trace.register_token(
                            token, initial_sentinel, parent_tokens=[], time=0
                        )
                self.causal_trace.register_transition(
                    transition=initial_sentinel,
                    input_tokens=[],
                    output_tokens=[t for p in self.places for t in p.marking],
                    is_action=False,
                    reward=0.0,
                    time=0
                )

        env = AEPN_Env(self)

        if args.test_in_train:
            test_env = copy.deepcopy(env)
            test_freq = args.test_freq
        else:
            test_env = None
            test_freq = None

        # nodes_list, edges_list = self.get_nodes_edges_types()
        metadata = self.make_metadata()

        agent = make_agent(args, metadata=metadata)

        logdir = make_logdir(args)

        # Import logger for training output
        from gympn.logging_utils import get_logger
        logger = get_logger(verbose=args.verbose)

        logger.debug(f"Saving run in {logdir}")

        # Launch TensorBoard if requested
        tb_process = None
        if args.open_tensorboard:
            logger.info("Launching TensorBoard...")
            tb_process = launch_tensorboard(logdir)
            if tb_process is None:
                logger.warning("TensorBoard launch failed, continuing without it")

        # Initialize W&B logger if requested
        wandb_logger = None
        if args.use_wandb:
            try:
                from gympn.wandb_integration import init_wandb

                # Prepare config for W&B
                config = {
                    'algorithm': args.algorithm,
                    'episodes': args.episodes,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'policy_lr': args.policy_lr,
                    'value_lr': args.value_lr,
                    'gam': args.gam,
                    'lam': args.lam,
                    'eps': args.eps,
                    'vf_coeff': args.vf_coeff,
                    'causal_rl': args.causal_rl,
                }

                wandb_logger = init_wandb(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    run_name=os.path.basename(logdir),
                    config=config,
                    mode=args.wandb_mode,
                    open_dashboard=args.open_wandb,  # Use the open_wandb argument
                    dashboard_port=8080
                )
                logger.info("Weights & Biases logger initialized")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
            except Exception as e:
                logger.warning(f"Could not initialize W&B: {e}")

        logger.training_start(agent.__class__.__name__, args.epochs, args.episodes)

        # Log if causal RL is enabled
        if hasattr(env, 'causal_rl') and env.causal_rl:
            logger.causal_rl_enabled()

        history = agent.train(env, episodes=args.episodes, epochs=args.epochs,
                    save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                    max_episode_length=args.max_episode_length, batch_size=args.batch_size,
                    test_env=test_env, test_freq=test_freq, wandb_logger=wandb_logger)

        # Store training history for programmatic access
        self.training_history = history if history is not None else {}

        # Report best metric: prefer test metric, fall back to best training return
        best_metric = getattr(agent, 'best_test_metric', float('-inf'))
        if best_metric == float('-inf') and history is not None and 'mean_returns' in history:
            best_metric = float(np.max(history['mean_returns']))
        logger.training_end(best_metric)

        # Finish W&B run
        if wandb_logger:
            try:
                wandb_logger.log_summary({
                    'best_return': agent.best_test_metric if hasattr(agent, 'best_test_metric') else 0.0,
                })
                wandb_logger.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"Could not finish W&B run: {e}")

        # Clean up TensorBoard process
        if tb_process is not None:
            try:
                logger.info("Stopping TensorBoard...")
                tb_process.terminate()
                tb_process.wait(timeout=5)
                logger.info("TensorBoard stopped")
            except subprocess.TimeoutExpired:
                logger.warning("TensorBoard did not stop gracefully, killing process...")
                tb_process.kill()
                tb_process.wait()
            except Exception as e:
                logger.warning(f"Error stopping TensorBoard: {e}")

    def _augment_bindings_with_postpone(self, bindings: List) -> List:
        """
        If postpone is allowed, we are in action phase, and we didn't just postpone,
        expose postpone as a pseudo-binding option:
            (['postpone'], current_clock, None)

        Ensures we don't add duplicates.
        """
        if not self.allow_postpone or not self.network_tag.is_action():
            return bindings

        already_present = any(isinstance(b, tuple) and b and b[0] == ['postpone'] for b in bindings)
        if already_present:
            return bindings

        augmented = list(bindings)
        augmented.append((['postpone'], self.clock, None))
        return augmented

    def step(self, reporter=None, length=None):
        """
        Executes one simulation step.

        Evolution phase:
            - Fire one evolution binding (as before).

        Action phase:
            - GymSolver: use policy over graph observation (postpone already included).
            - Other solvers: expose 'postpone' via a pseudo-binding and pass the augmented list to the solver.
              Solver may return:
                * 'postpone' (string)
                * the postpone pseudo-binding tuple
                * a real timed binding from the augmented list
        """
        if self.solver is None:
            print("Warning: no solver was provided. Use set_solver(...) or RandomSolver for simple simulation.")

        bindings, active_model = self.bindings()

        if self.clock <= self.length:
            # -------- EVOLUTIONS --------
            if len(bindings) > 0 and self.network_tag.is_evolution():
                timed_binding = bindings[0]
                output_tokens = self.fire(timed_binding)
                if timed_binding[-1]._id in self.reward_functions.keys():
                    self.update_reward(timed_binding)
                if reporter is not None:
                    self.print_report(reporter, timed_binding)
                print(f"Fired binding {timed_binding}")
                return timed_binding, active_model

            # -------- ACTIONS --------
            elif len(bindings) > 0 and self.network_tag.is_action():
                # --- Gym branch ---
                if isinstance(self.solver, GymSolver):
                    obs = self.get_graph_observation()
                    act_probs = self.solver.solve(obs)
                    max_index = torch.argmax(act_probs).item()
                    timed_binding = obs['actions_dict'][max_index]

                    if self.allow_postpone and isinstance(timed_binding, tuple) and timed_binding[0] == ['postpone']:
                        self.postpone()
                        print("Postponed!")
                    else:
                        output_tokens = self.fire(timed_binding)
                        print(f"Fired binding {timed_binding}")
                        if timed_binding[-1]._id in self.reward_functions.keys():
                            self.update_reward(timed_binding)
                        if reporter is not None:
                            self.print_report(reporter, timed_binding)
                    return timed_binding, active_model

                # --- Heuristic / Random branch ---
                else:
                    obs = self.get_heuristic_observation()
                    aug_bindings = self._augment_bindings_with_postpone(bindings)
                    timed_binding = self.solver.solve(obs, aug_bindings)

                    postponed = False
                    if timed_binding == 'postpone' or (isinstance(timed_binding, tuple) and timed_binding[0] == ['postpone']):
                        postponed = True
                        timed_binding = (['postpone'], self.clock, None)

                    if postponed:
                        self.postpone()
                        print("Postponed!")
                        #return timed_binding, active_model
                    else:
                        # Fire selected binding
                        self.fire(timed_binding)
                        if timed_binding[-1]._id in self.reward_functions.keys():
                            self.update_reward(timed_binding)
                        if reporter is not None:
                            self.print_report(reporter, timed_binding)
                        print(f"Fired binding {timed_binding}")
                    return timed_binding, active_model

        return None, active_model

    def set_solver(self, solver):
        """
        Sets the solver for the problem.

        The solver determines which action to execute when multiple actions are enabled.

        :param solver: An instance of a solver class implementing the `BaseSolver` interface.
        """
        self.solver = solver

    def testing_run(self, solver, length=10, reporter=None, visualize=False):
        """
        Test run aligned with AEPN_Env semantics:
          1) Ensure we start in ACTION phase (like env.reset()).
          2) Loop:
               - Choose exactly ONE action (or postpone).
               - Apply it (fire or postpone).
               - Run all evolutions: obs, terminated, _ = run_evolutions(...)
               - Stop when 'terminated' is True (same as AEPN_Env.step).
          3) Return total accumulated reward (same scalar your env exposes in info['pn_reward']).
        """
        # ---- Setup identical to environment.reset() ----
        self.length = length
        if visualize:
            # Keep your existing visualization path if needed
            from gympn.visualisation import Visualisation
            Visualisation(self).show()
            return self.reward

        # Ensure we are at the first action (env.reset does this)
        if self.network_tag.is_evolution():
            self.get_to_first_action()  # moves clock/tag to first action when possible  [1](https://tuenl-my.sharepoint.com/personal/r_lo_bianco_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/environment%20-%20Copy.txt)

        # Local "run" log & cursor, like AEPN_Env keeps (not strictly needed, but harmless)
        run_log, i = [], 0

        # Helper to pick & apply exactly one action (or postpone)
        def _apply_one_action_gymsolver():
            """GymSolver branch (policy model). Returns True if an action/postpone was applied."""
            import torch
            obs = self.get_graph_observation(normalize=False)  # keep features stationary like training
            actions = obs.get('actions_dict', [])
            if not actions:
                return False
            # Policy returns per-action logits/probs; pick argmax deterministically for testing.
            probs = solver.solve(
                obs)  # GymSolver.solve calls policy.forward(obs)  [1](https://tuenl-my.sharepoint.com/personal/r_lo_bianco_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/environment%20-%20Copy.txt)
            if isinstance(probs, torch.Tensor) and probs.dim() > 1 and probs.size(-1) == 1:
                probs = probs.squeeze(-1)
            a_idx = int(torch.argmax(probs).item()) if isinstance(probs, torch.Tensor) else int(probs)
            a_idx = max(0, min(a_idx, len(actions) - 1))
            chosen = actions[a_idx]

            # Postpone option (pseudo-binding): (['postpone'], current_clock, None)
            if isinstance(chosen, tuple) and chosen and chosen[0] == ['postpone']:
                if self.allow_postpone:
                    self.postpone()
                    self.just_postponed = True
                    if reporter is not None:
                        # Optional: reporter callback for postpone sentinel (no timed binding)
                        try:
                            reporter.callback(chosen)
                        except Exception:
                            pass
                    return True
                # If postpone is NOT allowed, fallback to first real binding if any
                real = [b for b in actions if isinstance(b, tuple) and b and isinstance(b[0], list)]
                if real:
                    chosen = real[0]
                else:
                    return False

            # Fire the real binding
            self.just_postponed = False
            result_tokens = self.fire(
                chosen)  # applies behavior and places tokens  [1](https://tuenl-my.sharepoint.com/personal/r_lo_bianco_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/environment%20-%20Copy.txt)
            self.update_reward(chosen,
                               result_tokens)  # accumulate reward (non-causal mode uses deltas)  [1](https://tuenl-my.sharepoint.com/personal/r_lo_bianco_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/environment%20-%20Copy.txt)
            if reporter is not None:
                try:
                    reporter.callback(chosen)
                except Exception:
                    pass
            # Refresh tag like env.step does before run_evolutions
            self.bindings()
            return True

        def _apply_one_action_heuristic_like():
            """Heuristic/Random branch. Returns True if applied."""
            # Expect to be in ACTION phase here; get current action bindings
            bindings, _active = self.bindings()
            if not bindings:
                return False

            # Expose postpone as a pseudo-binding if configured
            aug = self._augment_bindings_with_postpone(bindings)

            # Heuristic solver API: solve(observable_net, bindings)
            try:
                selection = solver.solve(self.get_heuristic_observation(), aug)
            except TypeError:
                # Some heuristics take (obs, tokens_comb) in your codebase; fallback to postpone-first real binding
                selection = aug[0] if aug else None

            if selection is None:
                return False

            # Handle 'postpone' sentinel in the same way as env
            if selection == 'postpone' or (isinstance(selection, tuple) and selection and selection[0] == ['postpone']):
                if self.allow_postpone:
                    self.postpone()
                    self.just_postponed = True
                    if reporter is not None:
                        try:
                            reporter.callback((['postpone'], self.clock, None))
                        except Exception:
                            pass
                    # Refresh tag before evolutions (match env.step)
                    self.bindings()
                    return True
                # fell through: if postpone not allowed, try to fire a real one
                real = [b for b in aug if isinstance(b, tuple) and b and isinstance(b[0], list)]
                if not real:
                    return False
                selection = real[0]

            # If selection is an index
            if isinstance(selection, int):
                if selection < 0 or selection >= len(aug):
                    return False
                selection = aug[selection]

            # selection should now be a real timed binding
            if not (isinstance(selection, tuple) and selection and isinstance(selection[0], list)):
                return False

            self.just_postponed = False
            result_tokens = self.fire(selection)
            self.update_reward(selection, result_tokens)
            if reporter is not None:
                try:
                    reporter.callback(selection)
                except Exception:
                    pass
            # Refresh tag like env.step before run_evolutions
            self.bindings()
            return True

        # ------------------ Main loop (env-aligned) ------------------
        terminated = False
        while not terminated:
            # Choose & apply ONE action/postpone
            applied = False
            try:
                from gympn.solvers import GymSolver
                if isinstance(solver, GymSolver):
                    applied = _apply_one_action_gymsolver()
                else:
                    applied = _apply_one_action_heuristic_like()
            except Exception:
                # If solver type cannot be imported/checked, try policy path first
                applied = _apply_one_action_gymsolver()

            # If nothing could be applied, we still need to check evolutions once to determine termination
            # (e.g., no action bindings but evolutions remain)
            # Run evolutions exactly like AEPN_Env.step does:
            _obs_after, terminated, i = self.run_evolutions(run_log, i,
                                                            active_model=True)  # returns terminated flag  [1](https://tuenl-my.sharepoint.com/personal/r_lo_bianco_tue_nl/Documents/Microsoft%20Copilot%20Chat%20Files/environment%20-%20Copy.txt)

            # If we are over the horizon or inactive, stop (this mirrors AEPN_Env)
            if terminated:
                break

        return self.reward

    def set_unobservable(self, simvars=None, token_attrs=None):
        """
        Defines the set of unobservable simulation variables and token attributes.

        Unobservable variables and attributes are excluded from observations and graph representations.

        :param simvars: A list of SimVar IDs that are unobservable.
        :param token_attrs: A dictionary mapping SimVar IDs to lists of unobservable token attribute names.
        """

        if token_attrs is None:
            token_attrs = {}
        if simvars is None:
            simvars = []
        places_id = [el._id for el in self.places]

        if type(simvars) is not list:
            raise Exception("Unobservable SimVars must be a list of SimVars ids")
        if type(token_attrs) is not dict:
            raise Exception("Unobservable token attributes must be a dictionary with SimVar ids as keys and lists of token attributes names as values")

        self.unobservable_simvars = simvars
        if not set(simvars) <= set(places_id):
            raise Exception("Unobservable SimVars must be in the list of places")

        if len(token_attrs) > 0:
            if not all(isinstance(v, list) for v in token_attrs.values()):
                raise Exception("Unobservable token attributes must be a dictionary with SimVar ids as keys and lists of token attributes names as values")
            if not all(isinstance(k, str) for k in token_attrs.keys()):
                raise Exception("Unobservable token attributes must be a dictionary with SimVar ids as keys and lists of token attributes names as values")
        self.unobservable_token_attrs = token_attrs
        for k, v in self.unobservable_token_attrs.items():
            if k not in places_id:
                raise Exception(f"Exception in setting unobservable token attrs. Unobservable SimVar {k} must be in the list of places.")
            for attr in v:
                if attr not in self.var_attributes[k]:
                    raise Exception(f"Exception in setting unobservable token attrs. Unobservable attribute {attr} must be present in the place's attributes.")

    def postpone(self):
        """
        Postpones the current action by advancing the clock to the next valid evolution event.
        Also registers all tokens that were available for action bindings but were not used
        because of the postpone into the eligibility trace with zero immediate reward.
        """
        if not self.allow_postpone:
            raise Exception("Postpone is not allowed in this A-E PN.")

        if self.causal_rl:
            # collect current enabled bindings (timed bindings: (binding, time, transition))
            bindings, _ = self.bindings()
            self.update_causal_trace_postpone(bindings)

        # perform original postpone behavior
        self.network_tag.tag = 'e'
        self.just_postponed = True

    def update_causal_trace_postpone(self, bindings):
        if not self.causal_rl:
            return

        if self.causal_postpone_tokenflow:
            self._update_causal_trace_postpone_tokenflow(bindings)
        else:
            self._update_causal_trace_postpone_sinkless(bindings)

    def _update_causal_trace_postpone_sinkless(self, bindings):
        # B.4 fix (see CAUSAL_RL_REDISTRIBUTION_PROBLEMS.md #3): postpone must NOT
        # re-ID the eligible tokens nor insert itself into the token lineage.
        # The old behavior gave every eligible token a fresh id with a
        # `postpone_` sentinel as its creating transition, which made postpone a
        # graph-ancestor of those tokens — so any downstream reward consuming them
        # leaked credit back to the no-op postpone. We now leave tokens untouched,
        # so credit flows *through* postpone to the real upstream producers.
        #
        # We still register a postpone sentinel *transition* (is_action=True,
        # reward 0, no output tokens) so it occupies exactly one action slot,
        # preserving the 1:1 credit/step alignment that TrajectoryBuffer.finish()
        # asserts. With no output tokens it can never be a credit sink under any
        # scheme, so it always receives zero redistributed reward.
        input_tokens = []
        for timed_binding in bindings:
            try:
                binding, _, transition = timed_binding
            except Exception:
                continue

            if isinstance(transition, SimAction):
                for place, token in binding:
                    if token not in input_tokens:
                        input_tokens.append(token)

        # Create a sentinel transition object with a unique _id
        # (TransitionHistory.add_transition accesses transition._id).
        import types
        sentinel_transition = types.SimpleNamespace(_id=f"postpone_{uuid.uuid4()}")

        # Record the eligible tokens (by their current, unchanged _id) as inputs
        # for diagnostics; no output tokens are produced and no token lineage is
        # mutated.
        self.causal_trace.register_transition(
            transition=sentinel_transition,
            input_tokens=input_tokens,
            output_tokens=[],
            is_action=True,
            reward=0.0,
            time=self.clock
        )

    def _update_causal_trace_postpone_tokenflow(self, bindings):
        # Token-flow postpone (Design C, see CAUSAL_ADVANTAGE_TD0.md): model
        # postpone as a real transition that CONSUMES the action-eligible tokens
        # and RECREATES them in place (same token object, fresh _id), with the
        # postpone sentinel as their producer. This inserts postpone into the
        # token DAG (old token -> postpone -> new token), so credit can flow back
        # to it like any action.
        #
        # No deferred re-emission is needed: in an A-E PN the clock is already
        # advanced by postpone() (tag A->E + clock to next event), and action
        # transitions fire at the clock, not at their input tokens' timestamps.
        # Recreating the tokens in place therefore preserves the dynamics exactly
        # and only changes lineage.
        #
        # NOTE: requires redistribute_rewards(include_postpone=True) for the
        # recreated (output) tokens to be treated as credit sinks owned by the
        # postpone action. That is wired via causal_trace.postpone_tokenflow.
        import types

        input_tokens = []        # eligible token objects (post-mutation)
        output_tokens = []       # same objects, now carrying the new _id (sinks)
        old_id_map = {}          # id(token_object) -> old _id (pre-mutation)

        for timed_binding in bindings:
            try:
                binding, _, transition = timed_binding
            except Exception:
                continue

            if isinstance(transition, SimAction):
                for place, token in binding:
                    if token in input_tokens:
                        continue
                    input_tokens.append(token)
                    # Save the pre-postpone id, then assign a fresh id to the
                    # exact object in the marking (identity lookup, NOT value
                    # equality, so duplicate-valued tokens are disambiguated).
                    old_id_map[id(token)] = getattr(token, '_id', None)
                    new_id = str(uuid.uuid4())
                    obj_idx = next(
                        (i for i, t in enumerate(place.marking) if t is token),
                        None
                    )
                    if obj_idx is not None:
                        place.marking[obj_idx]._id = new_id
                    else:
                        token._id = new_id
                    output_tokens.append(token)

        if not input_tokens:
            return

        sentinel_transition = types.SimpleNamespace(_id=f"postpone_{uuid.uuid4()}")

        # Lightweight wrappers carrying the OLD ids, so the transition/token
        # history records old-id inputs -> new-id outputs (postpone as the hop).
        class _OldIdToken:
            def __init__(self, old_id):
                self._id = old_id

        old_input_tokens = [
            _OldIdToken(old_id_map[id(t)]) for t in input_tokens
            if old_id_map.get(id(t)) is not None
        ]

        self.causal_trace.register_transition(
            transition=sentinel_transition,
            input_tokens=old_input_tokens,
            output_tokens=output_tokens,
            is_action=True,
            reward=0.0,
            time=self.clock
        )

        # Register each recreated token (new id) as a child of its old self with
        # the postpone sentinel as the creating transition, so the lineage DAG
        # routes downstream credit through postpone.
        for out_token in output_tokens:
            self.causal_trace.register_token(
                out_token, sentinel_transition, old_input_tokens, time=self.clock
            )

    def update_causal_trace(self, bindings, result_tokens, reward, transition):
        """
        Updates the causal trace with the provided bindings, result tokens, and reward.
        Parameters
        ----------
        bindings
        result_tokens
        reward

        Returns
        -------

        """

        #add the transition to the transition history in causal trace

        consumed_tokens = [t for (p, t) in bindings]

        for p_token in result_tokens:
            #add consumed tokens to the history of produced tokens
            self.causal_trace.register_token(p_token, transition, consumed_tokens, time=self.clock)

        #update transition history with the reward obtained
        self.causal_trace.register_transition(transition, consumed_tokens, result_tokens, is_action=(isinstance(transition, SimAction)), reward=reward, time=self.clock)


class SimAction(SimEvent):
    """
    SimAction is the base class for all actions in A-E PN.
    It serves as base class to SimAssignment (the action decides which specific binding to use for firing), and to
    SimControl, which sets the value of one or more attributes of the tokens resulting from a firing.
    """

    def __init__(self, _id, guard=None, behavior=None, incoming=None, outgoing=None, solver=None):
        """
        Initializes a SimAction with a name, a behavior, and a guard.

        :param _id: the name of the action.
        :param guard: the guard of the action.
        :param behavior: the behavior of the action.
        :param incoming: the incoming SimVar of the action.
        :param outgoing: the outgoing SimVar of the action.

        """
        super().__init__(_id, guard, behavior, incoming, outgoing)
        self.solver = solver
        if self.solver == None:
            self.solver = None


    def execute(self, binding, problem):
        """
        Executes the action for the specified binding.

        The method applies the action's behavior to the binding and updates the problem's state.

        :param binding: The binding to execute, represented as a tuple ([(place, token), ...], time, action).
        :param problem: The `GymProblem` instance representing the simulation problem.
        """

        return self.solver.solve(binding, problem)

    def create_environment(self):
        """
        Creates a Gymnasium environment that learns to optimize the actions taken on this SimAction (for multi-agent setting, currently unimplemented).

        :return:
        """
        return AEPN_Env(self)



class NetworkTag():
    def __init__(self, tag='e'):
        self.tag = tag.strip().lower()
        if self.tag not in ['e', 'a']:
            raise ValueError("Network tag must be either 'e' or 'a'.")

    def update_tag(self):
        """
        Updates the network tag to switch between evolution ('e') and action ('a') phases.
        """
        if self.is_evolution():
            self.tag = 'a'
            #print("Switching to action mode.")
        elif self.is_action():
            self.tag = 'e'
            #print("Switching to evolution mode.")
        else:
            raise ValueError("Network tag must be either 'e' or 'a'.")

    def is_action(self):
        """
        Checks if the current network tag indicates an action phase.

        :return: True if the tag is 'a', False otherwise.
        """
        return self.tag == 'a'


    def is_evolution(self):
        """
        Checks if the current network tag indicates an evolution phase.

        :return: True if the tag is 'e', False otherwise.
        """
        return self.tag == 'e'



class GymVar(SimVar):
    def __init__(self, _id, attributes: dict, priority=lambda token: token.time):
        super().__init__(_id, priority=priority)
        self.attributes = attributes


class SafeNormalizeFeatures(BaseTransform):

    def __init__(self, attrs=None):
        """
        Normalize the features of the graph data.
        :param attrs: the features to normalize. Normally, the features are the 'x' attributes of the nodes.
        """
        if attrs is None:
            attrs = ["x"]
        self.attrs = attrs
        self.eps = 1e-8

    def forward(self, data):
        for node_type, store in data.node_items():
            if 'x' in store and store['x'].numel() > 0 and node_type != 'a_transition': # a_transitions are one-hot encoded and empty tensors does not need scaling
                x = store['x']
                min_val = x.min(dim=0, keepdim=True).values
                max_val = x.max(dim=0, keepdim=True).values

                store['x'] = (x - min_val) / (max_val - min_val + self.eps)
        #print(data.node_stores)
        return data

