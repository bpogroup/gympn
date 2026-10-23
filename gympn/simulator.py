import copy
import os
import itertools
import random
import inspect
import subprocess

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

from simpn.simulator import SimProblem, SimEvent, SimVar, SimVarQueue, SimToken
try:
    from simpn.simulator import SimVarTime
except Exception:  # pragma: no cover
    SimVarTime = ()

from gympn.environment import AEPN_Env
from gympn.solvers import GymSolver
from gympn.train import make_agent, make_parser, make_logdir, launch_tensorboard


from typing import List

def _is_real_binding_tuple(b) -> bool:
    return (
        isinstance(b, tuple) and len(b) >= 2 and
        isinstance(b[0], list) and len(b[0]) > 0 and isinstance(b[0][0], tuple)
    )


class PostponeRef:
    """Third element of a component-scoped postpone pseudo-binding
    (['postpone'], clock, PostponeRef(c)): which component waits."""
    __slots__ = ('comp', '_id')

    def __init__(self, comp):
        self.comp = comp
        self._id = f"postpone_c{comp}"

    def __repr__(self):
        return f"PostponeRef({self.comp})"


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

    def __init__(self, debugging=True, binding_priority=lambda bindings: bindings[0], tag='e', has_var_attrs=True, solver=None, plot_observations=False, allow_postpone=False):
        super().__init__(debugging, binding_priority)

        self.network_tag = NetworkTag(tag)  # boolean to indicate if it is time to take action ('a') or evolutions (i.e. normal events, 'e')
        self.has_var_attrs = has_var_attrs  # boolean to indicate if the problem has variable attributes (necessary for DRL)
        self.allow_postpone = allow_postpone  # boolean to indicate if the problem allows postponing actions
        self.just_postponed = False  # boolean to indicate if the system just postponed

        self.reward_functions = {} # reward functions are associated to events/actions through a dictionary for backward compatibility with simpn events
        self.var_attributes = {} # similarly, places need to be associated with the attributes of their tokens
        self.reward = 0 #total cumulated reward (for DRL)
        # Cumulated reward per net component ({component id: reward}), see
        # net_partition(). Read by the nfgae scheme; the total stays in self.reward.
        self.comp_reward = {}
        self._net_partition_cache = None
        self._net_place_partition = None
        # Marking-based certificate (nfgae): recompute the partition whenever
        # the set of marked places changes, so transitions that can no longer
        # fire stop coupling components. Off by default: the partition is then
        # computed once and cached. With it on, comp_reward is keyed by
        # transition id instead of component id (component ids are renumbered
        # when the partition refines; see net_component_members).
        self.marking_certificate = False
        self._net_partition_sig = None
        # Global view: score every action from the whole visible net. 'edges'
        # (the default) adds the reverse of every edge type (u, rel, v) as
        # (v, rel + '_rev', u), so messages flow against token flow too, and
        # gives every node its net component (_add_component_ids); the AEPN
        # encoder then attends over the whole component
        # (AEPNStack(global_attn=True, count_aggr=True), its default). False
        # gives the old forward-only graph, where an action's logit depends
        # only on what is forward-reachable from it within num_layers hops.
        self.global_view = 'edges'
        # Postpone scope. 'global' (default): one postpone pseudo-binding; choosing
        # it blocks every action until the next event fires anywhere.
        # 'component': one postpone_c per net component (net_partition) with an
        # enabled action; choosing it blocks only c's actions, the decision round
        # continues for the others, and c is asked again at its OWN next event.
        # With a single component the two are identical. See
        # suite/paper/NFGAE_THEORY.md (Theorem 1, condition C3').
        self.postpone_scope = 'global'
        self.postponed_comps = set()

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
            try:
                from gympn.plotter import GraphPlotter
            except ImportError as e:
                raise ImportError("plot_observations needs matplotlib and networkx; "
                                  "install them with `pip install gympn[viz]`") from e
            self.plotter = GraphPlotter()
        else:
            self.plotter = None

        #unobservable simvars
        self.unobservable_simvars = []
        #unobservable token attributes
        self.unobservable_token_attrs = {} #this is a dictionary with the place._id as key and a list of unobservable token attrs for that place as values
        #categorical token attributes
        self.categorical_token_attrs = {}

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



    def add_action(self, inflow, outflow, behavior, name=None, guard=None, reward_function=None, solver=None, _fast=False):
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

        if _fast:
            # Trusted internal path (expand_no_future_tokens): inputs are correct
            # by construction, so skip all validation — the arity check via
            # inspect.signature (~0.3s/search, profiled), the SimVar type checks,
            # the id2node uniqueness check, and the postpone-guard exec. The
            # expanded net is a throwaway structural artifact rebuilt every step.
            result = SimAction(name, guard=guard, behavior=behavior,
                               incoming=inflow, outgoing=outflow, solver=solver)
            self.actions.append(result)
            return result

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

        if self._component_postpone() and self.postponed_comps:
            if len(timed_bindings_evo) == 0:
                # No event will ever fire again, so a waiting component would
                # never be woken: wake everyone (the stated exception to
                # component independence; never triggered with recurring arrivals).
                self.postponed_comps.clear()
            else:
                timed_bindings_act = [b for b in timed_bindings_act
                                      if not self._postponed_action(b[2])]


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
            if getattr(self, 'postponed_comps', None):
                self.postponed_comps.discard(self.net_partition().get(event._id))

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
                         allow_postpone=self.allow_postpone)  # propagate flags

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

        # copy var_attributes for observable places, minus the unobservable attrs
        for p in ret.places:
            if p._id in self.var_attributes:
                hidden = self.unobservable_token_attrs.get(p._id, []) if self.unobservable_token_attrs else []
                ret.var_attributes[p._id] = [a for a in self.var_attributes[p._id] if a not in hidden]

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

    def compute_pn_actions(self):
        """Refresh ``self.pn_actions`` (the firable binding list) WITHOUT building
        the HeteroData tensor graph.

        This is the cheap path for planners that only need to pick and fire
        actions (MCTS descent + rollout), skipping the ~66% of
        ``get_graph_observation`` spent on tensor/HeteroData construction. It
        reproduces exactly the binding-map that ``get_graph_observation`` builds
        in its ``a_transition`` branch — same ``expand_no_future_tokens``, same
        iteration order over ``expanded_pn.actions``, same ``b_time <= clock``
        filter, same trailing postpone entry — so action indices stay identical
        to those the graph path produces (the network prior/eval alignment).
        """
        self.expanded_pn, _ = self.expand_no_future_tokens(actions_only=True)
        keep_comp = self._turn_component(self.expanded_pn) if getattr(self, 'component_turns', False) else None
        part = self.net_partition() if keep_comp is not None else None
        transition_binding_map = []
        for t in self.expanded_pn.actions:
            if keep_comp is not None and part.get(GymProblem._get_string_before_last_dot(t._id)) != keep_comp:
                continue
            b_list = [el.marking[0] for el in t.incoming]
            binds = [([place for place in self.places
                       if place._id == GymProblem._get_string_before_last_dot(el._id)][0],
                      el.marking[0]) for el in t.incoming]
            b_time = max([tok.time for tok in b_list])
            original_transition = [action for action in self.actions
                                   if action._id == GymProblem._get_string_before_last_dot(t._id)][0]
            if b_time <= self.clock:
                transition_binding_map.append((binds, b_time, original_transition))
        if self.allow_postpone:
            transition_binding_map.extend(self._postpone_entries(transition_binding_map))
        self.pn_actions = transition_binding_map
        return transition_binding_map

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

        if self._use_direct_flat(minimal_obs, normalize, remove_empty_nodes, add_self_loops):
            return self._direct_flat_observation()

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

        # Component turns (nfgae local encoding): the observation holds only the
        # component whose turn it is, and pn_actions only its bindings. None =
        # whole net (default, unchanged).
        keep_comp = self._turn_component(self.expanded_pn) if getattr(self, 'component_turns', False) else None
        if keep_comp is not None:
            part, ppart = self.net_partition(), self.net_place_partition()
            node_types = [n for n in node_types
                          if n in ('a_transition', 'e_transition') or ppart.get(n) == keep_comp]

        # Expanded places by original place, in expanded order; original places
        # and actions by id (first occurrence, as the list scans returned).
        expanded_by_original = {}
        for p in self.expanded_pn.places:
            expanded_by_original.setdefault(GymProblem._get_string_before_last_dot(p._id), []).append(p)
        place_by_id, action_by_id = {}, {}
        for p in self.places:
            place_by_id.setdefault(p._id, p)
        for a in self.actions:
            action_by_id.setdefault(a._id, a)

        # Update the HeteroData object with the places attributes and the transition nodes
        for n_t in node_types:
            if n_t != 'a_transition' and n_t != 'e_transition':
                p_nodes = []
                for p in expanded_by_original.get(n_t, ()):
                    p_name_original = n_t
                    if True:

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
                    t_name = GymProblem._get_string_before_last_dot(t._id)
                    if keep_comp is not None:
                        if part.get(t_name) != keep_comp:
                            continue
                        i = len(t_nodes)        # node position within the kept component
                    if add_self_loops:
                        t_values = [1 if action._id == t_name else 0 for action in self.actions] #a_transitions know which transition type they are (useful with self loops and multiple actions)
                    else:
                        t_values = [] #otherwise they do not need to carry information

                    b_list = [el.marking[0] for el in t.incoming]
                    binds = [(place_by_id[GymProblem._get_string_before_last_dot(el._id)], el.marking[0]) for el in t.incoming]
                    b_time = max([t.time for t in b_list]) #changed min to max, it was an error!
                    original_transition = action_by_id[t_name]

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
                    if keep_comp is not None:
                        if part.get(t_name) != keep_comp:
                            continue
                        i = len(t_nodes)
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


        # Update the HeteroData object with the arcs: the (source, target)
        # pairs of every edge type are collected in arc order and turned into
        # one tensor per type (types in order of first appearance; an arc whose
        # place is unmarked adds the type but no edge).
        edge_pairs = {}
        for a in self.expanded_pn.arcs:
            #a[0] is the source
            source_name = GymProblem._get_string_before_last_dot(a[0]._id)
            dest_name = GymProblem._get_string_before_last_dot(a[1]._id)
            key = None
            if keep_comp is not None:
                # only arcs between the kept component's places and transitions
                place_name = source_name if type(a[0]) is SimVar else dest_name
                tr_id = a[1]._id if type(a[0]) is SimVar else a[0]._id
                if ppart.get(place_name) != keep_comp or (
                        tr_id not in a_transition_dict and tr_id not in e_transition_dict):
                    continue

            if type(a[0]) is SimVar and source_name not in self.unobservable_simvars:
                if type(a[1]) is SimAction:
                    key = (source_name, 'edge_to_a_transition', 'a_transition')
                    pair = (int(a[0]._id.split('.')[-1]), a_transition_dict[a[1]._id]) if a[0].marking else None
                else:
                    key = (source_name, 'edge_to_e_transition', 'e_transition')
                    pair = (int(a[0]._id.split('.')[-1]), e_transition_dict[a[1]._id]) if a[0].marking else None
            elif type(a[1]) is SimVar and dest_name not in self.unobservable_simvars:
                if type(a[0]) is SimAction:
                    key = ('a_transition', 'edge_to_' + dest_name, dest_name)
                    pair = (a_transition_dict[a[0]._id], int(a[1]._id.split('.')[-1])) if a[1].marking else None
                else:
                    key = ('e_transition', 'edge_to_' + dest_name, dest_name)
                    pair = (e_transition_dict[a[0]._id], int(a[1]._id.split('.')[-1])) if a[1].marking else None

            if key is not None:
                pairs = edge_pairs.setdefault(key, [])
                if pair is not None:
                    pairs.append(pair)

        for key, pairs in edge_pairs.items():
            ret_graph[key].edge_index = (torch.tensor(pairs, dtype=torch.int64).t().contiguous() if pairs
                                         else torch.empty([2, 0], dtype=torch.int64))

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
            self._remove_empty_nodes_inplace(ret_graph)   # ret_graph is fresh: no copy needed

        if minimal_obs:
            ret_graph = self.minimal_graph(ret_graph)

        if normalize:
            # Normalize the features
            transform = SafeNormalizeFeatures()
            #transform2 = remove_isolated_nodes.RemoveIsolatedNodes()
            #transform = Compose([transform1, transform2])
            ret_graph = transform(ret_graph)

        self.pn_actions = transition_binding_map

        if self._component_postpone():
            self._add_component_postpone_nodes(ret_graph, transition_binding_map)
        elif self.allow_postpone:
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


        if getattr(self, 'global_view', False):
            self._add_global_view(ret_graph)
            self._add_component_ids(ret_graph, transition_binding_map, keep_comp)

        if self.metadata is None: #TODO: handle case where postpone is initially not available
            # a component-turn observation holds one component only
            self.metadata = self.make_metadata() if keep_comp is not None else ret_graph.metadata()

        if self.plot_observations:
            self.plotter.plot_side_by_side(self.expanded_pn, ret_graph)

        if getattr(self, 'flat_obs', False):
            ret_graph = self._to_flat(ret_graph)

        obs = {'graph': ret_graph, 'actions_dict': transition_binding_map}
        if keep_comp is not None:
            # whose turn it is, by member transitions (gympn.component_agents)
            obs['component'] = self.net_component_members(keep_comp)
        return obs


    # Build flat observations directly (see _direct_flat_observation). False
    # forces the HeteroData path everywhere (equivalence testing).
    DIRECT_FLAT_OBS = True

    def _use_direct_flat(self, minimal_obs, normalize, remove_empty_nodes, add_self_loops):
        """The direct builder covers the flat observation with default options,
        on nets of plain SimVar places, once the first observation (HeteroData
        path) has fixed the flat metadata and feature width."""
        if not (self.DIRECT_FLAT_OBS and getattr(self, 'flat_obs', False)
                and not minimal_obs and not normalize and remove_empty_nodes and add_self_loops
                and not self.plot_observations and not getattr(self, 'component_turns', False)
                and getattr(self, '_flat_meta', None) is not None and self.metadata is not None):
            return False
        ok = self.__dict__.get('_direct_flat_supported')
        if ok is None:
            places = set(map(id, self.places))
            ok = (all(type(p) is SimVar for p in self.places)
                  and all(id(p) in places for t in list(self.events) + list(self.actions)
                          for p in list(t.incoming) + list(t.outgoing))
                  and all(len(a.outgoing) > 0 for a in self.actions))
            self._direct_flat_supported = ok
        return ok

    def _token_feature_list(self, p_name, token):
        """A token's node features as a list of floats: the HeteroData path's
        token_value, value for value."""
        v = token.value
        if isinstance(v, dict):
            hidden = self.unobservable_token_attrs.get(p_name, [])
            cat = self.categorical_token_attrs.get(p_name, {})
            row = [float(val) for key, val in v.items() if key not in hidden and key not in cat]
            if p_name in self.categorical_token_attrs:
                for key, val in v.items():
                    if key in cat:
                        one_hot = [0.0] * len(cat[key])
                        one_hot[val] = 1.0
                        row += one_hot
            return row
        if isinstance(v, (tuple, list)):
            if all(isinstance(item, dict) for item in v):
                return [float(val) for d in v for val in d.values()]
            return [float(val) for val in v]
        raise TypeError("Unsupported token value type.")

    def _direct_flat_observation(self):
        """get_graph_observation for flat_obs, built straight from the marking.

        The HeteroData path expands the net into 1-bounded places and one
        action per token combination (expand_no_future_tokens), builds a
        HeteroData with one tensor per token, adds reverse edges and component
        ids type by type, and then flattens it: thousands of small Python and
        torch operations per step. This builds the same FlatGraph, and the same
        pn_actions, from plain lists and creates each tensor once. It follows
        the HeteroData path rule for rule, including its quirks: an action gets
        a node for EVERY combination of its input places' present tokens (guards
        decide only whether the action has nodes at all), its outgoing edges go
        to its FIRST output place only, edge types are ordered by first
        appearance, and a place type with no nodes or zero-width features loses
        its edges (remove_empty_nodes); a place whose tokens have no features
        gets no nodes at all.
        tests/test_direct_obs.py checks the two paths for equality."""
        from gympn.flat_graph import FlatGraph, vocab
        clock = self.clock
        unobs = set(self.unobservable_simvars)
        cache = self.__dict__.get('_direct_flat_cache')
        if cache is None:
            obs_places = [p._id for p in self.places if p._id not in unobs]
            place_by_id = {}
            for p in self.places:
                place_by_id.setdefault(p._id, p)
            a_onehot = {a._id: [1.0 if b._id == a._id else 0.0 for b in self.actions] for a in self.actions}
            e_rows = [[1.0 if f._id == e._id else 0.0 for f in self.events] for e in self.events]
            empty_w = {}
            for nt in obs_places:
                if nt in self.unobservable_token_attrs:
                    empty_w[nt] = len([el for el in self.var_attributes[nt]
                                       if el not in self.unobservable_token_attrs[nt]])
                else:
                    empty_w[nt] = len(self.var_attributes[nt])
            cache = self._direct_flat_cache = (obs_places, place_by_id, a_onehot, e_rows, empty_w,
                                               vocab(self._flat_meta))
        obs_places, place_by_id, a_onehot, e_rows, empty_w, (ntype_idx, etype_idx) = cache
        width = self._flat_width

        # present tokens (time <= clock) per place, in marking order
        valid = {p._id: [t for t in p.marking if t.time <= clock] for p in self.places}

        # ---- node types, in HeteroData insertion order: places, a_transition, e_transition
        types = []          # [name, n, feature width, rows or None]
        for nt in obs_places:
            toks = valid[nt]
            if toks:
                rows = [self._token_feature_list(nt, tok) for tok in toks]
                w = len(rows[0])
                if any(len(r) != w for r in rows):
                    raise RuntimeError(f"tokens of place {nt!r} have different feature widths")
                if w == 0:      # featureless tokens: the HeteroData path keeps no nodes for them
                    types.append([nt, 0, empty_w[nt], None])
                else:
                    types.append([nt, len(rows), w, rows])
            else:
                types.append([nt, 0, empty_w[nt], None])

        # ---- action nodes and arcs (events' arcs come first in arc order)
        edge_pairs = {}     # edge type -> [(src, dst)], in order of first appearance

        def arc(key, pair):
            pairs = edge_pairs.setdefault(key, [])
            if pair is not None:
                pairs.append(pair)

        def copies(pid):
            n = len(valid[pid])
            return list(range(n)) if n else [None]       # None: the empty copy "<pid>.0"

        for j, e in enumerate(self.events):
            for p in e.incoming:
                if p._id not in unobs:
                    for ci in copies(p._id):
                        arc((p._id, 'edge_to_e_transition', 'e_transition'), None if ci is None else (ci, j))
            for p in e.outgoing:
                if p._id not in unobs:
                    for ci in copies(p._id):
                        arc(('e_transition', 'edge_to_' + p._id, p._id), None if ci is None else (j, ci))

        binding_map, a_rows = [], []
        for t in self.actions:
            if self._postponed_action(t):
                continue
            vb = [b for (b, tm) in self.event_bindings(t) if tm <= clock]
            if not vb:
                continue
            in_places = {}
            for place, _tok in vb[-1]:
                in_places.setdefault(place._id, []).extend(copies(place._id))
            pids = list(in_places)
            out = t.outgoing[0]._id
            # (the HeteroData path lists a marked output place's copies once per arc to it)
            out_copies = copies(out) * (sum(p._id == out for p in t.outgoing) if valid[out] else 1)
            for combo in itertools.product(*in_places.values()):
                k = len(a_rows)
                toks = [valid[pid][ci] for pid, ci in zip(pids, combo)]
                binding_map.append(([(place_by_id[pid], SimToken(tok.value, tok.time))
                                     for pid, tok in zip(pids, toks)],
                                    max(tok.time for tok in toks), t))
                a_rows.append(a_onehot[t._id])
                for pid, ci in zip(pids, combo):
                    if pid not in unobs:
                        arc((pid, 'edge_to_a_transition', 'a_transition'), (ci, k))
                if out not in unobs:
                    for ci in out_copies:
                        arc(('a_transition', 'edge_to_' + out, out), None if ci is None else (k, ci))

        if a_rows:
            types.append(['a_transition', len(a_rows), len(self.actions), a_rows])
        if self.events:
            types.append(['e_transition', len(e_rows), len(self.events), e_rows])
        if a_rows:
            edge_pairs[('a_transition', 'self_loop', 'a_transition')] = [(i, i) for i in range(len(a_rows))]

        # remove_empty_nodes: edge types touching a type with no features lose their edges
        empty = {nt for nt, n, w, _ in types if n * w == 0}
        if empty:
            for key in edge_pairs:
                if key[0] in empty or key[2] in empty:
                    edge_pairs[key] = []

        self.pn_actions = binding_map
        self.expanded_pn = None
        sizes = {nt: n for nt, n, _, _ in types}
        need_comp = bool(getattr(self, 'global_view', False))
        if need_comp or self._component_postpone():
            part, place_part = self.net_partition(), self.net_place_partition()
            a_comp = [part.get(b[2]._id) for b in binding_map]
            e_comp = [part.get(GymProblem._get_string_before_last_dot(e._id) if '.' in e._id else e._id)
                      for e in self.events]

            def node_comps(nt, n):
                if nt == 'a_transition':
                    return a_comp[:n]
                if nt == 'e_transition':
                    return e_comp[:n]
                return [place_part.get(nt)] * n

        # ---- postpone node(s)
        if self._component_postpone():
            entries = self._postpone_entries(binding_map)
            slot = {e[2].comp: i for i, e in enumerate(entries)}
            for nt, n, _, _ in list(types):
                cs = node_comps(nt, n)
                edge_pairs[(nt, 'to_postpone', 'postpone')] = [(i, slot[c]) for i, c in enumerate(cs) if c in slot]
            types.append(['postpone', len(entries), 1, [[0.0] for _ in entries]])
            binding_map.extend(entries)
        elif self.allow_postpone:
            for nt, n, _, _ in list(types):
                edge_pairs[(nt, 'to_postpone', 'postpone')] = [(i, 0) for i in range(n)]
            types.append(['postpone', 1, 1, [[0.0]]])
            binding_map.append((['postpone'], self.clock, None))
        sizes = {nt: n for nt, n, _, _ in types}

        # ---- whole-net view: reverse edges, component of every node
        comp = None
        if need_comp:
            for (src, rel, dst) in list(edge_pairs):
                if rel != 'self_loop':
                    edge_pairs[(dst, rel + '_rev', src)] = [(b, a) for (a, b) in edge_pairs[(src, rel, dst)]]
            real = [b for b in binding_map if b[0] != ['postpone']]
            pp = self._postpone_entries(real) if self._component_postpone() else []
            comp = []
            for nt, n, _, _ in types:
                if n == 0:
                    continue
                if nt == 'postpone':
                    cs = [e[2].comp for e in pp][:n] if pp else [None] * n
                else:
                    cs = node_comps(nt, n)
                comp += [-1 if c is None else int(c) for c in cs]

        # ---- flatten (hetero_to_flat, same order)
        w_max = max((w for _, _, w, _ in types), default=0)
        if w_max > width:
            raise ValueError(f"observation feature width {w_max} > fixed flat width {width}")
        x_rows, ntype, offset = [], [], {}
        for nt, n, w, rows in types:
            offset[nt] = len(x_rows)
            if n == 0:
                continue
            if nt not in ntype_idx:
                raise KeyError(f"node type {nt!r} not in the network's metadata")
            pad = [0.0] * (width - w)
            x_rows += [r + pad for r in rows]
            ntype += [ntype_idx[nt]] * n
        src_all, dst_all, etype = [], [], []
        for key, pairs in edge_pairs.items():
            if not pairs:
                continue
            if key not in etype_idx:
                raise KeyError(f"edge type {key!r} not in the network's metadata")
            s, _, d = key
            if sizes.get(s, 0) == 0 or sizes.get(d, 0) == 0:
                continue
            so, do = offset[s], offset[d]
            src_all += [a + so for a, _ in pairs]
            dst_all += [b + do for _, b in pairs]
            etype += [etype_idx[key]] * len(pairs)

        def positions(nt):
            if sizes.get(nt, 0) == 0:
                return torch.arange(0, dtype=torch.long)
            return torch.arange(offset[nt], offset[nt] + sizes[nt], dtype=torch.long)

        x = (torch.tensor(x_rows, dtype=torch.float32) if x_rows
             else torch.zeros((0, width), dtype=torch.float32))
        kw = {'comp': torch.tensor(comp, dtype=torch.long)} if comp is not None and x_rows else {}
        graph = FlatGraph(x=x, ntype=torch.tensor(ntype, dtype=torch.long),
                          edge_index=torch.tensor([src_all, dst_all], dtype=torch.long),
                          etype=torch.tensor(etype, dtype=torch.long),
                          a_idx=positions('a_transition'), p_idx=positions('postpone'),
                          num_nodes=len(x_rows), **kw)
        return {'graph': graph, 'actions_dict': binding_map}

    def match_policy_view(self, model):
        """Set global_view to the one a trained policy was built for
        (networks.policy_global_view), so policies trained before the
        whole-net view became the default still get the graphs they know.
        Component agents (gympn.component_agents.ComponentPolicy) also bring
        the turn and certificate settings they were trained under."""
        from .networks import policy_global_view
        view = policy_global_view(model)
        if view is not None and view != self.global_view:
            self.global_view = view
            self.metadata = None
            self._flat_meta = None
        for k, v in (getattr(model, 'pn_settings', None) or {}).items():
            setattr(self, k, v)

    def _add_component_ids(self, g, transition_binding_map, keep_comp=None):
        """Store each node's net component (net_partition, so the marking
        certificate when it is on) as g[nt].comp, a long tensor, -1 for nodes
        of no component (dead parts, a global postpone). The AEPN encoder's
        global attention stays inside a component, so an action's logit is a
        function of its own component only (NF-GAE Theorem 1, C2), while it
        still reads every node of that component. Same node attribution as
        _add_component_postpone_nodes."""
        part, place_part = self.net_partition(), self.net_place_partition()
        real = [b for b in transition_binding_map if b[0] != ['postpone']]
        a_comp = [part.get(b[2]._id) for b in real]
        e_comp = [part.get(GymProblem._get_string_before_last_dot(t._id) if '.' in t._id else t._id)
                  for t in self.expanded_pn.events]
        pp = self._postpone_entries(real) if self._component_postpone() else []
        for nt in list(g.node_types):
            x = g[nt].x
            n = 0 if x is None else x.size(0)
            if keep_comp is not None:
                comps = [keep_comp] * n
            elif nt == 'a_transition':
                comps = a_comp[:n]
            elif nt == 'e_transition':
                comps = e_comp[:n]
            elif nt == 'postpone':
                comps = [e[2].comp for e in pp][:n] if pp else [None] * n
            else:
                comps = [place_part.get(nt)] * n
            g[nt].comp = torch.tensor([-1 if c is None else int(c) for c in comps], dtype=torch.long)

    def _add_global_view(self, g):
        """global_view: add the reverse of every edge type (self-loops
        excepted), in place. make_metadata lists the same types."""
        for (src, rel, dst) in list(g.edge_types):
            if rel == 'self_loop':
                continue
            g[(dst, rel + '_rev', src)].edge_index = g[(src, rel, dst)].edge_index.flip(0)

    def _to_flat(self, ret_graph):
        """HeteroData observation -> FlatGraph (flat_obs; see gympn/flat_graph.py).

        The type vocabulary is make_metadata(), the same one the networks are
        built from. The padding width must be the same for every observation of
        this net (graphs of different widths cannot be batched), and the widest
        type can be absent from a given observation (a_transition, a one-hot over
        all actions, when nothing is enabled), so it is fixed on first use from
        the net definition and that first observation."""
        from gympn.flat_graph import feature_width, hetero_to_flat
        if getattr(self, '_flat_meta', None) is None:
            self._flat_meta = self.make_metadata()
            self._flat_width = max(feature_width(ret_graph.x_dict), len(self.actions), len(self.events), 1)
        if feature_width(ret_graph.x_dict) > self._flat_width:
            raise ValueError(f"observation feature width {feature_width(ret_graph.x_dict)} > "
                             f"fixed flat width {self._flat_width}")
        return hetero_to_flat(ret_graph, self._flat_meta, self._flat_width)

    def remove_empty_nodes(self, ret_graph):
        """
        Removes empty nodes (e.g. nodes that have numel()==0) from the graph observation by modifying the connectivity of the graph.
        If an empty node has incoming and outgoing edges, those will be removed.

        Empty nodes are those that do not have any tokens or attributes associated with them.

        :param ret_graph: The graph observation as a `HeteroData` object.
        :return: The updated `HeteroData` object with empty nodes removed.
        """
        updated_graph = ret_graph.clone()
        self._remove_empty_nodes_inplace(updated_graph)
        return updated_graph

    @staticmethod
    def _remove_empty_nodes_inplace(graph):
        """remove_empty_nodes on graph itself: every edge type touching a node
        type with no nodes (x.numel() == 0) loses its edges."""
        empty = {nt for nt in graph.node_types if graph[nt].x.numel() == 0}
        if empty:
            for edge_type in graph.edge_types:
                if edge_type[0] in empty or edge_type[2] in empty:
                    graph[edge_type].edge_index = torch.tensor([], dtype=torch.int64).reshape(2, 0)

    def expand_no_future_tokens(self, actions_only=False):
        """
        Expands the attributed A-E PN into a 1-bounded attributed A-E PN.
        Filters out tokens and bindings with time > self.clock.

        ``actions_only=True`` builds only the expanded PLACES and ACTIONS
        (skipping the expanded events and all arc bookkeeping). ``pn_actions``
        is derived solely from ``expanded_pn.actions`` and their input places, so
        the events/arcs are dead weight for ``compute_pn_actions`` — this is the
        cheap path in the planning hot loop and produces an IDENTICAL action
        order/content to the full expansion (verified), so alignment with the
        graph's a_transition nodes is preserved.

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
            else:
                new_place_id = f"{p._id}.0"
                expanded_pn.add_var(name=new_place_id, var_attributes=self.var_attributes[p._id])

        # original place id -> its expanded copies, IN place order. Precomputed
        # once so the actions loop below does O(1) lookups instead of rescanning
        # every expanded place (with a _get_string call) per binding — that scan
        # was ~672k _get_string calls/search, profiled. Order is preserved, so
        # the expanded-action order (hence pn_actions↔graph alignment) is intact.
        expanded_by_original = {}
        for pl in expanded_pn.places:
            expanded_by_original.setdefault(
                self._get_string_before_last_dot(pl._id), []).append(pl)

        # Expand events (evolutions) — skipped for actions_only (unused by
        # compute_pn_actions, which reads only expanded_pn.actions).
        for t in ([] if actions_only else self.events):
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
            if self._postponed_action(t):
                continue        # its component is waiting (component postpone)
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
                    for new_place in expanded_by_original.get(place._id, []):
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

                new_tr = expanded_pn.add_action(new_inflow, new_outflow, behavior=dummy_behavior, name=f"{t._id}.{index_new_action}", _fast=True)
                if not actions_only:
                    for el in new_tr.incoming:
                        expanded_pn.arcs.append((el, new_tr))
                    for el in new_tr.outgoing:
                        expanded_pn.arcs.append((new_tr, el))

        return expanded_pn, transition_binding_map

    def save_state(self):
        """Lightweight snapshot of the MUTABLE simulation state, avoiding the
        deep-copy of the (immutable-during-a-search) net STRUCTURE — SimVars,
        transitions, arcs, id2node, metadata. Profiled: the full-pn deepcopy in
        env.set_state was ~20% of a search; this copies only markings + a few
        scalars.

        Restores identically via ``restore_state``. Token values are dicts that
        behaviors may mutate in place, so marking tokens are deep-copied.
        """
        markings = {}
        for p in self.places:
            if SimVarTime and isinstance(p, SimVarTime):
                continue
            if isinstance(p, SimVarQueue):
                continue
            markings[p._id] = [copy.deepcopy(tok) for tok in p.marking]
        state = {
            "markings": markings,
            "clock": self.clock,
            "reward": self.reward,
            "comp_reward": dict(getattr(self, "comp_reward", {})),
            "postponed_comps": set(getattr(self, "postponed_comps", set())),
            "tag": self.network_tag.tag,
            "just_postponed": self.just_postponed,
        }
        return state

    def restore_state(self, state):
        """Restore a snapshot from ``save_state`` in place (self.pn stays the
        same object; only mutable state is rewritten)."""
        for p in self.places:
            if SimVarTime and isinstance(p, SimVarTime):
                continue
            if isinstance(p, SimVarQueue):
                continue
            saved = state["markings"].get(p._id)
            if saved is None:
                continue
            p.marking.clear()
            for tok in saved:
                p.marking.add(copy.deepcopy(tok))
        self.clock = state["clock"]
        self.reward = state["reward"]
        self.comp_reward = dict(state.get("comp_reward", {}))
        self.postponed_comps = set(state.get("postponed_comps", set()))
        self.network_tag.tag = state["tag"]
        self.just_postponed = state["just_postponed"]
        # pn_actions references tokens; the restored tokens are fresh copies, so
        # rebuild the binding list to point at them (cheap path, no tensor graph).
        self.compute_pn_actions()

    _dummy_behavior_cache = {}

    def _make_dummy_behavior(self, num_args):
        # Cache by arity: the dummy behavior is stateless (returns [None]*n and
        # takes n args), so one function per distinct arity is reused everywhere.
        # expand_no_future_tokens builds one per expanded action in the hot
        # planning loop; without the cache that was ~14k exec() calls/search
        # (profiled the #1 cost after the graph-skip refactor).
        fn = GymProblem._dummy_behavior_cache.get(num_args)
        if fn is None:
            arg_list = ', '.join([f'arg{i}' for i in range(num_args)])
            func_code = f"def dummy_behavior({arg_list}):\n    return [None] * {num_args}"
            local_vars = {}
            exec(func_code, {}, local_vars)
            fn = GymProblem._dummy_behavior_cache[num_args] = local_vars['dummy_behavior']
        return fn

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


    _sbd_cache = {}

    @staticmethod
    def _get_string_before_last_dot(s: str) -> str:
        """
        Extracts the substring before the last dot in a string.

        Memoized: this is a pure function of ``s`` called ~660k times/search in
        the expansion/binding hot loop (profiled), over a small set of repeating
        place/transition ids.

        :param s: The input string.
        :return: The substring before the last dot.
        """
        cache = GymProblem._sbd_cache
        out = cache.get(s)
        if out is None:
            last_dot_index = s.rfind('.')
            out = s if last_dot_index == -1 else s[:last_dot_index]
            cache[s] = out
        return out

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


    def run_evolutions(self, run, i, active_model, build_obs=True):
        """
        Function invoked by Gym environment to let the network perform evolutions when no actions are required

        ``build_obs=False`` returns ``None`` instead of the HeteroData graph
        observation at each return point (the graph build is ~80% of step cost;
        cheap planner rollouts don't need it). ``pn.pn_actions`` is still updated
        by ``bindings()``, so callers can pick/fire actions without the graph.
        """
        def _obs():
            if build_obs:
                return self.get_graph_observation()
            # cheap path: refresh pn_actions (needed to fire) without the tensor
            # graph, and return no observation.
            self.compute_pn_actions()
            return None

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
                    # A single binding is not a decision -- there is nothing to
                    # choose -- so it is fired without consulting the agent.
                    if len(bindings) > 1:  # only consult the agent on a real choice
                        return _obs(), self.clock > self.length or not active_model, i
                    else:
                        binding = bindings[0]
                        run.append(binding)
                        result_tokens = self.fire(binding)
                        self.update_reward(binding, result_tokens)
                        i += 1
                else:
                    # allow_postpone: the agent may also choose to postpone.
                    return _obs(), self.clock > self.length or not active_model, i
            else:
                active_model = False

        return _obs(), self.clock > self.length or not active_model, i

    def net_partition(self):
        """Component id of every transition: the connected components of the
        undirected place-transition graph (an edge joins a place to every
        transition that consumes from or produces into it). Two transitions
        sharing ANY place -- a queue, a resource pool -- are in one component,
        so pre-emption never crosses a component boundary. Depends on the net
        only, never on the marking or the trajectory: this is the influence
        certificate of nfgae (suite/paper/NFGAE_THEORY.md, Theorem 1).
        Postpone is a pseudo-binding, not a transition, so it never merges
        components.

        DEAD transitions are pruned first: a place is markable if it is marked
        now or produced into by a live transition, a transition is live if all
        its input places are markable (least fixpoint). This over-approximates
        what can ever fire (it ignores token counts and guards), so pruning is
        sound -- a dead transition never fires under any policy. It matters
        for e.g. multi-site with n_flex=0, whose empty `flex` place would
        otherwise glue every site into one component. Dead transitions map
        to None (they never fire, so are never looked up). Computed on first use from the marking
        at that time (the initial one, in practice) and cached; the topology
        is fixed."""
        cache = getattr(self, '_net_partition_cache', None)
        if getattr(self, 'marking_certificate', False):
            # Marking-based certificate: valid while the set of marked places is
            # unchanged. Live(M') is a subset of Live(M) for every M' reachable
            # from M (the places marked at M' are markable from M), so the
            # partition only ever refines along a trajectory.
            sig = frozenset(p._id for p in self.places if p.marking)
            if cache is not None and sig == getattr(self, '_net_partition_sig', None):
                return cache
            self._net_partition_sig = sig
            # The partition is a function of the set of marked places only.
            memo = self.__dict__.setdefault('_net_partition_memo', {})
            if sig in memo:
                self._net_partition_cache, self._net_place_partition = memo[sig]
                return self._net_partition_cache
        elif cache is not None:
            return cache
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        transitions = list(self.actions) + list(self.events)
        markable = {p._id for p in self.places if p.marking}
        live, changed = set(), True
        while changed:
            changed = False
            for t in transitions:
                if t._id not in live and all(p._id in markable for p in t.incoming):
                    live.add(t._id)
                    markable.update(p._id for p in t.outgoing)
                    changed = True

        for t in transitions:
            if t._id not in live:
                parent.setdefault(('t', t._id), ('t', t._id))
                continue
            tk = ('t', t._id)
            parent.setdefault(tk, tk)
            for p in list(t.incoming) + list(t.outgoing):
                pk = ('p', p._id)
                parent.setdefault(pk, pk)
                ra, rb = find(tk), find(pk)
                if ra != rb:
                    parent[ra] = rb
        roots, comp = {}, {}
        for t in transitions:
            comp[t._id] = (roots.setdefault(find(('t', t._id)), len(roots))
                           if t._id in live else None)     # dead: never fires
        # Places inherit the component of the live transitions touching them
        # (None if only dead transitions touch them).
        self._net_place_partition = {
            p._id: (roots.get(find(('p', p._id))) if ('p', p._id) in parent else None)
            for p in self.places}
        self._net_partition_cache = comp
        if getattr(self, 'marking_certificate', False):
            self.__dict__.setdefault('_net_partition_memo', {})[self._net_partition_sig] = (comp, self._net_place_partition)
        return comp

    def net_place_partition(self):
        """Component id of every place (see net_partition)."""
        self.net_partition()
        return self._net_place_partition

    def net_component_members(self, c):
        """The live transitions of component c in the current partition, as a
        frozenset of transition ids. With the marking-based certificate this is
        the stable name of a component: ids are renumbered when the partition
        refines, member sets are not."""
        part = self.net_partition()
        return frozenset(t for t, k in part.items() if k == c and k is not None)

    def _turn_component(self, expanded_pn):
        """Component turns (nfgae local encoding): the lowest component id with
        an enabled action binding now, or None if nothing is enabled.

        With component_turns the decision process offers one component at a
        time, and the policy chooses among that component's bindings only. No
        simulated time passes between turns, components share no place, and
        the setting is work-conserving, so the order of turns cannot change any
        component's outcomes; each decision then needs only its own
        component's subgraph (actor, critic and observation building)."""
        part = self.net_partition()
        comps = set()
        for t in expanded_pn.actions:
            if max(el.marking[0].time for el in t.incoming) <= self.clock:
                c = part.get(GymProblem._get_string_before_last_dot(t._id))
                if c is not None:
                    comps.add(c)
        return min(comps) if comps else None

    def _component_postpone(self):
        return self.allow_postpone and getattr(self, 'postpone_scope', 'global') == 'component'

    def _postponed_action(self, t):
        """True if t's component is currently waiting (component postpone)."""
        return (self._component_postpone() and bool(self.postponed_comps)
                and self.net_partition().get(t._id) in self.postponed_comps)

    def _postpone_entries(self, transition_binding_map):
        """Postpone pseudo-bindings to append after the production bindings.
        Global scope: one (['postpone'], clock, None). Component scope: one
        (['postpone'], clock, PostponeRef(c)) per component with an enabled
        action, in increasing component id (the order of the postpone nodes)."""
        if not self._component_postpone():
            return [(['postpone'], self.clock, None)]
        part = self.net_partition()
        comps = sorted({part[b[2]._id] for b in transition_binding_map
                        if not (isinstance(b[0], list) and b[0] == ['postpone'])})
        return [(['postpone'], self.clock, PostponeRef(c)) for c in comps]

    def _add_component_postpone_nodes(self, ret_graph, transition_binding_map):
        """Component scope: one postpone node per postpone pseudo-binding, in
        the same order, each receiving edges ONLY from its own component's
        nodes (place nodes by place, a_transition nodes by binding, e_transition
        nodes by event). Nodes of dead parts connect to nothing. With one
        component this is exactly the global block (every live node -> the
        single postpone node)."""
        entries = self._postpone_entries(transition_binding_map)
        comps = [e[2].comp for e in entries]
        slot = {c: i for i, c in enumerate(comps)}
        ret_graph['postpone'].x = torch.zeros((len(comps), 1), dtype=torch.float32)
        part = self.net_partition()
        place_part = self.net_place_partition()
        a_comp = [part.get(b[2]._id) for b in transition_binding_map]
        e_comp = [part.get(GymProblem._get_string_before_last_dot(t._id) if '.' in t._id else t._id)
                  for t in self.expanded_pn.events] if getattr(self, 'expanded_pn', None) is not None else []
        for n_t in list(ret_graph.node_types):
            if n_t == 'postpone':
                continue
            num_nodes = ret_graph[n_t].x.size(0)
            if n_t == 'a_transition':
                node_comps = a_comp[:num_nodes]
            elif n_t == 'e_transition':
                node_comps = e_comp[:num_nodes]
            else:
                node_comps = [place_part.get(n_t)] * num_nodes
            src = [i for i, c in enumerate(node_comps) if c in slot]
            dst = [slot[node_comps[i]] for i in src]
            edge_indices = torch.tensor([src, dst], dtype=torch.int64).reshape(2, -1)
            edge_type = (n_t, 'to_postpone', 'postpone')
            if edge_type not in ret_graph.edge_types:
                ret_graph[edge_type].edge_index = edge_indices
            else:
                ret_graph[edge_type].edge_index = torch.cat(
                    (ret_graph[edge_type].edge_index, edge_indices), dim=1)
        transition_binding_map.extend(entries)

    def update_reward(self, timed_binding, result_tokens=None):
        """Accrue the firing's reward, in total and for its net component."""
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
            if getattr(self, 'marking_certificate', False):
                c = transition._id         # matched against member sets by nfgae
            else:
                c = self.net_partition().get(transition._id)
            self.comp_reward[c] = self.comp_reward.get(c, 0.0) + r_f

        else:
            r_f = 0

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

        if getattr(self, 'global_view', False):
            # the reverse edge types _add_global_view creates
            for (src, rel, dst) in list(edges_meta):
                rev = (dst, rel + '_rev', src)
                if rel != 'self_loop' and rev not in edges_meta:
                    edges_meta.append(rev)

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

    def set_seed(self, seed, *, deterministic: bool = True):
        """Seed every stochastic source (Python, NumPy, PyTorch, cuDNN) for a
        reproducible run. Single source of truth is :func:`gympn.seed_everything`.
        Environment stochasticity (arrivals, delays) uses the global ``random`` /
        ``numpy`` modules, so this also makes the net's dynamics reproducible."""
        from .seeding import seed_everything
        return seed_everything(seed, deterministic=deterministic)

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
        # Defaults only: settings come from args_dict, never from the command
        # line of whatever program calls this (a notebook, pytest, a script).
        args = make_parser().parse_args([])

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
            # Single source of truth: seeds Python/NumPy/PyTorch/cuDNN and sets
            # deterministic kernels, so the whole run is reproducible.
            from .seeding import seed_everything
            seed_everything(args.agent_seed)

        if getattr(args, 'flat_obs', False):
            flat_encoders = ('type_embed', 'temb', 'type_embed_film', 'tembf', 'aepn')
            encs = [(getattr(args, k, None) or {}).get('encoder', 'aepn') for k in ('policy_kwargs', 'value_kwargs')]
            if any(e not in flat_encoders for e in encs):
                raise ValueError(f"flat_obs needs a flat encoder for actor and critic, got {encs}")
        if getattr(args, 'component_agents', False):
            # one agent per component: component turns and net-factored credit
            args.local_obs = True
            args.nfgae = True
        if getattr(args, 'local_obs', False):
            if not args.nfgae:
                raise ValueError("local_obs is for nfgae: its advantage and critic are per component")
            if self.allow_postpone and getattr(self, 'postpone_scope', 'global') != 'component':
                raise ValueError("local_obs needs allow_postpone=False or postpone_scope='component' "
                                 "(a global postpone couples the components)")
        # Set before AEPN_Env deep-copies the problem, so the frozen copy (reset)
        # and the test env carry it too.
        self.component_turns = bool(getattr(args, 'local_obs', False))
        if getattr(args, 'marking_certificate', False) and not args.nfgae:
            raise ValueError("marking_certificate is for nfgae: it changes how decisions are credited")
        self.marking_certificate = bool(getattr(args, 'marking_certificate', False))
        self.flat_obs = bool(getattr(args, 'flat_obs', False))
        self.global_view = getattr(args, 'global_view', 'edges') or False
        self._flat_meta = None
        if self.global_view:
            self.metadata = None        # re-read with the extra edge types

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
                    'nfgae': args.nfgae,
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
                logger.warning("wandb is not installed, skipping W&B logging; install it with `pip install gympn[wandb]`")
            except Exception as e:
                logger.warning(f"Could not initialize W&B: {e}")

        logger.training_start(agent.__class__.__name__, args.epochs, args.episodes)

        history = agent.train(env, episodes=args.episodes, epochs=args.epochs,
                    save_freq=args.save_freq, logdir=logdir, verbose=args.verbose,
                    max_episode_length=args.max_episode_length, batch_size=args.batch_size,
                    test_env=test_env, test_freq=test_freq,
                    test_episodes=getattr(args, 'test_episodes', 10),
                    eval_seed=getattr(args, 'eval_seed', None),
                    wandb_logger=wandb_logger)

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
        augmented.extend(self._postpone_entries(bindings))
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
                        self.postpone(comp=getattr(timed_binding[2], 'comp', None))
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
                        self.postpone(comp=getattr(timed_binding[2], 'comp', None))
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

    def testing_run(self, solver, length=10, reporter=None, visualize=False, seed=None):
        """
        Test run aligned with AEPN_Env semantics:
          1) Ensure we start in ACTION phase (like env.reset()).
          2) Loop:
               - Choose exactly ONE action (or postpone).
               - Apply it (fire or postpone).
               - Run all evolutions: obs, terminated, _ = run_evolutions(...)
               - Stop when 'terminated' is True (same as AEPN_Env.step).
          3) Return total accumulated reward (same scalar your env exposes in info['pn_reward']).

        The testing run evaluates the solver's performance over a specified duration.

        :param solver: An instance of a solver class implementing the `BaseSolver` interface.
        :param length: The maximum duration of the testing run. The simulation will stop if the clock exceeds (or matches) this length.
        :param reporter: A reporter to log simulation events.
        :param visualize: Whether to visualize the simulation.
        :param seed: if given, seed all stochastic sources first (via
            :func:`gympn.seed_everything`) so the evaluation episode is reproducible.
        :return: The total reward accumulated during the testing run.
        """
        # ---- Setup identical to environment.reset() ----
        if hasattr(solver, 'policy_model'):
            self.match_policy_view(solver.policy_model)
        if seed is not None:
            self.set_seed(seed)
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
                               result_tokens)  # accumulate reward
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

    def postpone(self, comp=None):
        """
        Postpones the current action by advancing the clock to the next valid evolution event.

        Component scope (postpone_scope='component'): only component ``comp``
        waits -- its actions are blocked until one of ITS events fires -- and the
        decision round continues for the other components.
        """
        if not self.allow_postpone:
            raise Exception("Postpone is not allowed in this A-E PN.")

        if self._component_postpone():
            if comp is None:
                raise ValueError("component postpone needs the component id")
            self.postponed_comps.add(comp)
            return

        self.network_tag.tag = 'e'
        self.just_postponed = True


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

