"""One learning agent per net component.

The net's components (GymProblem.net_partition, the independence certificate)
decide where the agent boundaries are. Each component gets its own PPO agent,
with its own policy, critic and optimizer and no shared weights. Decisions come
in component turns (GymProblem.component_turns): each decision offers the
bindings of one component, observed as that component's whole subgraph, and is
taken by that component's agent.

Credit follows the certificate (suite/paper/NFGAE_THEORY.md). A decision is
credited with its own component's rewards, discounted on the component's own
clock and bootstrapped with its own agent's critic. Under the marking-based
certificate (GymProblem.marking_certificate) a component splits once the
transitions that coupled its parts can no longer fire, e.g. once a start-of-
shift pool has been allocated. The agent of the coupled component (the
parent) then hands over to one agent per part (the children). Its last
decision is credited with the children's discounted returns, bootstrapped with
the children's critics at their first decisions
(TrajectoryBuffer._nfgae_refining). At lam = 1 this is the parent's
discounted return minus its value.

Agents are named by their component's member transitions
(GymProblem.net_component_members), which stay the same across episodes and
across refinements. An agent is created the first time its component decides.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from gympn.data import TrajectoryBuffer
from gympn.logging_utils import get_logger


def component_key(pn):
    """The member transitions of the component whose turn it is, or None."""
    if not pn.pn_actions:
        return None
    b = pn.pn_actions[0]
    if isinstance(b[0], list) and b[0] == ['postpone']:
        c = b[2].comp
    else:
        c = pn.net_partition()[b[2]._id]
    return pn.net_component_members(c)


class ComponentPolicy(nn.Module):
    """The agents' policies as one module, for evaluation and replay: the
    observation's 'component' entry (set under component turns) selects the
    actor. `pn_settings` are the GymProblem flags the policies were trained
    under (GymProblem.match_policy_view applies them)."""

    def __init__(self, marking_certificate=False):
        super().__init__()
        self.actors = nn.ModuleDict()
        self.names = {}                          # member frozenset -> ModuleDict key
        self.pn_settings = {'component_turns': True,
                            'marking_certificate': bool(marking_certificate)}
        self.metadata = None

    def add(self, key, actor):
        name = f"c{len(self.names)}"
        self.names[key] = name
        self.actors[name] = actor
        if self.metadata is None:
            self.metadata = getattr(actor, 'metadata', None)

    def forward(self, data):
        key = data.get('component') if isinstance(data, dict) else None
        if key not in self.names:
            raise KeyError("no agent for this component; is the observation built with "
                           "component turns (GymProblem.component_turns)?")
        return self.actors[self.names[key]](data)


class ComponentAgents:
    """One PPO agent per net component.

    Parameters
    ----------
    make_agent : callable
        Returns a fresh agent (PPOAgent with its own networks) for a new
        component. Called with no arguments.
    beta, lam : float
        SMDP discount rate and GAE parameter of the credit (as in nfgae).
    marking_certificate : bool
        Whether the problem refines its partition at the current marking; only
        recorded for replay.
    equal_steps : bool
        True (default): each agent's batch size is batch_size times its share of
        the epoch's decisions, so every agent's networks take as many gradient
        steps per epoch as a single agent trained on all decisions would. False:
        every agent uses batch_size.
    """

    def __init__(self, make_agent, beta=0.0, lam=1.0, marking_certificate=False, equal_steps=True):
        self.make_agent = make_agent
        self.beta = float(beta)
        self.lam = float(lam)
        self.equal_steps = bool(equal_steps)
        self.agents = {}
        self.policy_model = ComponentPolicy(marking_certificate)
        self.buffer = TrajectoryBuffer(lam=lam, beta=beta, smdp_discount=True, nfgae=True)
        self.step_keys = []
        self.best_test_metric = float('-inf')
        self.agent_seed = None

    # -- agents ---------------------------------------------------------------
    def agent(self, key):
        """The agent of component `key`, created on first use."""
        ag = self.agents.get(key)
        if ag is None:
            ag = self.make_agent()
            self.agents[key] = ag
            self.policy_model.add(key, ag.policy_model)
        return ag

    def act(self, state, deterministic=False):
        return self.agent(state['component']).act(state, deterministic=deterministic)

    # -- rollouts -------------------------------------------------------------
    def run_episode(self, env, max_episode_length=None, buffer=None):
        """One episode; with a buffer, every step is stored with its agent's
        value and component, and the episode's credit is computed."""
        state = env.reset()
        pn = env.pn
        if not getattr(pn, 'component_turns', False):
            raise ValueError("ComponentAgents needs component turns (GymProblem.component_turns)")
        if (getattr(pn, 'allow_postpone', False)
                and getattr(pn, 'postpone_scope', 'global') != 'component'):
            raise ValueError("a global postpone couples the components; use postpone_scope='component'")
        marking = getattr(pn, 'marking_certificate', False)
        done, length, info = False, 0, {'pn_reward': 0}
        comps, rews, total = [], [], 0.0
        while not done:
            key = state['component']
            ag = self.agent(key)
            action, logprob, logpis = ag.act(state, return_logprob=True)
            decision_time = float(pn.clock)
            b = pn.pn_actions[action]
            c = b[2].comp if (isinstance(b[0], list) and b[0] == ['postpone']) else pn.net_partition()[b[2]._id]
            comps.append(pn.net_component_members(c) if marking else c)
            if buffer is not None:
                value = float(ag.value(state).reshape(-1)[0])
            next_state, reward, done, _, info = env.step(action)
            rews.append(dict(info.get('comp_reward', {})))
            total += reward
            if buffer is not None:
                buffer.store(state, action, reward, logprob, value, logpis, time=decision_time)
                self.step_keys.append(key)
            length += 1
            if max_episode_length is not None and length > max_episode_length:
                break
            state = next_state
        if buffer is not None:
            buffer._nf_comp, buffer._nf_rew = comps, rews
            buffer.finish()
        return info.get('pn_reward', total), length

    # -- training -------------------------------------------------------------
    def _agent_batches(self, batch_size):
        """Split the epoch's steps by agent: {key: (agent, buffer, batch size)}."""
        N = len(self.buffer)
        out = {}
        for key in dict.fromkeys(self.step_keys):
            idx = [i for i, k in enumerate(self.step_keys) if k == key]
            bs = (max(2, int(round(batch_size * len(idx) / N))) if self.equal_steps else batch_size)
            out[key] = (self.agents[key], self.buffer.subset(idx), bs)
        return out

    def train(self, env, episodes=10, epochs=1, max_episode_length=None, verbose=0, save_freq=1,
              logdir=None, batch_size=64, sort_states=False, test_env=None, test_freq=5, test_episodes=10,
              wandb_logger=None, num_workers=1, eval_seed=None):
        """Train every agent on its own decisions; the same interface and
        history keys as Agent.train."""
        if self.agent_seed is not None:
            from gympn.seeding import seed_network_init
            seed_network_init(self.agent_seed)
        keys = ['mean_returns', 'min_returns', 'max_returns', 'std_returns', 'mean_ep_lens',
                'policy_updates', 'policy_ent', 'policy_kld', 'n_agents']
        history = {k: np.zeros(epochs) for k in keys}
        if test_env is not None:
            n_tests = epochs // test_freq + 1
            for k in ('test_mean_returns', 'test_min_returns', 'test_max_returns', 'test_std_returns'):
                history[k] = np.zeros(n_tests)
        for i in range(epochs):
            frac = max(0.0, 1.0 - i / max(1, epochs - 1))          # entropy annealing, as Agent.train
            self.buffer.clear()
            self.step_keys = []
            R, L = np.zeros(episodes), np.zeros(episodes)
            for e in range(episodes):
                R[e], L[e] = self.run_episode(env, max_episode_length, buffer=self.buffer)
            ent, kld, updates, w = 0.0, 0.0, 0, 0
            for key, (ag, buf, bs) in self._agent_batches(batch_size).items():
                ag._current_epoch = i
                ag.ent_bonus = ag._ent_bonus_initial * frac
                loader = buf.get(normalize_advantages=ag.normalize_advantages,
                                 normalize_returns=ag.normalize_returns, batch_size=bs,
                                 sort=sort_states, drop_remainder=True)
                ph = ag._fit_policy_and_value_models(loader, epochs=ag.policy_updates)
                updates += len(ph['loss'])
                if len(ph['loss']):
                    ent += len(buf) * float(ph['ent'][-1])
                    kld += len(buf) * float(ph['kld'][-1])
                    w += len(buf)
            history['mean_returns'][i], history['min_returns'][i] = R.mean(), R.min()
            history['max_returns'][i], history['std_returns'][i] = R.max(), R.std()
            history['mean_ep_lens'][i] = L.mean()
            history['policy_updates'][i] = updates
            history['policy_ent'][i] = ent / w if w else 0.0
            history['policy_kld'][i] = kld / w if w else 0.0
            history['n_agents'][i] = len(self.agents)
            if test_env is not None and (i + 1) % test_freq == 0:
                tm = self.test_in_train(test_env, test_episodes, max_episode_length, logdir=logdir,
                                        eval_seed=eval_seed)
                j = (i + 1) // test_freq - 1
                for k in ('mean', 'min', 'max', 'std'):
                    history[f'test_{k}_returns'][j] = tm[f'{k}_returns']
            if verbose > 0:
                print(f"\rEpoch {i + 1}/{epochs} - mean_returns: {R.mean():.4f} - agents: {len(self.agents)}",
                      end="\n" if i + 1 == epochs else "")
        if test_env is not None and logdir is not None and self.best_test_metric > float('-inf'):
            self.load_best(logdir)
        return history

    # -- evaluation and checkpoints ------------------------------------------
    def test_in_train(self, env, episodes=10, max_episode_length=None, logdir=None, eval_seed=None):
        """Greedy evaluation of all agents, under common random numbers when
        eval_seed is set (as Agent.test_in_train); saves the best policies."""
        import random
        returns = np.zeros(episodes)
        if eval_seed is not None:
            saved = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            for e in range(episodes):
                if eval_seed is not None:
                    random.seed(eval_seed + e)
                    np.random.seed((eval_seed + e) % (2 ** 32))
                state, done, info = env.reset(), False, {'pn_reward': 0}
                while not done:
                    action = self.act(state, deterministic=True)
                    state, _, done, _, info = env.step(action)
                returns[e] = info['pn_reward']
        finally:
            if eval_seed is not None:
                random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
        m = {'mean_returns': returns.mean(), 'min_returns': returns.min(),
             'max_returns': returns.max(), 'std_returns': returns.std()}
        if m['mean_returns'] > self.best_test_metric:
            self.best_test_metric = m['mean_returns']
            if logdir is not None:
                self.save_policy_network(os.path.join(logdir, "best_policy.pth"))
                get_logger().best_policy_saved(logdir, self.best_test_metric)
        for ag in self.agents.values():
            ag.policy_model.train()
            ag.value_model.train()
        return m

    def save_policy_network(self, filename):
        """Save all agents' policies as one ComponentPolicy."""
        torch.save(self.policy_model, filename)

    def load_best(self, logdir):
        """Put the best saved policies back into the agents."""
        path = os.path.join(logdir, "best_policy.pth")
        if not os.path.exists(path):
            return
        best = torch.load(path, weights_only=False)
        for key, name in best.names.items():
            if key in self.agents:
                self.agents[key].policy_model = best.actors[name]
        for key, ag in self.agents.items():          # agents first seen after the best evaluation
            if key not in best.names:
                best.add(key, ag.policy_model)
        self.policy_model = best
        get_logger().info(f"Restored best policies (eval metric = {self.best_test_metric:.4f}) from {path}")
