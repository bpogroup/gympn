"""Tests for gympn.component_agents: one PPO agent per net component, in
component turns, credited with its component's return.

Routing: every decision goes to the agent of the component whose turn it is,
and that agent only sees and chooses among its component's bindings. Under the
marking certificate the allocation agent (the whole region) hands over to one
agent per process once the pool is empty. Batches give every agent the steps a
single agent would take. The policies replay through GymSolver, and component
turns work with component postponement."""
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.agents import PPOAgent  # noqa: E402
from gympn.component_agents import ComponentAgents, ComponentPolicy  # noqa: E402
from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor, HeteroCritic  # noqa: E402
from insurer_env import make_insurer, make_insurer_alloc  # noqa: E402


def _env(pn, marking=False, length=20, seed=3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn.length = length
    pn.component_turns = True
    pn.marking_certificate = marking
    pn.flat_obs = True
    return AEPN_Env(pn)


def _agents(env, beta=0.5):
    meta = env.pn.make_metadata()

    def make():
        actor = HeteroActor(hidden_size=16, num_layers=2, metadata=meta, num_heads=2, dropout=0.0)
        critic = HeteroCritic(hidden_size=16, num_layers=2, metadata=meta, num_heads=2, dropout=0.0)
        return PPOAgent(policy_network=actor, value_network=critic, method='clip', eps=0.2,
                        beta=beta, smdp_discount=True, nfgae=True, lam=1.0)
    return ComponentAgents(make, beta=beta, lam=1.0,
                           marking_certificate=env.pn.marking_certificate)


def _rollout(env, ag, episodes=1):
    """Run episodes, recording every decision's component and offered bindings."""
    seen = []
    orig = ag.agent

    def agent(key):
        pn = env.pn
        part = pn.net_partition()
        comps = {b[2].comp if b[0] == ['postpone'] else part[b[2]._id] for b in pn.pn_actions}
        seen.append((key, {pn.net_component_members(c) for c in comps}))
        return orig(key)
    ag.agent = agent
    for _ in range(episodes):
        ag.run_episode(env, buffer=ag.buffer)
    ag.agent = orig
    return seen


def test_each_decision_goes_to_its_components_agent():
    env = _env(make_insurer(1))
    ag = _agents(env)
    seen = _rollout(env, ag)
    assert seen and all(offered == {key} for key, offered in seen)   # one component, its own agent
    assert len(ag.agents) == 3                                       # claims, underwriting, complaints
    keys = list(ag.agents)
    assert all(not (a & b) for i, a in enumerate(keys) for b in keys[i + 1:])
    assert ag.step_keys == [k for k, _ in seen]


def test_marking_certificate_hands_the_region_over_to_process_agents():
    env = _env(make_insurer_alloc(1), marking=True)
    ag = _agents(env, beta=0.05)
    seen = _rollout(env, ag)
    keys = [k for k, _ in seen]
    parent = keys[0]
    n_parent = keys.count(parent)                                    # allocations, and any other
    assert n_parent >= 3 and keys[:n_parent] == [parent] * n_parent  # decision before the pool is empty
    children = set(keys[n_parent:])
    assert parent not in children and len(children) == 3
    assert all(c < parent for c in children)                         # children are parts of the region
    assert len(ag.agents) == 4


def test_parent_credit_is_its_childrens_discounted_return():
    """lam = 1: the allocation agent's advantage at its last decision is the
    region's return from there, each child's rewards discounted on that
    child's own decision clock, minus the allocation agent's value."""
    env = _env(make_insurer_alloc(1), marking=True)
    ag = _agents(env, beta=0.05)
    rews = []
    orig_step = env.step

    def step(a):
        out = orig_step(a)
        rews.append(dict(out[4].get('comp_reward', {})))
        return out
    env.step = step
    seen = _rollout(env, ag)
    keys = [k for k, _ in seen]
    t = keys.count(keys[0]) - 1                                      # the region's last decision
    buf, times = ag.buffer, ag.buffer.times
    G = 0.0
    for s in range(t, len(rews)):
        for tr, r in rews[s].items():
            child = next((k for k in set(keys[t + 1:]) if tr in k), None)
            last = max((u for u in range(t + 1, s + 1) if keys[u] == child), default=t) if child else t
            G += r * np.exp(-0.05 * (times[last] - times[t]))
    assert float(buf.advantages_[t]) == pytest.approx(G - float(buf.values_pred[t]), abs=1e-4)


def test_equal_steps_batches_and_subset_keep_the_credit():
    env = _env(make_insurer(1))
    ag = _agents(env)
    _rollout(env, ag, episodes=2)
    N = len(ag.buffer)
    batches = ag._agent_batches(32)
    assert sum(len(b) for _, b, _ in batches.values()) == N
    for key, (_, b, bs) in batches.items():
        idx = [i for i, k in enumerate(ag.step_keys) if k == key]
        assert bs == max(2, round(32 * len(idx) / N))
        assert torch.equal(b.advantages_, ag.buffer.advantages_[idx])
        assert torch.equal(b.returns_, ag.buffer.returns_[idx])
        assert b.actions == [ag.buffer.actions[i] for i in idx]


def test_training_epoch_runs_and_policy_replays(tmp_path):
    from gympn.solvers import GymSolver
    env = _env(make_insurer_alloc(1), marking=True)
    ag = _agents(env, beta=0.05)
    test_env = _env(make_insurer_alloc(1), marking=True)
    h = ag.train(env, episodes=2, epochs=1, batch_size=8, test_env=test_env, test_freq=1,
                 test_episodes=2, logdir=str(tmp_path), eval_seed=7)
    assert h['n_agents'][0] == 4 and h['policy_updates'][0] > 0
    path = tmp_path / "best_policy.pth"
    assert path.exists()
    pol = torch.load(path, weights_only=False)
    assert isinstance(pol, ComponentPolicy) and len(pol.names) == 4
    pn = make_insurer_alloc(1)                       # a fresh net, default settings
    pn.flat_obs = True
    random.seed(1); np.random.seed(1)
    r = pn.testing_run(solver=GymSolver(str(path), pn.make_metadata()), length=20)
    assert pn.component_turns and pn.marking_certificate
    assert np.isfinite(float(r))


def test_component_turns_with_component_postponement():
    pn = make_insurer(1, allow_postpone=True)
    pn.postpone_scope = 'component'
    env = _env(pn)
    ag = _agents(env)
    obs = env.reset()
    n_pp = 0
    for _ in range(200):
        pp = [b for b in env.pn.pn_actions if b[0] == ['postpone']]
        assert len(pp) <= 1                                          # its own postpone only
        key = obs['component']
        part = env.pn.net_partition()
        assert all(env.pn.net_component_members(b[2].comp if b[0] == ['postpone'] else part[b[2]._id]) == key
                   for b in env.pn.pn_actions)
        a = len(env.pn.pn_actions) - 1 if pp and n_pp < 5 else 0     # postpone a few times
        n_pp += bool(pp) and a == len(env.pn.pn_actions) - 1
        obs, _, done, _, _ = env.step(a)
        if done:
            break
    assert n_pp > 0
    ag.run_episode(env, buffer=ag.buffer)
    assert len(ag.buffer) == len(ag.step_keys) > 0
