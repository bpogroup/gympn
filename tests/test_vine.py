"""Vine decisions (Agent._vine_label): paired-rollout option values at a
random share of the training decisions, and their expected-advantage loss in
the policy update."""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.train import make_parser, make_agent  # noqa: E402
from insurer_env import INSURER_BUILDERS, INSURER_HEURISTICS  # noqa: E402


def _setup(vine_prob, merge=False):
    pn = INSURER_BUILDERS["insurer"](1, allow_postpone=False)
    pn.length = 6
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    pn.merge_options = merge
    pn.rollout_base = INSURER_HEURISTICS["insurer"]
    env = AEPN_Env(pn)
    args = make_parser().parse_args([])
    args.policy_kwargs = {"encoder": "aepn", "encoder_kwargs": {"global_attn": True, "count_aggr": True}}
    args.value_kwargs = dict(args.policy_kwargs)
    args.flat_obs, args.global_view = True, 'edges'
    args.nfgae, args.smdp_discount, args.marking_certificate = True, True, True
    args.vine_prob, args.vine_rollouts, args.vine_weight = vine_prob, 2, 1.0
    args.agent_seed = 0
    torch.manual_seed(0)
    agent = make_agent(args, metadata=pn.make_metadata())
    return env, agent


def _episode(env, agent, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    R, L = agent.run_episode(env, buffer=agent.buffer)
    return R, L, list(agent.buffer.actions)


def test_vine_labels_leave_the_episode_unchanged():
    env0, a0 = _setup(0.0)
    env1, a1 = _setup(1.0)
    a1.policy_model.load_state_dict(a0.policy_model.state_dict())
    a1.value_model.load_state_dict(a0.value_model.state_dict())
    for seed in (3, 4):
        a0.buffer.clear(); a1.buffer.clear()
        assert _episode(env0, a0, seed) == _episode(env1, a1, seed)


def test_vine_labels_and_loss():
    env, agent = _setup(0.5, merge=True)
    for seed in (1, 2, 3):
        _episode(env, agent, seed)
    gs = [s['graph'] for s in agent.buffer.states]
    lab = [g for g in gs if not torch.isnan(g.t_scores).any()]
    assert lab and len(lab) < len(gs)
    for g in lab:
        # the taken option is the reference class, and it is one of the scored ones
        assert 0 <= int(g.t_base) < int(g.t_ncls) == g.t_scores.numel()
        assert int(g.t_chosen) == int(g.t_base)
    loader = agent.buffer.get(batch_size=8)
    before = [p.detach().clone() for p in agent.policy_model.parameters()]
    hist = agent._fit_policy_and_value_models(loader, epochs=1)
    assert np.isfinite(hist['loss']).all()
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.policy_model.parameters()))
