"""run_episodes with rollout_batch > 1 (Agent._run_episodes_lockstep) steps
several episodes side by side and scores their observations in one batched
forward pass. What it stores must be what one-at-a-time collection would
store for the same states: each step's action probabilities and value equal
a single-graph forward pass on the stored state, and each episode reaches the
buffer whole and in order, with its returns and NF-GAE bookkeeping."""
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.train import make_agent, make_parser  # noqa: E402
from insurer_env import INSURER_BUILDERS  # noqa: E402


def _setup(regions=2, postpone=True, marking=False, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = INSURER_BUILDERS["insurer"](regions, allow_postpone=postpone)
    pn.length = 8
    pn.flat_obs = True
    pn.global_view = 'edges'
    pn.marking_certificate = marking
    if postpone:
        pn.postpone_scope = 'component'
    env = AEPN_Env(pn)
    env.reset()
    args = make_parser().parse_args([])
    args.nfgae, args.smdp_discount, args.flat_obs = True, True, True
    args.marking_certificate = marking
    agent = make_agent(args, metadata=env.pn.make_metadata())
    return env, agent


def _check_buffer(agent, hist, atol):
    buf = agent.buffer
    buf._materialize_steps()
    assert len(buf) == int(hist['lengths'].sum())
    assert buf.advantages_.numel() == len(buf)               # every episode was finished
    agent.policy_model.eval(); agent.value_model.eval()
    with torch.no_grad():
        for i, st in enumerate(buf.states):
            lp = agent.policy_model(st).log().flatten()
            assert torch.allclose(lp, buf.logpis_nodes[i], atol=atol), i
            assert float(buf.logprobs_sel[i]) == pytest.approx(float(lp[buf.actions[i]]), abs=atol)
            v = agent.value_model(st['graph']).flatten()[0]
            assert float(buf.values_pred[i]) == pytest.approx(float(v), abs=atol)
    # episodes in order, each whole: their rewards sum to the returns
    o = 0
    for R, L in zip(hist['returns'], hist['lengths']):
        L = int(L)
        assert float(buf.rewards_raw[o:o + L].sum()) == pytest.approx(R, abs=1e-6)
        o += L


@pytest.mark.parametrize("postpone,marking", [(True, False), (False, True)])
def test_lockstep_stores_what_single_forwards_give(postpone, marking):
    env, agent = _setup(postpone=postpone, marking=marking)
    agent.rollout_batch = 3
    agent.buffer.clear()
    hist = agent.run_episodes(env, episodes=5, store=True)
    assert (hist['lengths'] > 0).all()
    _check_buffer(agent, hist, atol=1e-5)


def test_lockstep_is_reproducible():
    out = []
    for _ in range(2):
        env, agent = _setup(seed=4)
        agent.rollout_batch = 4
        agent.buffer.clear()
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        hist = agent.run_episodes(env, episodes=4, store=True)
        out.append((hist['returns'].tolist(), list(agent.buffer.actions)))
    assert out[0] == out[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_lockstep_on_gpu():
    env, agent = _setup()
    agent.rollout_batch = 4
    agent.update_device = 'cuda'
    agent.buffer.clear()
    hist = agent.run_episodes(env, episodes=4, store=True)
    assert next(agent.policy_model.parameters()).device.type == 'cpu'
    assert next(agent.value_model.parameters()).device.type == 'cpu'
    _check_buffer(agent, hist, atol=2e-4)
