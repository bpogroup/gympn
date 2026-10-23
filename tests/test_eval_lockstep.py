"""test_in_train with common random numbers and rollout_batch > 1 plays its
evaluation episodes side by side (Agent._test_lockstep), each environment
with its own random state seeded eval_seed + i. Every episode must be exactly
the one the sequential evaluation plays, and the training random streams must
be left as they were."""
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


def _setup(name, regions, postpone):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    pn = INSURER_BUILDERS[name](regions, allow_postpone=postpone)
    pn.length = 10
    pn.flat_obs, pn.global_view = True, 'edges'
    if postpone:
        pn.postpone_scope = 'component'
    env = AEPN_Env(pn)
    env.reset()
    args = make_parser().parse_args([])
    args.nfgae, args.smdp_discount, args.flat_obs = True, True, True
    agent = make_agent(args, metadata=env.pn.make_metadata())
    with torch.no_grad():
        agent.policy_model(env.reset())               # materialise lazy layers
    return env, agent


def _sequential(agent, env, eval_seed, n):
    R, L = [], []
    for i in range(n):
        random.seed(eval_seed + i); np.random.seed((eval_seed + i) % 2 ** 32)
        st, done, length = env.reset(), False, 0
        while not done:
            st, _, done, _, info = env.step(agent.act(st, deterministic=True))
            length += 1
        R.append(info['pn_reward']); L.append(length)
    return np.array(R), np.array(L)


@pytest.mark.parametrize("name,regions,postpone", [("insurer", 2, True), ("insurer_alloc", 1, False)])
def test_batched_evaluation_plays_the_same_episodes(name, regions, postpone):
    env, agent = _setup(name, regions, postpone)
    R_seq, L_seq = _sequential(agent, env, 4242, 7)
    agent.rollout_batch = 3                           # fewer slots than episodes: slots get reused
    R_b, L_b = agent._test_lockstep(env, 7, 4242)
    assert np.array_equal(R_seq, R_b) and np.array_equal(L_seq, L_b)


def test_batched_evaluation_restores_training_streams():
    env, agent = _setup("insurer", 1, False)
    agent.rollout_batch = 4
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    m = agent.test_in_train(env, episodes=6, eval_seed=999)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1])
    assert torch.equal(torch.get_rng_state(), before[2])
    agent.rollout_batch = 1
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    m_seq = agent.test_in_train(env, episodes=6, eval_seed=999)
    assert m['mean_returns'] == m_seq['mean_returns'] and m['mean_ep_lens'] == m_seq['mean_ep_lens']
