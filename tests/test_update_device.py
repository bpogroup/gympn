"""PPO updates on a GPU (Agent.update_device, --use_gpu): the models, their
Adam state and the batches move to the GPU for the update and back to the CPU
for rollouts. One epoch from the same seed gives the same weights as a CPU
update up to float rounding (the GPU encoder takes the basis order and its
scatter-adds are not bit-reproducible), and the models are back on the CPU
afterwards. Skipped without CUDA."""
import copy
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.agents import PPOAgent  # noqa: E402
from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor, HeteroCritic  # noqa: E402
from insurer_env import make_insurer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup(seed=3):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = make_insurer(1)
    pn.length = 20
    pn.flat_obs = True
    env = AEPN_Env(pn)
    meta = pn.make_metadata()
    actor = HeteroActor(hidden_size=16, num_layers=2, metadata=meta, num_heads=2, dropout=0.0)
    critic = HeteroCritic(hidden_size=16, num_layers=2, metadata=meta, num_heads=2, dropout=0.0)
    ag = PPOAgent(policy_network=actor, value_network=critic, method='clip', eps=0.2,
                  beta=0.05, smdp_discount=True, nfgae=True, lam=1.0)
    return env, ag


def _one_epoch(device):
    env, ag = _setup()
    ag.update_device = device
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    ag.train(env, episodes=2, epochs=1, batch_size=8)
    return ag


def test_gpu_gradients_match_cpu_gradients():
    """One PPO update step's gradients, CPU vs GPU, on the same batch."""
    env, ag = _setup()
    random.seed(11); np.random.seed(11); torch.manual_seed(11)
    ag.run_episodes(env, episodes=2, store=True)
    batch = next(iter(ag.buffer.get(normalize_advantages=True, batch_size=8, drop_remainder=True)))
    grads = {}
    for dev in ('cpu', 'cuda'):
        a = copy.deepcopy(ag)
        a._models_to(dev)
        a.policy_optimizer.step = lambda *args, **kw: None       # keep the gradients, skip the step
        a.value_optimizer.step = lambda *args, **kw: None
        a._fit_policy_and_value_model_step(batch.to(dev))
        a._fit_value_model_step(batch.to(dev))
        grads[dev] = {k: p.grad.cpu() for m in (a.policy_model, a.value_model)
                      for k, p in m.named_parameters(prefix=type(m).__name__) if p.grad is not None}
    assert grads['cpu'].keys() == grads['cuda'].keys()
    num = sum(((grads['cpu'][k] - grads['cuda'][k]) ** 2).sum() for k in grads['cpu']).sqrt()
    den = sum((g ** 2).sum() for g in grads['cpu'].values()).sqrt()
    assert num / den < 1e-4


def test_gpu_update_matches_cpu_update_and_returns_to_cpu():
    """A whole epoch: the trained networks give the same outputs, and
    everything is back on the CPU. Outputs, not parameters: Adam divides by
    the gradient's RMS, so rounding noise on a gradient that is exactly zero
    (the attention's key bias, which softmax ignores) becomes steps of up to
    lr in random directions on parameters that do not change any output."""
    cpu, gpu = _one_epoch(None), _one_epoch('cuda')
    for m in (gpu.policy_model, gpu.value_model):
        assert all(p.device.type == 'cpu' for p in m.parameters())
    for opt in (gpu.policy_optimizer, gpu.value_optimizer):
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    assert v.device.type == 'cpu', k
    env, _ = _setup(seed=5)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    with torch.no_grad():
        for _ in range(10):
            for name in ('policy_model', 'value_model'):
                a = getattr(cpu, name).eval()({'graph': obs['graph']})
                b = getattr(gpu, name).eval()({'graph': obs['graph']})
                assert torch.allclose(a, b, atol=2e-3), name
            obs, _, done, _, _ = env.step(0)
            if done:
                break
