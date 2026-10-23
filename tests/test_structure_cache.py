"""During a PPO update the AEPN encoder's structure (type groups, attention
plan) is computed once per minibatch and reused (networks._STRUCTURE_CACHE).
The structure depends on the graph only, so cached passes must equal
uncached ones bitwise, for the actor (attention masked to components) and the
critic (unmasked), and the actor and critic must not share an entry."""
import os
import random
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor, HeteroCritic, _STRUCTURE_CACHE, clear_structure_cache  # noqa: E402
from insurer_env import INSURER_BUILDERS  # noqa: E402

KW = {"global_attn": True, "count_aggr": True}


def _batch():
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    pn = INSURER_BUILDERS["insurer_alloc"](1, allow_postpone=False)
    pn.length = 20
    pn.flat_obs, pn.global_view, pn.marking_certificate = True, 'edges', True
    env = AEPN_Env(pn)
    obs = env.reset()
    graphs, rng = [], random.Random(2)
    while env.pn.pn_actions and len(graphs) < 12:
        graphs.append(obs['graph'])
        obs, _, done, _, _ = env.step(rng.randrange(len(env.pn.pn_actions)))
        if done:
            break
    meta = env.pn.make_metadata()
    return Batch.from_data_list(graphs), meta


def _nets(meta, g):
    torch.manual_seed(0)
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=2, metadata=meta, num_heads=1,
                        dropout=0.0, residual=True, encoder="aepn", encoder_kwargs=dict(KW))
    critic = HeteroCritic(input_size=-1, hidden_size=32, num_layers=2, metadata=meta, num_heads=1,
                          dropout=0.0, residual=True, encoder="aepn", encoder_kwargs=dict(KW))
    with torch.no_grad():
        actor({'graph': g}); critic(g)
    return actor, critic


def _pass(net, g, arg):
    net.zero_grad()
    out = net(arg).float().reshape(-1)
    w = torch.randn(out.numel(), generator=torch.Generator().manual_seed(3))
    (out * w).sum().backward()
    return out.detach().clone(), [p.grad.clone() for p in net.parameters() if p.grad is not None]


def test_cached_structure_gives_identical_passes():
    g, meta = _batch()
    actor, critic = _nets(meta, g)
    for net, arg in ((actor, {'graph': g}), (critic, g)):
        clear_structure_cache()
        with torch.no_grad():                       # no caching outside training
            net(arg)
        assert not _STRUCTURE_CACHE.entries
        first = _pass(net, g, arg)                  # computes and caches
        assert len(_STRUCTURE_CACHE.entries) == 1
        second = _pass(net, g, arg)                 # cache hit
        assert len(_STRUCTURE_CACHE.entries) == 1
        clear_structure_cache()
        fresh = _pass(net, g, arg)                  # recomputed
        for other in (second, fresh):
            assert torch.equal(first[0], other[0])
            assert all(torch.equal(a, b) for a, b in zip(first[1], other[1]))


def test_actor_and_critic_do_not_share_an_entry():
    g, meta = _batch()
    actor, critic = _nets(meta, g)
    assert actor.encoder.component_mask and not critic.encoder.component_mask
    clear_structure_cache()
    _pass(actor, g, {'graph': g})
    _pass(critic, g, g)
    assert len(_STRUCTURE_CACHE.entries) == 2
    clear_structure_cache()
    assert not _STRUCTURE_CACHE.entries
