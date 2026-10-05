"""Tests for net-factored GAE (scheme 'nfgae'; examples/paper_examples/suite/
paper/NFGAE_THEORY.md): the net partition, the per-component reward split,
K=1 exactness against PPO's SMDP-GAE, a hand-checked K=2 recursion, actor
locality (Theorem 1, condition C2) and the critic's component pooling."""
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite"))

from gympn.data import TrajectoryBuffer, smdp_gae  # noqa: E402
from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor, HeteroCritic  # noqa: E402
from bpm_envs import make_next_activity, make_next_activity_split  # noqa: E402
from multisite_env import make_multisite  # noqa: E402


def _n_comps(pn):
    return len({c for c in pn.net_partition().values() if c is not None})


# --------------------------------------------------------------------------- #
# Partition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 4])
def test_partition_copies(n):
    assert _n_comps(make_next_activity(n, allow_postpone=False)) == n


def test_partition_split_queue_has_two_per_copy():
    assert _n_comps(make_next_activity_split(1, allow_postpone=False)) == 2


def test_partition_multisite():
    assert _n_comps(make_multisite(4, 1, 0)) == 4      # dedicated sites
    assert _n_comps(make_multisite(4, 1, 2)) == 1      # shared flex pool couples all


# --------------------------------------------------------------------------- #
# Per-component reward split
# --------------------------------------------------------------------------- #
def test_comp_reward_sums_to_step_reward():
    random.seed(0); np.random.seed(0)
    pn = make_next_activity(2, allow_postpone=False)
    pn.length = 20
    env = AEPN_Env(pn)
    env.reset()
    for _ in range(80):
        k = len(env.pn.pn_actions)
        if k == 0:
            break
        _, r, done, _, info = env.step(np.random.randint(k))
        assert abs(sum(info['comp_reward'].values()) - r) < 1e-9
        if done:
            break
    assert abs(sum(env.pn.comp_reward.values()) - env.pn.reward) < 1e-9


# --------------------------------------------------------------------------- #
# GAE recursion
# --------------------------------------------------------------------------- #
def _finish(buf, rewards, values, times, comp=None, rw=None):
    for r, v, t in zip(rewards, values, times):
        buf.times.append(float(t))
        buf._rewards_buf.append(torch.tensor([float(r)]))
        buf._values_buf.append(torch.tensor([float(v)]))
        buf._logprobs_buf.append(torch.tensor([0.0]))
        buf.end += 1
    buf._steps_dirty = True
    if comp is not None:
        buf._nf_comp, buf._nf_rew = comp, rw
    buf.finish(credits=None)
    return buf.advantages_.clone()


def test_k1_identity_with_ppo():
    rng = np.random.default_rng(0)
    T = 40
    rewards = rng.integers(0, 3, T).astype(float)
    values = rng.normal(size=T)
    times = np.cumsum(rng.integers(0, 3, T)).astype(float)
    kw = dict(gam=0.99, lam=0.95, causal_beta=0.5, smdp_discount=True)
    a_ppo = _finish(TrajectoryBuffer(causal_scheme='lrq', **kw), rewards, values, times)
    a_nf = _finish(TrajectoryBuffer(causal_scheme='nfgae', **kw), rewards, values, times,
                   comp=[0] * T, rw=[{0: r} for r in rewards])
    assert torch.allclose(a_ppo, a_nf, atol=1e-6)


def test_k2_hand_computed():
    # steps: t0 (c0), t1 (c1), t2 (c0); times 0, 1, 3; beta 0.5, lam 1
    rewards = [1.0, 2.0, 4.0]
    rw = [{1: 1.0}, {0: 2.0}, {0: 4.0}]  # c1's reward at step 0, c0's at steps 1, 2
    values = [0.5, 0.25, 0.75]
    times = [0.0, 1.0, 3.0]
    buf = TrajectoryBuffer(causal_scheme='nfgae', gam=1.0, lam=1.0,
                           causal_beta=0.5, smdp_discount=True)
    a = _finish(buf, rewards, values, times, comp=[0, 1, 0], rw=rw)
    # c0: epoch t0 owns steps [0, 2) -> R = 0 + 2; next c0 epoch t2 after 3 units
    a_t2 = 4.0 - 0.75
    a_t0 = 2.0 + math.exp(-1.5) * 0.75 - 0.5 + math.exp(-1.5) * a_t2
    # c1: its only epoch t1 owns steps [1, 3) -> no c1 reward there
    a_t1 = 0.0 - 0.25
    assert torch.allclose(a, torch.tensor([a_t0, a_t1, a_t2]), atol=1e-6)


# --------------------------------------------------------------------------- #
# Theorem 1, C2: action logits are local to their component
# --------------------------------------------------------------------------- #
def _obs(extra_waiting_copy1):
    random.seed(1); np.random.seed(1)
    pn = make_next_activity(2, allow_postpone=False)
    P = {p._id: p for p in pn.places}
    for _ in range(extra_waiting_copy1):
        P['waiting_1'].put({'risk': 2, 'bad': 1})
    g = pn.get_graph_observation()['graph']
    part = pn.net_partition()
    comps = [part[b[2]._id] for b in pn.pn_actions]
    return pn, g, comps


@pytest.mark.parametrize("encoder", ["hgt", "type_embed", "type_embed_film", "aepn"])
def test_actor_logits_local_to_component(encoder):
    torch.manual_seed(0)
    pn_a, g_a, c_a = _obs(0)
    pn_b, g_b, c_b = _obs(3)
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3,
                        metadata=pn_a.make_metadata(), num_heads=2, global_context=False,
                        encoder=encoder)
    actor.eval()

    def raw_logits(g):
        with torch.no_grad():
            actor({'graph': g})
            x = actor.encoder(x_dict=g.x_dict, edge_index_dict=g.edge_index_dict,
                              input_size=-1, graph=g, params_iter=iter(actor.parameters()))
            return actor.decoder(x['a_transition']).flatten()

    la, lb = raw_logits(g_a), raw_logits(g_b)
    c0 = next(c for c in c_a if c == c_a[0])
    sel_a = [i for i, c in enumerate(c_a) if c == c0]
    sel_b = [i for i, c in enumerate(c_b) if c == c0]
    assert len(sel_a) == len(sel_b) > 0
    assert torch.equal(la[sel_a], lb[sel_b])      # bit-identical
    # and the perturbed component's logits did move (the test can fail)
    assert la.numel() != lb.numel() or not torch.equal(la, lb)


# --------------------------------------------------------------------------- #
# Critic: component pooling
# --------------------------------------------------------------------------- #
def test_critic_pool_mask():
    torch.manual_seed(0)
    pn, g, comps = _obs(0)
    critic = HeteroCritic(input_size=-1, hidden_size=32, num_layers=2,
                          metadata=pn.make_metadata(), num_heads=1, dropout=0.0)
    critic.eval()
    with torch.no_grad():
        v_all = critic({'graph': g})
        g['a_transition'].pool_mask = torch.ones(len(comps), dtype=torch.bool)
        v_mask_all = critic({'graph': g})
        assert torch.allclose(v_all, v_mask_all)           # K=1: same as before
        g['a_transition'].pool_mask = torch.tensor([c == comps[0] for c in comps])
        v_c = critic({'graph': g})
        assert v_c.shape == v_all.shape


# --------------------------------------------------------------------------- #
# Component-scoped postpone (Theorem 1, C3')
# --------------------------------------------------------------------------- #
def _env(n, scope, seed=3):
    random.seed(seed); np.random.seed(seed)
    pn = make_next_activity(n, allow_postpone=True)
    pn.postpone_scope = scope
    pn.length = 20
    env = AEPN_Env(pn)
    return env, env.reset()


def _is_pp(b):
    return isinstance(b[0], list) and b[0] == ['postpone']


def test_component_postpone_equals_global_at_k1():
    """One component: the two scopes give the identical trajectory and the
    identical observation graphs under the same action choices (run one after
    the other: both draw from the global random stream)."""
    def run(scope):
        env, obs = _env(1, scope)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(60):
            g = obs['graph']
            k = len(env.pn.pn_actions)
            a = int(rng.integers(k))
            n_pp = sum(_is_pp(b) for b in env.pn.pn_actions)
            obs, r, d, _, _ = env.step(a)
            out.append((g, k, n_pp, a, r, d, env.pn.clock))
            if d:
                break
        return out

    tg, tc = run('global'), run('component')
    assert len(tg) == len(tc) > 5
    assert any(a == k - 1 for (_, k, _, a, *_rest) in tg)      # postpone was exercised
    for (g, k, npg, a, r, d, t), (c, k2, npc, a2, r2, d2, t2) in zip(tg, tc):
        assert (k, npg, a, r, d, t) == (k2, npc, a2, r2, d2, t2)
        assert set(g.node_types) == set(c.node_types)
        for nt in g.node_types:
            assert torch.equal(g[nt].x, c[nt].x), nt
        assert set(g.edge_types) == set(c.edge_types)
        for et in g.edge_types:
            assert torch.equal(g[et].edge_index, c[et].edge_index), et


def test_component_postpone_blocks_only_its_component():
    env, obs = _env(2, 'component')
    part = env.pn.net_partition()
    comps = lambda: {(b[2].comp if _is_pp(b) else part[b[2]._id]) for b in env.pn.pn_actions}
    assert comps() == {0, 1}
    clock0 = env.pn.clock
    pp0 = next(i for i, b in enumerate(env.pn.pn_actions) if _is_pp(b) and b[2].comp == 0)
    env.step(pp0)
    # same decision round: only component 1 is offered, no time has passed
    assert env.pn.clock == clock0
    assert comps() == {1}
    assert 0 in env.pn.postponed_comps
    # let component 1 wait too: time advances; component 0 comes back only
    # after one of ITS events has fired
    while 0 in env.pn.postponed_comps:
        pp = [i for i, b in enumerate(env.pn.pn_actions) if _is_pp(b)]
        _, _, done, _, _ = env.step(pp[0])
        if done:
            break
    assert 0 not in env.pn.postponed_comps


@pytest.mark.parametrize("encoder", ["hgt", "type_embed", "type_embed_film", "aepn"])
def test_actor_logits_local_with_component_postpone(encoder):
    """Postpone logits are local too: component 0's action AND postpone
    logits are bit-identical when component 1's marking changes."""
    def obs(extra):
        random.seed(1); np.random.seed(1)
        pn = make_next_activity(2, allow_postpone=True)
        pn.postpone_scope = 'component'
        P = {p._id: p for p in pn.places}
        for _ in range(extra):
            P['waiting_1'].put({'risk': 2, 'bad': 1})
        g = pn.get_graph_observation()['graph']
        part = pn.net_partition()
        comps = [(b[2].comp if _is_pp(b) else part[b[2]._id]) for b in pn.pn_actions]
        return pn, g, comps

    torch.manual_seed(0)
    pn_a, g_a, c_a = obs(0)
    pn_b, g_b, c_b = obs(3)
    actor = HeteroActor(input_size=-1, hidden_size=32, num_layers=3,
                        metadata=pn_a.make_metadata(), num_heads=2, global_context=False,
                        encoder=encoder)
    actor.eval()

    def raw(g):
        with torch.no_grad():
            actor({'graph': g})
            x = actor.encoder(x_dict=g.x_dict, edge_index_dict=g.edge_index_dict,
                              input_size=-1, graph=g, params_iter=iter(actor.parameters()))
            return torch.cat([actor.decoder(x['a_transition']).flatten(),
                              actor.decoder(x['postpone']).flatten()])

    la, lb = raw(g_a), raw(g_b)
    sel_a = [i for i, c in enumerate(c_a) if c == 0]
    sel_b = [i for i, c in enumerate(c_b) if c == 0]
    assert len(sel_a) == len(sel_b) > 1          # actions + its postpone
    assert torch.equal(la[sel_a], lb[sel_b])
