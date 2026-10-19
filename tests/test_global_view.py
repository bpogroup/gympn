"""Tests for whole-net conditioning: every action is scored from the whole
visible net ('full': GymProblem.global_view='edges' plus
AEPNStack(global_attn=True, count_aggr=True): reverse edges, per-layer
attention over the nodes of the action's net component and count-aware
aggregation). Requirement, checked on the encoder's raw action logits of an
untrained actor: every logit has a nonzero gradient with respect to the
features of every observed node of its component, and adding a token
anywhere in it changes every logit. Without global_view both fail on the insurer allocation (an action
only sees what flows into it). The metadata lists exactly the observation's
types, and episodes and the flat path still work. 'full' is the default
(GymProblem, AEPNStack, the train CLI), and a policy trained on another view
is replayed on the graphs it knows."""
import os
import random
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite", "envs"))

from gympn.environment import AEPN_Env  # noqa: E402
from gympn.networks import HeteroActor, _encode_graph  # noqa: E402
from insurer_env import make_insurer, make_insurer_alloc  # noqa: E402

BUILDERS = {"alloc": lambda: make_insurer_alloc(1),
            "insurer": lambda: make_insurer(1),
            "insurer_pp": lambda: make_insurer(2, allow_postpone=True)}


def _env(build="alloc", view=True, flat=False, seed=5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pn = BUILDERS[build]()
    pn.length = 20
    pn.global_view = view
    pn.flat_obs = flat
    env = AEPN_Env(pn)
    obs = env.reset()
    return env, obs[0] if isinstance(obs, tuple) else obs


FULL_KW = {"global_attn": True, "count_aggr": True}
OFF_KW = {"global_attn": False, "count_aggr": False}
PN_VIEW = {"full": "edges", "off": False}


def _actor(env, encoder, enc_kw=None, init_seed=0):
    """enc_kw None: the AEPN encoder with its whole-net layers off (they
    are on by default)."""
    if enc_kw is None and encoder == "aepn":
        enc_kw = OFF_KW
    torch.manual_seed(init_seed)
    return HeteroActor(hidden_size=32, num_layers=2, metadata=env.pn.make_metadata(),
                       num_heads=2, dropout=0.0, encoder=encoder, encoder_kwargs=enc_kw).eval()


def _mode_env_actor(mode, build="alloc", encoder="aepn", flat=False):
    env, obs = _env(build, view=PN_VIEW[mode], flat=flat)
    return env, obs, _actor(env, encoder, FULL_KW if mode == "full" else None)


def _raw_logits(actor, graph):
    x = _encode_graph(actor.encoder, graph, -1, iter(actor.parameters()))
    return actor.decoder(x['a_transition']).flatten()


def _blind_pairs(actor, graph, own_component=False):
    """(action, node) pairs whose logit has zero gradient w.r.t. the node's
    features; nodes without features are skipped (they are covered by the
    multiplicity test). own_component: only nodes of the action's own net
    component (the observation's .comp) count."""
    _raw_logits(actor, graph)                                   # materialize lazy params
    if hasattr(graph, 'ntype'):
        xs = {'flat': graph.x.detach().clone().requires_grad_()}
        graph.x = xs['flat']
        comps = {'flat': getattr(graph, 'comp', None)}
        a_comp = None if comps['flat'] is None else comps['flat'][graph.a_idx]
    else:
        xs, comps = {}, {}
        for nt in graph.node_types:
            x = graph[nt].x
            if x is not None and x.numel() > 0:
                xs[nt] = graph[nt].x = x.detach().clone().requires_grad_()
                comps[nt] = getattr(graph[nt], 'comp', None)
        a_comp = getattr(graph['a_transition'], 'comp', None)
    logits = _raw_logits(actor, graph)
    blind, checked = [], 0
    for i in range(logits.numel()):
        grads = torch.autograd.grad(logits[i], list(xs.values()), retain_graph=True, allow_unused=True)
        for (nt, x), gr in zip(xs.items(), grads):
            for r in range(x.size(0)):
                if own_component and int(comps[nt][r]) != int(a_comp[i]):
                    continue
                checked += 1
                if gr is None or gr[r].abs().sum() == 0:
                    blind.append((i, nt, r))
    assert checked > 0
    return logits.numel(), blind


@pytest.mark.parametrize("build", ["alloc", "insurer", "insurer_pp"])
@pytest.mark.parametrize("flat", [False, True])
def test_every_logit_depends_on_every_node(build, flat):
    """Every node of the action's own net component (the attention stays
    inside it, for NF-GAE's C2); on the allocation, at allocation time, that
    is the whole net."""
    env, obs, actor = _mode_env_actor("full", build, "aepn", flat)
    n, blind = _blind_pairs(actor, obs['graph'], own_component=True)
    assert n > 1 and blind == []
    if build == "alloc":
        assert _blind_pairs(actor, obs['graph'])[1] == []


@pytest.mark.parametrize("flat", [False, True])
def test_full_mode_logits_are_local_to_their_component(flat):
    """NF-GAE Theorem 1, C2 under the default view: a binding's raw logit is
    bit-identical when another component's marking changes."""
    out = []
    for extra in (0, 2):
        env, _ = _env("insurer", view="edges", flat=flat)
        if extra:
            p = next(p for p in env.pn.places if p._id == "co_waiting_0")
            for _ in range(extra):
                p.put(dict(p.marking[0].value), time=p.marking[0].time)
        obs = env.pn.get_graph_observation()
        part = env.pn.net_partition()
        comps = [part[b[2]._id] for b in env.pn.pn_actions]
        if not extra:
            actor = _actor(env, "aepn", FULL_KW)
        with torch.no_grad():
            out.append((_raw_logits(actor, obs['graph']), comps))
    (la, ca), (lb, cb) = out
    moved = env.pn.net_place_partition()["co_waiting_0"]
    keep = [c for c in set(ca) if c != moved]
    assert keep and len(la) != len(lb) or not torch.equal(la, lb)   # the change is visible
    for c in keep:
        sa = [i for i, x in enumerate(ca) if x == c]
        sb = [i for i, x in enumerate(cb) if x == c]
        assert sa and torch.equal(la[sa], lb[sb])


def test_full_mode_adds_reverse_edges():
    env, obs, _ = _mode_env_actor("full")
    assert set(obs['graph'].edge_types) <= set(env.pn.make_metadata()[1])
    assert any(r.endswith('_rev') for _, r, _ in obs['graph'].edge_types)


@pytest.mark.parametrize("encoder,flat", [("aepn", False), ("aepn", True)])
def test_without_global_view_the_blind_spot_exists(encoder, flat):
    env, obs = _env("alloc", view=False, flat=flat)
    _, blind = _blind_pairs(_actor(env, encoder), obs['graph'])
    assert blind


def _logits_with_extra_case(mode, encoder="aepn", init_seed=0):
    """Raw logits of the uw/co/al actions before and after duplicating one
    waiting claims case (a token no such action consumes)."""
    out = []
    for extra in (False, True):
        env, _ = _env("alloc", view=PN_VIEW[mode])
        if extra:
            p = next(p for p in env.pn.places if p._id == "cl_waiting_0")
            tok = p.marking[0]
            p.put(dict(tok.value), time=tok.time)
        obs = env.pn.get_graph_observation()
        names = [str(b[2]._id) for b in env.pn.pn_actions]
        if not extra:
            actor = _actor(env, encoder, FULL_KW if mode == "full" else None, init_seed)
        with torch.no_grad():
            lg = _raw_logits(actor, obs['graph'])
        far = [i for i, nm in enumerate(names) if not nm.startswith(("cl_", "al_cl"))]
        out.append([float(lg[i]) for i in far])        # same actions, same order
    return out


def _moves(mode, seeds=(0, 1, 2)):
    """Per far action, the largest logit change over a few initializations.
    Float noise from a bigger graph is ~1e-8; a real dependency can be near
    zero for one initialization by cancellation, not for all of them."""
    best = None
    for sd in seeds:
        before, after = _logits_with_extra_case(mode, init_seed=sd)
        d = [abs(x - y) for x, y in zip(before, after)]
        best = d if best is None else [max(u, v) for u, v in zip(best, d)]
    return best


def test_token_multiplicity_reaches_every_action():
    moves = _moves("full")
    assert len(moves) > 5 and min(moves) > 1e-6


def test_token_multiplicity_is_invisible_without_a_global_view():
    assert max(_moves("off")) < 1e-6


@pytest.mark.parametrize("count_aggr", [False, True])
def test_count_aggregation_tells_one_neighbour_from_two(count_aggr):
    """Node 0 hears from one vs two identical neighbours. The attention
    aggregate alone is a weighted average, so both give node 0 the same
    embedding; with count_aggr they differ."""
    from gympn.networks import AEPNStack
    torch.manual_seed(0)
    enc = AEPNStack(in_channels=-1, hidden_channels=8, num_layers=1, heads=2, dropout=0.0,
                    metadata=(['a', 'b'], [('b', 'e', 'a')]), global_attn=False,
                    count_aggr=count_aggr).eval()
    def h0(k):
        x = torch.tensor([[1.0]] + [[0.5]] * k)
        t = torch.tensor([0] + [1] * k)
        ei = torch.stack([torch.arange(1, k + 1), torch.zeros(k, dtype=torch.long)])
        with torch.no_grad():
            return enc.encode_flat(x, t, ei, torch.zeros(k, dtype=torch.long))[0]
    same = torch.allclose(h0(1), h0(2), atol=1e-6)
    assert same != count_aggr


def test_flat_batches_and_episode_runs():
    env, obs, actor = _mode_env_actor("full", flat=True)
    gs = [obs['graph']]
    for _ in range(3):
        obs, *_ = env.step(0)
        gs.append(obs['graph'])
    with torch.no_grad():
        single = torch.cat([actor({'graph': g}).flatten() for g in gs])
        batched = actor({'graph': Batch.from_data_list(gs)}).flatten()
    assert torch.allclose(single, batched, atol=1e-5)
    done, steps = False, 0
    while not done and env.pn.pn_actions and steps < 2000:
        obs, r, done, *_ = env.step(0)
        steps += 1
    assert steps > 3


def test_full_mode_hetero_batch_matches_per_graph():
    """Global attention must not leak across the graphs of a batch."""
    env, obs, actor = _mode_env_actor("full")
    gs = [obs['graph']]
    for _ in range(3):
        obs, *_ = env.step(0)
        gs.append(obs['graph'])
    with torch.no_grad():
        single = [_raw_logits(actor, g) for g in gs]
        batched = _raw_logits(actor, Batch.from_data_list(gs))
    assert torch.allclose(torch.cat(single), batched, atol=1e-5)


def test_full_view_is_the_default(tmp_path):
    from gympn.networks import AEPNStack
    from gympn.train import make_parser, _view_encoder_kwargs
    pn = make_insurer_alloc(1)
    assert pn.global_view == 'edges'
    enc = AEPNStack(-1, 8, 1, pn.make_metadata())
    assert enc.global_attn and enc.count_aggr
    args = make_parser().parse_args([])
    assert args.global_view == 'edges'
    assert _view_encoder_kwargs(args, {"encoder": "aepn"}) is None          # AEPN defaults: on
    args.global_view = False
    assert _view_encoder_kwargs(args, {"encoder": "aepn"}) == OFF_KW
    assert _view_encoder_kwargs(args, {"encoder": "aepn", "encoder_kwargs": {"global_attn": True}}
                                )["global_attn"] is True                     # explicit wins
    assert _view_encoder_kwargs(args, {"encoder": "hgt"}) is None


@pytest.mark.parametrize("mode", ["off", "full"])
def test_a_policy_is_replayed_on_the_view_it_was_trained_on(mode):
    from gympn.networks import policy_global_view
    view = PN_VIEW[mode]
    env, obs, actor = _mode_env_actor(mode)
    assert policy_global_view(actor) == view
    pn = make_insurer_alloc(1)                                                # default view
    pn.match_policy_view(actor)
    assert pn.global_view == view
    g = pn.get_graph_observation()['graph']
    with torch.no_grad():
        assert torch.isfinite(actor({'graph': g})).all()


def test_component_mask_off_reads_across_components():
    """component_mask=False (allowed for a critic): attention crosses
    components, so a logit now moves when another component changes."""
    out = []
    for extra in (0, 2):
        env, _ = _env("insurer", view="edges")
        if extra:
            p = next(p for p in env.pn.places if p._id == "co_waiting_0")
            for _ in range(extra):
                p.put(dict(p.marking[0].value), time=p.marking[0].time)
        obs = env.pn.get_graph_observation()
        part = env.pn.net_partition()
        comps = [part[b[2]._id] for b in env.pn.pn_actions]
        if not extra:
            actor = _actor(env, "aepn", {**FULL_KW, "component_mask": False})
        with torch.no_grad():
            out.append((_raw_logits(actor, obs['graph']), comps))
    (la, ca), (lb, cb) = out
    moved = env.pn.net_place_partition()["co_waiting_0"]
    c = next(c for c in set(ca) if c != moved)
    sa = [i for i, x in enumerate(ca) if x == c]
    sb = [i for i, x in enumerate(cb) if x == c]
    assert not torch.equal(la[sa], lb[sb])


def test_critic_reads_across_components_by_default():
    """The actor's attention stays inside its component (C2); HeteroCritic's
    does not, unless asked to."""
    from gympn.networks import HeteroCritic
    env, _ = _env("insurer", view="edges")
    meta = env.pn.make_metadata()
    assert _actor(env, "aepn", FULL_KW).encoder.component_mask
    assert not HeteroCritic(metadata=meta).encoder.component_mask
    assert HeteroCritic(metadata=meta, encoder_kwargs={"component_mask": True}).encoder.component_mask
    assert not HeteroCritic(metadata=meta, encoder="hgt").encoder.__dict__.get("component_mask", False)
