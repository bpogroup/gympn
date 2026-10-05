"""Tests for TypeEmbedStack (gympn/networks.py), the shared-weight alternative
to HGTStack: default unchanged, batched == per-graph forwards, gradients reach
the type embeddings, and type identity is kept (two places with identical
features and edges still get different embeddings). Locality of action logits
is covered for both encoders in test_nfgae.py."""
import os
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "examples", "paper_examples", "suite"))

from gympn.networks import AEPNStack, HeteroActor, HeteroCritic, HGTStack, TypeEmbedStack  # noqa: E402
from bpm_envs import make_next_activity  # noqa: E402


def _graphs(n, k=3):
    out = []
    for s in range(k):
        random.seed(s); np.random.seed(s)
        pn = make_next_activity(n, allow_postpone=True)
        pn.postpone_scope = 'component'
        P = {p._id: p for p in pn.places}
        for _ in range(s):
            P['waiting_0'].put({'risk': 2, 'bad': 1})
        out.append(pn.get_graph_observation()['graph'])
    return pn.make_metadata(), out


def test_default_encoder_is_aepn():
    meta, _ = _graphs(1, 1)
    assert isinstance(HeteroActor(metadata=meta).encoder, AEPNStack)
    assert isinstance(HeteroCritic(metadata=meta).encoder, AEPNStack)
    assert isinstance(HeteroActor(metadata=meta, encoder="hgt").encoder, HGTStack)
    assert HeteroActor(metadata=meta, encoder_kwargs={"num_bases": 4}).encoder.num_bases == 4
    assert isinstance(HeteroActor(metadata=meta, encoder="type_embed").encoder, TypeEmbedStack)
    assert HeteroActor(metadata=meta, encoder="type_embed_film").encoder.film
    assert isinstance(HeteroActor(metadata=meta, encoder="aepn").encoder, AEPNStack)
    with pytest.raises(ValueError):
        HeteroActor(metadata=meta, encoder="nope")


@pytest.mark.parametrize("encoder", ["type_embed", "type_embed_film", "aepn"])
@pytest.mark.parametrize("n", [1, 2])
def test_batched_matches_single(n, encoder):
    torch.manual_seed(0)
    meta, gs = _graphs(n)
    actor = HeteroActor(hidden_size=32, num_layers=3, metadata=meta, num_heads=2,
                        dropout=0.0, encoder=encoder).eval()
    critic = HeteroCritic(hidden_size=32, num_layers=3, metadata=meta, num_heads=2,
                          dropout=0.0, encoder=encoder).eval()
    with torch.no_grad():
        single_p = [actor({'graph': g}).flatten() for g in gs]
        single_v = torch.cat([critic({'graph': g}).flatten() for g in gs])
        b = Batch.from_data_list(gs)
        batch_p = actor({'graph': b}).flatten()
        batch_v = critic({'graph': b}).flatten()
    # the actor's output is ordered [all a_transition ; all postpone]
    na = [g['a_transition'].x.size(0) for g in gs]
    npp = [g['postpone'].x.size(0) for g in gs]
    a_parts = torch.split(batch_p[:sum(na)], na)
    p_parts = torch.split(batch_p[sum(na):], npp)
    for sp, a, p in zip(single_p, a_parts, p_parts):
        assert torch.allclose(sp, torch.cat([a, p]), atol=1e-6)
    assert torch.allclose(single_v, batch_v, atol=1e-6)


def test_gradients_reach_embeddings():
    torch.manual_seed(0)
    meta, gs = _graphs(2, 1)
    critic = HeteroCritic(hidden_size=32, num_layers=2, metadata=meta, num_heads=2,
                          encoder="type_embed")
    critic({'graph': gs[0]}).sum().backward()
    enc = critic.encoder
    assert enc.in_weight.grad is not None and enc.in_weight.grad.abs().sum() > 0
    assert enc.node_emb[0].weight.grad.abs().sum() > 0
    assert enc.edge_emb[0].weight.grad.abs().sum() > 0


def test_places_stay_distinct():
    """waiting_0 and waiting_1 with identical tokens and mirror-image edges
    must still embed differently: the type identity is not merged."""
    torch.manual_seed(0)
    random.seed(0); np.random.seed(0)
    pn = make_next_activity(2, allow_postpone=False)
    g = pn.get_graph_observation()['graph']
    assert torch.equal(g['waiting_0'].x, g['waiting_1'].x)
    enc = TypeEmbedStack(-1, 32, 2, pn.make_metadata(), heads=2, dropout=0.0)
    with torch.no_grad():
        h = enc(g.x_dict, g.edge_index_dict)
    assert not torch.allclose(h['waiting_0'], h['waiting_1'])


def test_aepn_gradients_and_distinct_relations():
    """Gradients reach the relation coefficients, and two relations get
    different transforms (their coefficient rows differ at init)."""
    torch.manual_seed(0)
    meta, gs = _graphs(2, 1)
    critic = HeteroCritic(hidden_size=32, num_layers=2, metadata=meta, num_heads=2,
                          encoder="aepn")
    critic({'graph': gs[0]}).sum().backward()
    enc = critic.encoder
    for p in (enc.in_weight, enc.k_coef[0], enc.v_coef[0], enc.q_coef[0], enc.o_coef[0], enc.k_bases[0],
              enc.r_coef[0], enc.p_rel[0], enc.skip[1]):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert not torch.allclose(enc.v_coef[0][0], enc.v_coef[0][1])


def test_aepn_places_stay_distinct():
    torch.manual_seed(0)
    random.seed(0); np.random.seed(0)
    pn = make_next_activity(2, allow_postpone=False)
    g = pn.get_graph_observation()['graph']
    enc = AEPNStack(-1, 32, 2, pn.make_metadata(), heads=2, dropout=0.0)
    with torch.no_grad():
        h = enc(g.x_dict, g.edge_index_dict)
    assert not torch.allclose(h['waiting_0'], h['waiting_1'])


def _aepn_reference(enc, x_dict, edge_index_dict):
    """AEPNStack.forward written the slow, obvious way, on the heterogeneous
    dicts: explicit W_r = sum_b a[r, b] V_b per relation, and the attention
    softmax computed node by node."""
    H, B, nh = enc.hidden_channels, enc.num_bases, enc.heads
    d = H // nh
    ti, ri = enc._ntype_idx, enc._etype_idx

    def W(bases, coef, i):
        return sum(coef[i, b] * bases[:, b * H:(b + 1) * H] for b in range(B))

    h = {nt: x @ enc.in_weight[ti[nt], :x.size(1)] + enc.in_bias[ti[nt]]
         for nt, x in x_dict.items() if x.size(0) > 0}
    for li in range(enc.num_layers):
        incoming = {nt: [] for nt in h}          # dst type -> [(dst idx, score [nh], value [H])]
        for et, ei in edge_index_dict.items():
            s_t, _, d_t = et
            if ei.numel() == 0 or s_t not in h or d_t not in h:
                continue
            r = ri[tuple(et)]
            Wk, Wv = W(enc.k_bases[li], enc.k_coef[li], r), W(enc.v_bases[li], enc.v_coef[li], r)
            Wq = W(enc.q_bases[li], enc.q_coef[li], ti[d_t])
            for s, t_ in ei.t().tolist():
                k, v, q = h[s_t][s] @ Wk, h[s_t][s] @ Wv, h[d_t][t_] @ Wq
                score = (q.view(nh, d) * k.view(nh, d)).sum(-1) * enc.p_rel[li][r] / d ** 0.5
                incoming[d_t].append((t_, score, v))
        new = {}
        for nt, hn in h.items():
            Wo = W(enc.o_bases[li], enc.o_coef[li], ti[nt])
            Wr = W(enc.r_bases[li], enc.r_coef[li], ti[nt])
            rows = []
            for i in range(hn.size(0)):
                inc = [(sc, v) for (j, sc, v) in incoming[nt] if j == i]
                agg = torch.zeros(H)
                if inc:
                    a = torch.softmax(torch.stack([sc for sc, _ in inc]), dim=0)     # [deg, nh]
                    for (sc, v), a_e in zip(inc, a):
                        agg = agg + (v.view(nh, d) * a_e.unsqueeze(-1)).view(H)
                out = F.gelu(agg) @ Wo
                if li > 0:                                   # HGTConv: no gate on the first layer
                    g = torch.sigmoid(enc.skip[li][ti[nt]])
                    out = g * out + (1 - g) * hn[i]
                out = out + hn[i] @ Wr                       # HGTStack's skip_proj
                rows.append(torch.relu(out))
            new[nt] = torch.stack(rows)
        h = new
    return h


@pytest.mark.parametrize("bases", [4, 8])
@pytest.mark.parametrize("heads", [1, 2])
def test_aepn_matches_reference(heads, bases):
    torch.manual_seed(0)
    meta, gs = _graphs(2, 2)
    g = gs[1]
    enc = AEPNStack(-1, 16, 3, meta, heads=heads, dropout=0.0, num_bases=bases).eval()
    with torch.no_grad():                    # move gates/scales off their init so they are exercised
        for li in range(3):
            enc.skip[li].normal_(); enc.p_rel[li].uniform_(0.5, 2.0)
    with torch.no_grad():
        fast = enc(g.x_dict, g.edge_index_dict)
        ref = _aepn_reference(enc, g.x_dict, g.edge_index_dict)
    for nt, v in ref.items():
        assert torch.allclose(fast[nt], v, atol=1e-5), nt


# --------------------------------------------------------------------------- #
# Flat observations (gympn/flat_graph.py): same weights, same outputs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("encoder", ["type_embed", "type_embed_film", "aepn"])
@pytest.mark.parametrize("n", [1, 2])
def test_flat_matches_hetero(n, encoder):
    from gympn.flat_graph import FlatGraph, hetero_to_flat
    torch.manual_seed(0)
    meta, gs = _graphs(n)
    width = max(max(x.size(-1) for x in g.x_dict.values()) for g in gs)
    flats = [hetero_to_flat(g, meta, width) for g in gs]
    assert all(isinstance(f, FlatGraph) for f in flats)
    actor = HeteroActor(hidden_size=32, num_layers=3, metadata=meta, num_heads=2,
                        dropout=0.0, encoder=encoder).eval()
    critic = HeteroCritic(hidden_size=32, num_layers=3, metadata=meta, num_heads=2,
                          dropout=0.0, encoder=encoder).eval()
    with torch.no_grad():
        for g, f in zip(gs, flats):                          # one graph at a time
            assert torch.allclose(actor({'graph': g}), actor({'graph': f}), atol=1e-6)
            assert torch.allclose(critic({'graph': g}), critic({'graph': f}), atol=1e-6)
        bh, bf = Batch.from_data_list(gs), Batch.from_data_list(flats)   # batched
        assert torch.allclose(actor({'graph': bh}), actor({'graph': bf}), atol=1e-6)
        assert torch.allclose(critic({'graph': bh}), critic({'graph': bf}), atol=1e-6)
        # nfgae pooling masks
        for g, f in zip(gs, flats):
            na, npp = g['a_transition'].x.size(0), g['postpone'].x.size(0)
            ma = torch.arange(na) % 2 == 0
            mp = torch.ones(npp, dtype=torch.bool)
            g['a_transition'].pool_mask, g['postpone'].pool_mask = ma, mp
            f.a_pool_mask, f.p_pool_mask = ma, mp
        bh, bf = Batch.from_data_list(gs), Batch.from_data_list(flats)
        assert torch.allclose(critic({'graph': bh}), critic({'graph': bf}), atol=1e-6)


def test_flat_rejected_by_hgt():
    from gympn.flat_graph import hetero_to_flat
    meta, gs = _graphs(1, 1)
    with pytest.raises(TypeError):
        HeteroActor(hidden_size=32, metadata=meta, encoder="hgt")({'graph': hetero_to_flat(gs[0], meta)})


def test_simulator_flat_obs_fixed_width():
    """With flat_obs every observation of an episode is a FlatGraph of one
    width, and matches the converted HeteroData observation."""
    from gympn.environment import AEPN_Env
    from gympn.flat_graph import FlatGraph, hetero_to_flat
    random.seed(0); np.random.seed(0)
    pn = make_next_activity(2, allow_postpone=True)
    pn.postpone_scope = 'component'
    pn.length = 20
    pn.flat_obs = True
    env = AEPN_Env(pn)
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    widths = set()
    for _ in range(60):
        g = obs['graph']
        assert isinstance(g, FlatGraph)
        widths.add(g.x.size(1))
        assert g.a_idx.numel() + g.p_idx.numel() == len(env.pn.pn_actions)
        pn_ = env.pn
        pn_.flat_obs = False
        h = pn_.get_graph_observation()['graph']
        pn_.flat_obs = True
        ref = hetero_to_flat(h, pn_.make_metadata(), g.x.size(1))
        for k in ('x', 'ntype', 'edge_index', 'etype', 'a_idx', 'p_idx'):
            assert torch.equal(getattr(g, k), getattr(ref, k)), k
        k = len(env.pn.pn_actions)
        if k == 0:
            break
        obs, _, done, _, _ = env.step(np.random.randint(k))
        if done:
            break
    assert len(widths) == 1
