"""The AEPN encoder computes its per-item transforms and its global attention
in a cheaper order than the textbook form (AEPNStack._transform, _global).
These tests pin it to the textbook form: per-item transforms through all
bases then mixed, attention by nn.MultiheadAttention over every group padded
to the largest group. Both orders of _transform (basis and per-type) are
checked, on a graph with several attention groups of different sizes, for
the outputs and for the gradients of every parameter."""
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax, to_dense_batch

from gympn.networks import AEPNStack

N_TYPES, E_TYPES = 4, 6


def _encoder(seed=0, heads=2):
    torch.manual_seed(seed)
    nts = [f"n{i}" for i in range(N_TYPES)]
    ets = [(nts[i % N_TYPES], f"r{i}", nts[(i + 1) % N_TYPES]) for i in range(E_TYPES)]
    return AEPNStack(in_channels=-1, hidden_channels=16, num_layers=2, metadata=(nts, ets),
                     heads=heads, dropout=0.0, num_bases=3).eval()


def _graph(seed=1, n=60, e=200, groups=(1, 3, 7, 20, 29)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    t = torch.randint(0, N_TYPES, (n,), generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    et = torch.randint(0, E_TYPES, (e,), generator=g)
    batch = torch.cat([torch.full((k,), i) for i, k in enumerate(groups)])
    batch = batch[torch.randperm(n, generator=g)]          # groups interleaved, as components are
    return x, t, ei, et, batch


def _reference(enc, x, t, edge_index, e_t, batch):
    """The textbook form of AEPNStack.encode_flat."""
    from gympn.networks import _project_in
    H, B, nh = enc.hidden_channels, enc.num_bases, enc.heads
    d = H // nh

    def mix(h, bases, coef, idx):
        return torch.einsum('mb,mbh->mh', coef[idx], (h @ bases).view(h.size(0), B, H))

    h = _project_in(x, t, enc.in_weight, enc.in_bias)
    src, dst = edge_index
    for li in range(enc.num_layers):
        N = h.size(0)
        agg = h.new_zeros(N, H)
        if src.numel() > 0:
            Pk = (h @ enc.k_bases[li]).view(N, B, H)
            Pv = (h @ enc.v_bases[li]).view(N, B, H)
            k = torch.einsum('eb,ebh->eh', enc.k_coef[li][e_t], Pk[src])
            v = torch.einsum('eb,ebh->eh', enc.v_coef[li][e_t], Pv[src])
            q = mix(h, enc.q_bases[li], enc.q_coef[li], t)[dst]
            score = (q.view(-1, nh, d) * k.view(-1, nh, d)).sum(-1) * enc.p_rel[li][e_t] / d ** 0.5
            alpha = pyg_softmax(score, dst, num_nodes=N)
            agg = agg.index_add_(0, dst, (v.view(-1, nh, d) * alpha.unsqueeze(-1)).view(-1, H))
            deg = h.new_zeros(N).index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            mean_v = h.new_zeros(N, H).index_add_(0, dst, v) / deg.clamp(min=1).unsqueeze(-1)
            agg = agg + enc.cnt_lin[li](mean_v * torch.log1p(deg).unsqueeze(-1))
        out = mix(F.gelu(agg), enc.o_bases[li], enc.o_coef[li], t)
        perm = torch.argsort(batch, stable=True)
        hd, mask = to_dense_batch(h[perm], batch[perm])
        att, _ = enc.g_attn[li](hd, hd, hd, key_padding_mask=~mask, need_weights=False)
        glob = torch.empty_like(h)
        glob[perm] = att[mask]
        out = out + glob
        if li > 0:
            gate = torch.sigmoid(enc.skip[li][t]).unsqueeze(-1)
            out = gate * out + (1 - gate) * h
        out = out + mix(h, enc.r_bases[li], enc.r_coef[li], t)
        h = enc.act(out)
    return h


def _out_and_grads(fn, enc):
    enc.zero_grad()
    out = fn()
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
    (out * w).sum().backward()
    return out.detach(), {k: p.grad.clone() for k, p in enc.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("order", ["basis", "typed"])
@pytest.mark.parametrize("heads", [1, 2])
def test_encoder_matches_textbook_form(order, heads, monkeypatch):
    enc = _encoder(heads=heads)
    x, t, ei, et, batch = _graph()
    monkeypatch.setattr(AEPNStack, "TYPED_MIN_ROWS", 10 ** 9 if order == "basis" else 0)
    ref, g_ref = _out_and_grads(lambda: _reference(enc, x, t, ei, et, batch), enc)
    new, g_new = _out_and_grads(lambda: enc.encode_flat(x, t, ei, et, batch), enc)
    assert torch.allclose(ref, new, atol=1e-5, rtol=1e-4)
    assert g_ref.keys() == g_new.keys()
    for k in g_ref:
        assert torch.allclose(g_ref[k], g_new[k], atol=1e-5, rtol=1e-3), k


def test_one_group_and_no_edges():
    """A single graph (no batch index) and a graph without edges."""
    enc = _encoder()
    x, t, ei, et, _ = _graph(groups=(60,))
    one = torch.zeros(60, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(enc.encode_flat(x, t, ei, et), _reference(enc, x, t, ei, et, one), atol=1e-5)
        empty = torch.zeros((2, 0), dtype=torch.long)
        no_e = torch.zeros(0, dtype=torch.long)
        assert torch.allclose(enc.encode_flat(x, t, empty, no_e), _reference(enc, x, t, empty, no_e, one), atol=1e-5)
