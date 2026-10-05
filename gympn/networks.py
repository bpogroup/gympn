import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, TransformerConv, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

from gympn.flat_graph import cat_flat, feature_width, flatten_hetero, is_flat


# ---------------------------------------------------------------------
# Helper: ensure HGTConv never sees an empty node-type tensor.
# Returns a patched x_dict and the set of node types we "dummied".
# ---------------------------------------------------------------------
def _prepare_x_dict_for_conv(x_dict, input_size, graph=None, params_iter=None,
                              seen_dims=None):
    device = torch.device('cpu')
    dtype = torch.float32
    if params_iter is not None:
        try:
            p = next(params_iter)
            device, dtype = p.device, p.dtype
        except Exception:
            pass

    # Infer a global fallback input dim
    input_dim = None
    if isinstance(input_size, int) and input_size > 0:
        input_dim = input_size
    else:
        for v in (x_dict or {}).values():
            if v is not None and v.numel() > 0:
                input_dim = v.size(-1)
                break
        if input_dim is None and graph is not None:
            try:
                a = graph.x_dict.get('a_transition', None)
                if a is not None and a.size(0) > 0:
                    input_dim = a.size(-1)
            except Exception:
                pass
    if input_dim is None:
        input_dim = 1
        warnings.warn('Could not infer input feature dim; using fallback dim=1 for dummy nodes')

    x_fixed, dummies = {}, set()
    node_types = list((x_dict or {}).keys())
    if len(node_types) == 0 and graph is not None:
        try:
            node_types = graph.metadata()[0]
        except Exception:
            node_types = ['a_transition']

    for ntype in node_types:
        x = (x_dict or {}).get(ntype)
        if x is None or x.size(0) == 0:
            # Prefer the width already carried by an empty-but-present tensor: an
            # empty node type arrives as [0, W] with its CORRECT feature width W.
            # Only fall back to the per-type seen dim, then the global dim, when no
            # width is available. Using the global fallback for a present [0, W]
            # type was the bug: it gave e.g. busy_s ([0, 3]) a width-2 dummy,
            # locking its lazily-sized input Linear to 2, which then crashed when a
            # populated busy_s ([N, 3]) arrived. Seed-dependent because it hinged on
            # whether a type was first seen empty or populated.
            if x is not None and x.dim() == 2 and x.size(-1) > 0:
                dim = x.size(-1)
            else:
                dim = (seen_dims or {}).get(ntype, input_dim)
            x_fixed[ntype] = torch.zeros((1, dim), device=device, dtype=dtype)
            dummies.add(ntype)
        else:
            x_fixed[ntype] = x
    return x_fixed, dummies


# ---------------------------------------------------------------------
# Shared global-context pooling (action/postpone nodes -> one vector per
# graph). HeteroCritic has always used this to build its state value;
# HeteroActor's optional `global_context` (see below) reuses the SAME
# pooling so a node's context is exactly what the critic already sees,
# not a second independently-tuned summary.
# ---------------------------------------------------------------------
def _batch_index_for(graph, ntype, n, device):
    """Per-node graph-membership index for ntype (which graph, in a batch,
    each node belongs to), defaulting to all-zeros (single graph) when the
    type carries no PyG `.batch` attribute (the common case outside a
    batched training update)."""
    if is_flat(graph):
        idx = graph.a_idx if ntype == 'a_transition' else graph.p_idx
        b = getattr(graph, 'batch', None)
        if b is None:
            return torch.zeros(idx.numel(), dtype=torch.int64, device=device)
        return b[idx].to(device)
    try:
        if ntype in graph.x_dict and hasattr(graph[ntype], 'batch'):
            return graph[ntype].batch
    except Exception:
        pass
    return torch.zeros(n, dtype=torch.int64, device=device)


def _pool_action_postpone(x_enc, graph):
    """global_max_pool over [a_transition, postpone] (whichever are present
    and non-empty), keyed by per-graph batch index: one context vector per
    graph in the batch, or None if there is nothing to pool (no live
    bindings at all -- shouldn't happen in practice, guarded defensively)."""
    parts, idx_parts = [], []
    for ntype in ('a_transition', 'postpone'):
        x = x_enc.get(ntype)
        if x is not None and x.numel() > 0:
            parts.append(x)
            idx_parts.append(_batch_index_for(graph, ntype, x.size(0), x.device))
    if not parts:
        return None
    concat_nodes = torch.cat(parts, dim=0)
    pool_index = torch.cat(idx_parts, dim=0)
    return global_max_pool(concat_nodes, pool_index)


def _pool_mask(graph, ntype):
    """nfgae's per-node pooling mask for ntype, or None when absent."""
    if is_flat(graph):
        return getattr(graph, 'a_pool_mask' if ntype == 'a_transition' else 'p_pool_mask', None)
    try:
        if ntype in graph.node_types:
            return getattr(graph[ntype], 'pool_mask', None)
    except Exception:
        pass
    return None


def _encode_graph(encoder, graph, input_size, params):
    """Node embeddings for the action-side types, from a HeteroData or a
    FlatGraph (the latter needs a flat encoder: TypeEmbedStack / AEPNStack)."""
    if is_flat(graph):
        if not hasattr(encoder, 'encode_flat'):
            raise TypeError("flat observations need encoder='type_embed', 'type_embed_film' or 'aepn'")
        dev = encoder.in_weight.device
        h = encoder.encode_flat(graph.x.to(dev), graph.ntype.to(dev),
                                graph.edge_index.to(dev), graph.etype.to(dev))
        x_enc = {'a_transition': h[graph.a_idx.to(dev)], 'postpone': h[graph.p_idx.to(dev)]}
    else:
        x_enc = encoder(x_dict=graph.x_dict, edge_index_dict=graph.edge_index_dict,
                        input_size=input_size, graph=graph, params_iter=params)
    for k, v in x_enc.items():
        if v is not None:
            x_enc[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return x_enc


# ---------------------------------------------------------------------
# HGT stack: L layers, dropout, per-node-type residual projections.
# ---------------------------------------------------------------------
class HGTStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        metadata,
        heads: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.residual = bool(residual)
        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.hidden_channels = int(hidden_channels)

        # HGT layers
        self.convs = nn.ModuleList()
        in_c = in_channels
        for _ in range(self.num_layers):
            self.convs.append(HGTConv(in_c, hidden_channels, metadata, heads=heads))
            in_c = hidden_channels

        # Per-layer, per-node-type residual projections (lazy → no shape headaches)
        if self.residual:
            self.skip_proj = nn.ModuleList([
                nn.ModuleDict({nt: nn.LazyLinear(hidden_channels, bias=False) for nt in self.node_types})
                for _ in range(self.num_layers)
            ])
        else:
            self.skip_proj = None

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.GELU()

        self.drop = nn.Dropout(self.dropout)

        # Track per-node-type input feature dims so that dummy tensors for
        # temporarily-empty types use the correct width after LazyLinear
        # layers have already been materialized.
        self._input_dims: Dict[str, int] = {}

    def forward(self, x_dict, edge_index_dict, input_size, graph, params_iter):
        out = x_dict

        # Record input feature dims from non-empty types (persists across calls)
        for ntype, x in (out or {}).items():
            if x is not None and x.numel() > 0:
                self._input_dims[ntype] = x.size(-1)

        for li, conv in enumerate(self.convs):
            # First layer: node types may have heterogeneous input dims →
            # use the recorded per-type dims.  Later layers: all types share
            # hidden_channels, so the default single-dim inference is fine.
            layer_dims = self._input_dims if li == 0 else None
            x_clean, dummies = _prepare_x_dict_for_conv(
                out, input_size, graph=graph, params_iter=params_iter,
                seen_dims=layer_dims,
            )
            y = conv(x_clean, edge_index_dict)  # dict -> dict

            # HGTConv only emits node types that are a DESTINATION of some
            # metadata edge type; a source-only place (e.g. fed purely by the
            # initial marking, never by an event) vanishes from the dict after
            # layer 1, and the next layer's k_dict[src] lookup crashes with a
            # KeyError. Carry such types forward with the correct node count
            # (zeros; the residual projection below re-injects their features).
            for ntype, x_prev in x_clean.items():
                if y.get(ntype) is None:
                    y[ntype] = x_prev.new_zeros((x_prev.size(0), self.hidden_channels))

            # Residual (only if enabled)
            if self.residual:
                y_res = {}
                for ntype, y_nt in y.items():
                    if y_nt is None or y_nt.numel() == 0:
                        y_res[ntype] = y_nt
                        continue
                    x_nt = x_clean[ntype]
                    x_proj = self.skip_proj[li][ntype](x_nt)
                    # If shapes match, add skip; if they don't (edge case with dummies), skip adding
                    if x_proj.size(0) == y_nt.size(0):
                        y_nt = y_nt + x_proj
                    y_res[ntype] = y_nt
                y = y_res

            # Activation + Dropout
            for ntype in list(y.keys()):
                if y[ntype] is None or y[ntype].numel() == 0:
                    continue
                y[ntype] = self.drop(self.act(y[ntype]))

            # Restore empties for types we dummied
            for ntype in dummies:
                if ntype in y:
                    y_nt = y[ntype]
                    y[ntype] = y_nt.new_empty((0, y_nt.size(-1)))

            out = y
        return out


# ---------------------------------------------------------------------
# Shared by the flat encoders (TypeEmbedStack, AEPNStack): they run on the
# flattened graph (gympn/flat_graph.py). A HeteroData input is flattened on
# every forward; a FlatGraph input (GymProblem.flat_obs) arrives flat already.
# ---------------------------------------------------------------------
def _project_in(x, t, in_weight, in_bias):
    """Per-type input projection, gathered per node. Only the first x.size(1)
    rows of each type's matrix are used (features are zero-padded to the
    widest type, never to max_in_features)."""
    w = x.size(1)
    if w > in_weight.size(1):
        raise ValueError(f"{w} input features > max_in_features={in_weight.size(1)}")
    x = x.to(dtype=in_weight.dtype)
    return torch.einsum('nf,nfh->nh', x, in_weight[:, :w][t]) + in_bias[t]


def _encode_hetero(enc, x_dict, edge_index_dict):
    """HeteroData dicts -> dict of node embeddings by type, via enc.encode_flat."""
    x_dict, edge_index_dict = x_dict or {}, edge_index_dict or {}
    device, H = enc.in_weight.device, enc.hidden_channels
    w = feature_width(x_dict)
    xs, tidx, eis, eidx, spans = flatten_hetero(x_dict, edge_index_dict, enc._ntype_idx, enc._etype_idx, w)
    if not xs:
        return {nt: torch.empty((0, H), device=device) for nt in spans}
    x, t, edge_index, e_t = cat_flat(xs, tidx, eis, eidx, w, device=device, dtype=enc.in_weight.dtype)
    h = enc.encode_flat(x, t, edge_index.to(device), e_t.to(device))
    return {nt: h[o:o + n] if n > 0 else h.new_empty((0, H)) for nt, (o, n) in spans.items()}


# ---------------------------------------------------------------------
# Type-embedding stack: a drop-in alternative to HGTStack.
#
# HGT keeps separate weights per node type and per edge type, and loops over
# them in Python inside every layer. Here every place is its own node type, so
# those counts grow with the number of net copies (N=8 next_activity: 35 node
# types, 99 edge types), and the loops, not the arithmetic, dominate wall time.
#
# This stack flattens the heterogeneous graph into one homogeneous graph once
# per forward, then runs L shared TransformerConv layers over it:
#   * per-type input projection: each node type keeps its OWN input matrix,
#     gathered per node (no Python loop), so differently-sized and
#     differently-meaning features are read exactly as HGT's first layer would;
#   * a learned per-layer node-type embedding is added before every layer, so
#     every place stays individually identifiable (nothing is merged);
#   * a learned per-layer edge-type embedding enters the attention keys and
#     values (TransformerConv edge_dim), playing the role of HGT's
#     relation-specific matrices.
# Message flow is unchanged: the same directed edges, so action logits stay as
# local as under HGT. Output is a dict by node type, like HGTStack's.
#
# film=True adds per-layer, per-type FiLM on each layer's output,
# y * (1 + gamma[type]) + beta[type]: a multiplicative type-specific
# transform in the role of HGT's per-type output projection, instead of
# conditioning on type only additively.
# ---------------------------------------------------------------------
class TypeEmbedStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        metadata,
        heads: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        activation: str = "relu",
        max_in_features: int = 32,
        film: bool = False,
    ):
        super().__init__()
        if metadata is None:
            raise ValueError("TypeEmbedStack needs the graph metadata (node and edge types)")
        self.num_layers = int(num_layers)
        self.hidden_channels = int(hidden_channels)
        self.residual = bool(residual)
        self.max_in_features = int(max_in_features)
        self.node_types = list(metadata[0])
        self.edge_types = [tuple(e) for e in metadata[1]]
        self._ntype_idx = {nt: i for i, nt in enumerate(self.node_types)}
        self._etype_idx = {et: i for i, et in enumerate(self.edge_types)}
        n_nt, n_et, H = len(self.node_types), len(self.edge_types), self.hidden_channels

        # Per-type input projection, stored as one tensor and gathered per node.
        # Features narrower than max_in_features are zero-padded; the padded rows
        # of the weight never receive gradient.
        self.in_weight = nn.Parameter(torch.empty(n_nt, self.max_in_features, H))
        self.in_bias = nn.Parameter(torch.zeros(n_nt, H))
        bound = 1.0 / (self.max_in_features ** 0.5)
        nn.init.uniform_(self.in_weight, -bound, bound)

        self.node_emb = nn.ModuleList([nn.Embedding(n_nt, H) for _ in range(self.num_layers)])
        self.edge_emb = nn.ModuleList([nn.Embedding(max(n_et, 1), H) for _ in range(self.num_layers)])
        for emb in list(self.node_emb) + list(self.edge_emb):
            nn.init.normal_(emb.weight, std=0.1)

        self.film = bool(film)
        if self.film:
            self.film_gamma = nn.ModuleList([nn.Embedding(n_nt, H) for _ in range(self.num_layers)])
            self.film_beta = nn.ModuleList([nn.Embedding(n_nt, H) for _ in range(self.num_layers)])
            for emb in self.film_gamma:
                nn.init.normal_(emb.weight, std=0.5)
            for emb in self.film_beta:
                nn.init.normal_(emb.weight, std=0.1)

        if H % heads != 0:
            raise ValueError(f"hidden_channels={H} not divisible by heads={heads}")
        self.convs = nn.ModuleList([
            TransformerConv(H, H // heads, heads=heads, concat=True, edge_dim=H, root_weight=True)
            for _ in range(self.num_layers)
        ])

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

    def encode_flat(self, x, t, edge_index, e_t):
        """Flat graph -> node embeddings [N, H]."""
        h = _project_in(x, t, self.in_weight, self.in_bias)
        for li, conv in enumerate(self.convs):
            z = h + self.node_emb[li](t)
            y = conv(z, edge_index, self.edge_emb[li](e_t))
            if self.film:
                y = y * (1.0 + self.film_gamma[li](t)) + self.film_beta[li](t)
            if self.residual:
                y = y + h
            h = self.drop(self.act(y))
        return h

    def forward(self, x_dict, edge_index_dict, input_size=None, graph=None, params_iter=None):
        return _encode_hetero(self, x_dict, edge_index_dict)


# ---------------------------------------------------------------------
# AEPN stack: HGT's per-type / per-relation expressiveness, without the
# per-type Python loops.
#
# TypeEmbedStack shares one attention layer and tells relations apart only by
# an additive edge embedding, and it learned worse than HGT. This stack keeps
# HGT's structure but vectorizes it with basis decomposition (as in R-GCN):
#   * relation-specific keys and messages: W_r = sum_b a[r, b] V_b. Each node is
#     projected through the B shared bases once ([N, B, H]); each edge mixes
#     its source's B projections with its relation's coefficients. In an AEPN
#     each input arc of a binding is its own relation, so every arc role
#     (e.g. the waiting token vs the employee token of approve_0) gets its own
#     message transform;
#   * per-node-type queries and output projection, also via bases;
#   * per-relation, per-head attention scale p_rel (multiplies q.k, init 1),
#     softmax over each node's incoming edges (as HGTConv);
#   * per-type gated skip, sigmoid(skip[type]) (init 1 -> 0.73 on the new
#     state), from the second layer on (HGTConv gates only when input and
#     output widths match, i.e. not on the raw-feature layer);
#   * with residual=True, a per-type projection of the layer input added on top,
#     as HGTStack's skip_proj (also via bases).
# The edges are the same directed edges, so action logits stay as local as
# under HGT.
# ---------------------------------------------------------------------
class AEPNStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        metadata,
        heads: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        activation: str = "relu",
        max_in_features: int = 32,
        num_bases: int = 8,
    ):
        super().__init__()
        if metadata is None:
            raise ValueError("AEPNStack needs the graph metadata (node and edge types)")
        self.num_layers = int(num_layers)
        self.hidden_channels = H = int(hidden_channels)
        self.heads = int(heads)
        if H % self.heads != 0:
            raise ValueError(f"hidden_channels={H} not divisible by heads={heads}")
        self.residual = bool(residual)
        self.max_in_features = int(max_in_features)
        self.node_types = list(metadata[0])
        self.edge_types = [tuple(e) for e in metadata[1]]
        self._ntype_idx = {nt: i for i, nt in enumerate(self.node_types)}
        self._etype_idx = {et: i for i, et in enumerate(self.edge_types)}
        n_nt, n_et = len(self.node_types), max(len(self.edge_types), 1)
        B = self.num_bases = int(num_bases)

        self.in_weight = nn.Parameter(torch.empty(n_nt, self.max_in_features, H))
        self.in_bias = nn.Parameter(torch.zeros(n_nt, H))
        nn.init.uniform_(self.in_weight, -1.0 / self.max_in_features ** 0.5, 1.0 / self.max_in_features ** 0.5)

        def bases():
            w = torch.empty(H, B * H)
            for b in range(B):
                nn.init.xavier_uniform_(w[:, b * H:(b + 1) * H])
            return nn.Parameter(w)

        def coeffs(n):
            return nn.Parameter(torch.randn(n, B) / B ** 0.5)

        L = self.num_layers
        self.k_bases = nn.ParameterList([bases() for _ in range(L)])
        self.v_bases = nn.ParameterList([bases() for _ in range(L)])
        self.q_bases = nn.ParameterList([bases() for _ in range(L)])
        self.o_bases = nn.ParameterList([bases() for _ in range(L)])
        self.k_coef = nn.ParameterList([coeffs(n_et) for _ in range(L)])
        self.v_coef = nn.ParameterList([coeffs(n_et) for _ in range(L)])
        self.q_coef = nn.ParameterList([coeffs(n_nt) for _ in range(L)])
        self.o_coef = nn.ParameterList([coeffs(n_nt) for _ in range(L)])
        self.p_rel = nn.ParameterList([nn.Parameter(torch.ones(n_et, self.heads)) for _ in range(L)])
        self.skip = nn.ParameterList([nn.Parameter(torch.ones(n_nt)) for _ in range(L)])
        if self.residual:
            self.r_bases = nn.ParameterList([bases() for _ in range(L)])
            self.r_coef = nn.ParameterList([coeffs(n_nt) for _ in range(L)])

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

    def _mix(self, h, bases, coef, idx):
        """Per-item transform sum_b coef[idx, b] * (h @ V_b), for h [M, H]."""
        P = (h @ bases).view(h.size(0), self.num_bases, self.hidden_channels)
        return torch.einsum('mb,mbh->mh', coef[idx], P)

    def _layer(self, li, h, t, src, dst, e_t):
        N, H, nh = h.size(0), self.hidden_channels, self.heads
        d = H // nh
        if src.numel() > 0:
            # Project nodes through the bases once, then mix per edge.
            Pk = (h @ self.k_bases[li]).view(N, self.num_bases, H)
            Pv = (h @ self.v_bases[li]).view(N, self.num_bases, H)
            k = torch.einsum('eb,ebh->eh', self.k_coef[li][e_t], Pk[src])
            v = torch.einsum('eb,ebh->eh', self.v_coef[li][e_t], Pv[src])
            q = self._mix(h, self.q_bases[li], self.q_coef[li], t)[dst]
            score = (q.view(-1, nh, d) * k.view(-1, nh, d)).sum(-1) * self.p_rel[li][e_t] / d ** 0.5
            alpha = pyg_softmax(score, dst, num_nodes=N)                    # [E, heads]
            msg = (v.view(-1, nh, d) * alpha.unsqueeze(-1)).view(-1, H)
            agg = h.new_zeros(N, H).index_add_(0, dst, msg)
        else:
            agg = h.new_zeros(N, H)
        out = self._mix(F.gelu(agg), self.o_bases[li], self.o_coef[li], t)
        if li > 0:
            g = torch.sigmoid(self.skip[li][t]).unsqueeze(-1)
            out = g * out + (1 - g) * h
        if self.residual:
            out = out + self._mix(h, self.r_bases[li], self.r_coef[li], t)
        return self.drop(self.act(out))

    def encode_flat(self, x, t, edge_index, e_t):
        """Flat graph -> node embeddings [N, H]."""
        h = _project_in(x, t, self.in_weight, self.in_bias)
        src, dst = edge_index[0], edge_index[1]
        for li in range(self.num_layers):
            h = self._layer(li, h, t, src, dst, e_t)
        return h

    def forward(self, x_dict, edge_index_dict, input_size=None, graph=None, params_iter=None):
        return _encode_hetero(self, x_dict, edge_index_dict)



def _make_encoder(encoder, **kw):
    if encoder in (None, "hgt"):
        return HGTStack(**kw)
    if encoder in ("type_embed", "temb"):
        return TypeEmbedStack(**kw)
    if encoder in ("type_embed_film", "tembf"):
        return TypeEmbedStack(film=True, **kw)
    if encoder == "aepn":
        return AEPNStack(**kw)
    raise ValueError(f"unknown encoder {encoder!r} (choose 'hgt', 'type_embed', 'type_embed_film' or 'aepn')")


# ---------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------
class ActorCritic(nn.Module):
    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)
    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))


# ---------------------------------------------------------------------
# Deeper Actor
# ---------------------------------------------------------------------
class HeteroActor(ActorCritic):
    """
    Deeper actor:
      • HGTConv × L with residuals & dropout
      • Lazy decoder (robust to head/concat dims)
      • Per-graph softmax over [a_transition, postpone]
    """
    def __init__(
        self,
        input_size: int = -1,
        hidden_size: int = 128,
        num_layers: int = 3,
        metadata=None,
        num_heads: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        # Opt-in, default off (byte-identical logits when False -- same
        # safe-floor convention as every other knob in this project).
        # HeteroActor.forward decodes each action's logit from ONLY that
        # node's own final HGTConv embedding -- unlike HeteroCritic, which
        # has always max-pooled over all action/postpone nodes for its
        # value estimate. Combined with get_graph_observation's edges being
        # directed strictly along token flow (add_reverse_edges=False
        # everywhere in real use), an action's logit can depend on state
        # ONLY if a forward, token-flow-direction path reaches it within
        # `num_layers` hops -- provably zero otherwise (verified directly:
        # on a fully disjoint two-chain net, perturbing one chain's
        # downstream marking left the other chain's raw logit EXACTLY
        # bit-for-bit unchanged). When True, concatenates the SAME pooled
        # context HeteroCritic already builds (_pool_action_postpone) onto
        # each action/postpone node before decoding, giving the actor
        # access to state outside its own directed-reachable neighborhood.
        global_context: bool = False,
        # 'aepn' (default; AEPNStack), 'hgt' (HGTStack, the default until
        # 2026-10-05), 'type_embed' or 'type_embed_film' (TypeEmbedStack).
        # encoder_kwargs go to the encoder, e.g. {'num_bases': 4} for AEPNStack.
        encoder: str = "aepn",
        encoder_kwargs: Optional[dict] = None,
        # Backwards compatibility: accept legacy kwargs (e.g. output_size)
        output_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.metadata = metadata
        self.num_heads = num_heads
        self.dropout = dropout
        self.global_context = bool(global_context)

        self.encoder = _make_encoder(
            encoder,
            **(encoder_kwargs or {}),
            in_channels=self.input_size,
            hidden_channels=self.hidden_size,
            num_layers=num_layers,
            metadata=self.metadata,
            heads=self.num_heads,
            dropout=self.dropout,
            residual=residual,
            activation="relu",
        )

        # Shared decoder for both 'a_transition' and 'postpone'. LazyLinear
        # absorbs the wider [own_embedding ; pooled_context] input when
        # global_context=True with zero manual dimension bookkeeping (same
        # pattern already relied on for use_structural_features).
        self.decoder = nn.Sequential(
            nn.LazyLinear(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self._warned_index_mismatches = set()

    def _build_batch_index(self, graph, x_dict, logits):
        if is_flat(graph):
            return torch.cat([_batch_index_for(graph, nt, 0, logits.device)
                              for nt in ('a_transition', 'postpone')], dim=0)
        idx_parts = []
        try:
            # a_transition
            if 'a_transition' in graph.x_dict and hasattr(graph['a_transition'], 'batch'):
                idx_parts.append(graph['a_transition'].batch)
            else:
                nA = x_dict.get('a_transition', torch.empty(0)).size(0)
                idx_parts.append(torch.zeros(nA, dtype=torch.int64, device=logits.device))
            # postpone (optional)
            if 'postpone' in graph.x_dict and graph.x_dict.get('postpone') is not None:
                if hasattr(graph['postpone'], 'batch'):
                    idx_parts.append(graph['postpone'].batch)
                else:
                    nP = x_dict.get('postpone', torch.empty(0)).size(0)
                    idx_parts.append(torch.zeros(nP, dtype=torch.int64, device=logits.device))
            if len(idx_parts) > 0:
                return torch.cat(idx_parts, dim=0)
        except Exception:
            pass
        return None

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        graph = data['graph'] if isinstance(data, dict) and 'graph' in data else data

        # Encode (handles empty node types)
        x_enc = _encode_graph(self.encoder, graph, self.input_size, iter(self.parameters()))

        # Decode in the SAME order as actions_dict: [a_transition, postpone]
        device = next(self.parameters()).device
        logits_a = torch.empty((0, 1), device=device, dtype=torch.float32)
        logits_p = None

        context = _pool_action_postpone(x_enc, graph) if getattr(self, 'global_context', False) else None

        def _decode(ntype):
            x = x_enc.get(ntype)
            if x is None or x.numel() == 0:
                return None
            if context is not None:
                idx = _batch_index_for(graph, ntype, x.size(0), x.device)
                x = torch.cat((x, context[idx]), dim=-1)
            return self.decoder(x)

        la = _decode('a_transition')
        if la is not None:
            logits_a = la
        logits_p = _decode('postpone')

        logits = torch.cat((logits_a, logits_p), dim=0) if logits_p is not None else logits_a

        # Per-graph softmax
        index = self._build_batch_index(graph, x_enc, logits)
        if index is None:
            return torch.softmax(logits, dim=0)

        if index.numel() != logits.size(0):
            key = (int(index.numel()), int(logits.size(0)))
            if key not in self._warned_index_mismatches:
                warnings.warn(f"[Actor] Softmax index length {index.numel()} != nodes {logits.size(0)}. Truncating/padding.")
                self._warned_index_mismatches.add(key)
            if index.numel() > logits.size(0):
                index = index[:logits.size(0)]
            else:
                pad_val = int(index[-1].item() if index.numel() > 0 else 0)
                pad = index.new_full((logits.size(0) - index.numel(),), pad_val)
                index = torch.cat((index, pad), dim=0)

        return pyg_softmax(logits, index)


# ---------------------------------------------------------------------
# Per-action-node off-lineage Q head (LRQ-v3)
# ---------------------------------------------------------------------
class HeteroQOff(ActorCritic):
    """Per-action-node value head for the LRQ-v3 decomposition.

    Emits one RAW scalar per action node (a_transition nodes then postpone),
    in the same order as the actor's policy vector / actions_dict, with no
    softmax: q_off(s, a) estimates the OFF-LINEAGE component of Q(s, a) —
    E[discounted future rewards NOT caused by a | s, a] — and is regressed on
    trace-computed samples (mc_q credit minus lrq2 credit of the taken
    action). The diffuse part of Q is learned; the sharp lineage part stays
    Monte-Carlo. See CAUSAL_LRQ_PROPOSAL (v3) / PAPER_PLAN_LRQ.md.
    """

    def __init__(
        self,
        input_size: int = -1,
        hidden_size: int = 128,
        num_layers: int = 3,
        metadata=None,
        num_heads: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        encoder: str = "aepn",
        encoder_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = _make_encoder(
            encoder,
            **(encoder_kwargs or {}),
            in_channels=-1,
            hidden_channels=self.hidden_size,
            num_layers=num_layers,
            metadata=metadata,
            heads=num_heads,
            dropout=dropout,
            residual=residual,
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, data) -> torch.Tensor:
        graph = data['graph'] if isinstance(data, dict) and 'graph' in data else data
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict
        x_enc = self.encoder(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            input_size=self.input_size,
            graph=graph,
            params_iter=iter(self.parameters()),
        )
        for k, v in x_enc.items():
            if v is not None:
                x_enc[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        device = next(self.parameters()).device
        vals_a = torch.empty((0, 1), device=device, dtype=torch.float32)
        vals_p = None
        if 'a_transition' in x_enc and x_enc['a_transition'] is not None and x_enc['a_transition'].numel() > 0:
            vals_a = self.decoder(x_enc['a_transition'])
        if 'postpone' in x_enc and x_enc['postpone'] is not None and x_enc['postpone'].numel() > 0:
            vals_p = self.decoder(x_enc['postpone'])
        out = torch.cat((vals_a, vals_p), dim=0) if vals_p is not None else vals_a
        return out.squeeze(-1)


# ---------------------------------------------------------------------
# Deeper Critic
# ---------------------------------------------------------------------
class HeteroCritic(ActorCritic):
    """
    Deeper critic:
      • HGTConv × L with residuals & dropout
      • Pools over [a_transition, postpone] when present
      • Lazy value head
    """
    def __init__(
        self,
        input_size: int = -1,
        hidden_size: int = 128,
        num_layers: int = 3,
        metadata=None,
        num_heads: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        # LVA (lineage value auxiliary): add a second scalar head on the SAME
        # pooled encoding, regressed on the per-decision lineage credit
        # (an auxiliary representation task; see PAPER_PLAN_LCV.md). The main
        # value_head keeps its own unbiased GAE-return target, so the aux
        # gradients shape the shared encoder but never the value output's
        # target. False (default) = single-head critic, unchanged.
        aux_head: bool = False,
        encoder: str = "aepn",
        encoder_kwargs: Optional[dict] = None,
        # Backwards compatibility: accept legacy kwargs (e.g. output_size)
        output_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.metadata = metadata
        self.num_heads = num_heads
        self.dropout = dropout

        self.encoder = _make_encoder(
            encoder,
            **(encoder_kwargs or {}),
            in_channels=self.input_size,
            hidden_channels=self.hidden_size,
            num_layers=num_layers,
            metadata=self.metadata,
            heads=self.num_heads,
            dropout=self.dropout,
            residual=residual,
            activation="relu",
        )

        self.value_head = nn.Sequential(
            nn.LazyLinear(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.lineage_aux_head = None
        if aux_head:
            self.lineage_aux_head = nn.Sequential(
                nn.LazyLinear(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, 1),
            )

    def _encode_pool(self, data):
        """Shared encode+pool: one vector per graph in the batch."""
        graph = data['graph'] if isinstance(data, dict) and 'graph' in data else data

        # Encode
        x_enc = _encode_graph(self.encoder, graph, self.input_size, iter(self.parameters()))

        # nfgae: V_c(s) pools only over the deciding component's action nodes
        # (graph['a_transition'].pool_mask, set at rollout). With a single
        # component the mask is all-true and this is the pooling below.
        mask = _pool_mask(graph, 'a_transition')
        if mask is not None and x_enc.get('a_transition') is not None:
            parts, idx_parts = [], []
            for ntype in ('a_transition', 'postpone'):
                x = x_enc.get(ntype)
                if x is None or x.numel() == 0:
                    continue
                m = _pool_mask(graph, ntype)
                if m is None:          # postpone without a mask: not this component's
                    continue
                m = m.to(device=x.device, dtype=torch.bool)
                i_ = _batch_index_for(graph, ntype, x.size(0), x.device)
                parts.append(x[m]); idx_parts.append(i_[m])
            idx_all = _batch_index_for(graph, 'a_transition', x_enc['a_transition'].size(0),
                                       x_enc['a_transition'].device)
            n_graphs = int(idx_all.max().item()) + 1 if idx_all.numel() else 1
            return global_max_pool(torch.cat(parts, 0), torch.cat(idx_parts, 0), size=n_graphs)

        # Pool across targets (shared with HeteroActor's optional
        # global_context -- see _pool_action_postpone).
        return _pool_action_postpone(x_enc, graph)

    def forward(self, data):
        return self.value_head(self._encode_pool(data))

    def forward_with_aux(self, data):
        """LVA: (value, lineage-aux) predictions off the shared encoding.
        Requires aux_head=True at construction."""
        if self.lineage_aux_head is None:
            raise RuntimeError("forward_with_aux requires HeteroCritic(aux_head=True)")
        pooled = self._encode_pool(data)
        return self.value_head(pooled), self.lineage_aux_head(pooled)