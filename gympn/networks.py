import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax


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
            # Use the per-type dim if known, otherwise fall back to the global dim
            dim = (seen_dims or {}).get(ntype, input_dim)
            x_fixed[ntype] = torch.zeros((1, dim), device=device, dtype=dtype)
            dummies.add(ntype)
        else:
            x_fixed[ntype] = x
    return x_fixed, dummies


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

        self.encoder = HGTStack(
            in_channels=self.input_size,
            hidden_channels=self.hidden_size,
            num_layers=num_layers,
            metadata=self.metadata,
            heads=self.num_heads,
            dropout=self.dropout,
            residual=residual,
            activation="relu",
        )

        # Shared decoder for both 'a_transition' and 'postpone'
        self.decoder = nn.Sequential(
            nn.LazyLinear(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self._warned_index_mismatches = set()

    def _build_batch_index(self, graph, x_dict, logits):
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
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict

        # Encode (handles empty node types)
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

        # Decode in the SAME order as actions_dict: [a_transition, postpone]
        device = next(self.parameters()).device
        logits_a = torch.empty((0, 1), device=device, dtype=torch.float32)
        logits_p = None

        if 'a_transition' in x_enc and x_enc['a_transition'] is not None and x_enc['a_transition'].numel() > 0:
            logits_a = self.decoder(x_enc['a_transition'])
        if 'postpone' in x_enc and x_enc['postpone'] is not None and x_enc['postpone'].numel() > 0:
            logits_p = self.decoder(x_enc['postpone'])

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

        self.encoder = HGTStack(
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

    def forward(self, data):
        graph = data['graph'] if isinstance(data, dict) and 'graph' in data else data
        x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict

        # Encode
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

        # Pool across targets
        if 'postpone' in x_enc and x_enc['postpone'] is not None and x_enc['postpone'].numel() > 0:
            concat_nodes = torch.cat((x_enc['a_transition'], x_enc['postpone']), dim=0)
            try:
                idx_parts = []
                if hasattr(graph['a_transition'], 'batch'):
                    idx_parts.append(graph['a_transition'].batch)
                else:
                    idx_parts.append(torch.zeros(x_enc['a_transition'].size(0), dtype=torch.int64, device=concat_nodes.device))
                if hasattr(graph['postpone'], 'batch'):
                    idx_parts.append(graph['postpone'].batch)
                else:
                    idx_parts.append(torch.zeros(x_enc['postpone'].size(0), dtype=torch.int64, device=concat_nodes.device))
                pool_index = torch.cat(idx_parts, dim=0)
            except Exception:
                pool_index = torch.zeros(concat_nodes.size(0), dtype=torch.int64, device=concat_nodes.device)
            pooled = global_max_pool(concat_nodes, pool_index)
        else:
            try:
                index = graph['a_transition'].batch
            except Exception:
                index = torch.zeros(x_enc['a_transition'].size(0), dtype=torch.int64, device=x_enc['a_transition'].device)
            pooled = global_max_pool(x_enc['a_transition'], index)

        return self.value_head(pooled)