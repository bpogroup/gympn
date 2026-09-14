"""A Petri-net-native graph operator: conjunctive consumption, additive production.

WHY THIS EXISTS
---------------
The observation graph of an A-E PN is strictly bipartite (places <-> transitions;
measured: 17 place->transition edge types, 17 transition->place, ZERO
place->place). The two directions carry opposite semantics:

  place -> transition   CONSUMPTION / ENABLING.  A transition fires only when
                        EVERY input place is marked, and how many times it can
                        fire is  min_p (tokens in p).  A CONJUNCTION.

  transition -> place   PRODUCTION.  A firing deposits tokens into all of its
                        output places, and tokens accumulate.  ADDITIVE.

`HGTConv` -- the operator gympn uses today -- applies attention-weighted SUM in
both directions. Sum is right for production and wrong for consumption: it
cannot separate

    3 tokens in one input place, 0 elsewhere   -> capacity 0 (cannot fire)
    1 token in each of 3 input places          -> capacity 1 (can fire)

which are opposite in PN semantics and near-identical to a sum.

MEASURED, not assumed. Linear probes for per-transition firing capacity
(= min over input places), held-out R^2 with a shuffled-feature placebo:

    env         MIN(ceiling)   SUM(naive)   HGT encoder   placebo
    s1                1.000        0.192        0.011      -0.004   (TRAINED)
    s1                1.000        0.185        0.009      -0.004   (untrained)
    multisite         1.000        0.726        0.065      -0.003   (untrained)

The HGT embeddings sit at placebo level while a single scalar (the sum) reaches
0.19-0.73. Training does not fix it: a fully trained actor scores 0.0105 versus
0.0093 random. The conjunctive quantity that defines Petri-net firing is simply
absent from the representation the policy decodes from.

DESIGN
------
`PNConv` treats the two edge directions differently:

  * consumption edges use SOFT-MIN aggregation, implemented as a segment
    softmax over NEGATED scores:  w = softmax(-s / tau),  out = sum_p w_p * v_p.
    As tau -> 0 this selects the minimum exactly; as tau -> inf it becomes the
    mean. tau is LEARNABLE per edge type, so the operator CONTAINS mean
    aggregation as a limiting case and can fall back to it when the data
    prefers -- it cannot be strictly less expressive than averaging.

  * production edges use ordinary sum aggregation.

Nothing here reads token attribute values or assumes task-assignment semantics;
the split is a property of Petri-net arc direction and therefore general across
A-E PN shapes (resource pools, rework loops, joins, exclusive choice).

STATUS: research code for the custom-GNN line (see README.md). Not yet wired
into gympn.networks -- validated at the representational level first.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as segment_softmax

EdgeType = Tuple[str, str, str]

TRANSITION_TYPES = ("a_transition", "e_transition")


def is_consumption(edge_type: EdgeType) -> bool:
    """place -> transition: the conjunctive direction."""
    src, _, dst = edge_type
    return dst in TRANSITION_TYPES and src not in TRANSITION_TYPES


def is_production(edge_type: EdgeType) -> bool:
    """transition -> place: the additive direction."""
    src, _, dst = edge_type
    return src in TRANSITION_TYPES and dst not in TRANSITION_TYPES


class SoftMinAggregation(nn.Module):
    """Differentiable min over each target's incoming edges.

    w = softmax(-score / tau) within a target group;  out = sum w_i * value_i.

    tau is stored as log_tau so it stays positive under unconstrained gradient
    descent. tau -> 0 recovers a hard min; tau -> inf recovers the mean, so the
    module spans conjunctive and additive behaviour and can learn where to sit
    rather than having it imposed.
    """

    def __init__(self, init_tau: float = 1.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(init_tau)))))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-3, max=1e3)

    def forward(self, value: torch.Tensor, score: torch.Tensor,
                index: torch.Tensor, num_targets: int) -> torch.Tensor:
        """value: [E, C] messages. score: [E] scalar per edge to take the min OF.
        index: [E] target node id. Returns [num_targets, C]."""
        if value.numel() == 0:
            return value.new_zeros((num_targets, value.size(-1)))
        w = segment_softmax(-score / self.tau, index, num_nodes=num_targets)
        out = value.new_zeros((num_targets, value.size(-1)))
        out.index_add_(0, index, value * w.unsqueeze(-1))
        return out


class PNConv(nn.Module):
    """One Petri-net message-passing round.

    Consumption edges are aggregated conjunctively (soft-min), production edges
    additively (sum). Each edge type gets its own linear map and its own
    temperature, because different arcs carry different resources.

    Parameters
    ----------
    metadata : (node_types, edge_types) in PyG's HeteroData convention.
    in_channels, out_channels : feature widths (LazyLinear handles per-type
        input widths, matching how gympn builds its heterogeneous inputs).
    """

    def __init__(self, metadata, out_channels: int, init_tau: float = 1.0):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = [tuple(e) for e in edge_types]
        self.out_channels = out_channels

        self.msg = nn.ModuleDict()
        self.score = nn.ModuleDict()
        self.aggr = nn.ModuleDict()
        for et in self.edge_types:
            key = "__".join(et)
            self.msg[key] = nn.LazyLinear(out_channels)
            if is_consumption(et):
                # scalar per edge: WHAT the min is taken over
                self.score[key] = nn.LazyLinear(1)
                self.aggr[key] = SoftMinAggregation(init_tau)
        self.self_lin = nn.ModuleDict(
            {nt: nn.LazyLinear(out_channels) for nt in self.node_types})
        self.out_lin = nn.ModuleDict(
            {nt: nn.Linear(out_channels, out_channels) for nt in self.node_types})

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """Two-stage consumption aggregation.

        The conjunction is over INPUT PLACES, and a place's token count is the
        NUMBER OF EDGES it contributes (gympn expands places so each token is
        its own node). So the arithmetic has to be:

            within one consumption edge type : SUM   -- counts that place's tokens
            across consumption edge types    : MIN   -- the conjunction

        A first version soft-minned over individual edges and summed across edge
        types, i.e. exactly backwards, and probed identically to HGTConv
        (-27.1% vs -26.8% of the gap to ceiling) -- which is what caught it.

        Targets with no edge of a given type are EXCLUDED from that type's min
        rather than contributing zero; otherwise an absent input place would
        look like the scarcest one and dominate every minimum.
        """
        out = {}
        for nt in self.node_types:
            x = x_dict.get(nt)
            if x is None or x.numel() == 0:
                continue
            out[nt] = self.self_lin[nt](x)

        # dst -> list of (per-type aggregate [n_dst, C], present mask [n_dst])
        cons: Dict[str, list] = {}

        for et, ei in edge_index_dict.items():
            et = tuple(et)
            src, _, dst = et
            key = "__".join(et)
            if key not in self.msg:
                continue
            xs, xd = x_dict.get(src), x_dict.get(dst)
            if xs is None or xd is None or xs.numel() == 0 or xd.numel() == 0:
                continue
            if ei.numel() == 0:
                continue
            s_idx, d_idx = ei[0], ei[1]
            if int(s_idx.max()) >= xs.size(0) or int(d_idx.max()) >= xd.size(0):
                continue
            m = self.msg[key](xs[s_idx])
            n_dst = xd.size(0)
            agg = m.new_zeros((n_dst, m.size(-1)))
            agg.index_add_(0, d_idx, m)                 # SUM within the type
            if is_consumption(et):
                present = m.new_zeros(n_dst)
                present.index_add_(0, d_idx, torch.ones_like(d_idx, dtype=m.dtype))
                cons.setdefault(dst, []).append((agg, present > 0, key))
            else:
                out[dst] = out.get(dst, 0) + agg        # production stays additive

        for dst, parts in cons.items():
            if len(parts) == 1:
                agg = parts[0][0]
            else:
                stack = torch.stack([p[0] for p in parts], dim=0)      # [K,N,C]
                mask = torch.stack([p[1] for p in parts], dim=0)       # [K,N]
                # scalar score per (type, target): what the MIN is taken over
                scores = torch.stack(
                    [self.score[p[2]](p[0]).reshape(-1) for p in parts], dim=0)
                tau = torch.stack([self.aggr[p[2]].tau for p in parts]).mean()
                scores = scores.masked_fill(~mask, float('inf'))
                w = torch.softmax(-scores / tau, dim=0)                # [K,N]
                w = torch.nan_to_num(w, nan=0.0)
                agg = (stack * w.unsqueeze(-1)).sum(dim=0)             # MIN across types
            out[dst] = out.get(dst, 0) + agg

        return {nt: self.out_lin[nt](torch.relu(v)) for nt, v in out.items()
                if isinstance(v, torch.Tensor)}


class PNEncoder(nn.Module):
    """`num_layers` rounds of PNConv with residual connections.

    NOTE on depth: because the graph is bipartite, one TRANSITION FIRING costs
    two hops, so L layers span L/2 firings. Measured realized decision->reward
    lineage depth is 5-9 firings (median), i.e. 10-18 hops -- far beyond the
    default num_layers=3. Naively deepening HGT to 8 made s1 WORSE (12.60 vs
    13.70, and 55% slower), consistent with oversmoothing, which is part of why
    a better per-layer operator is the more promising lever than more layers.
    """

    def __init__(self, metadata, hidden: int = 32, num_layers: int = 3,
                 init_tau: float = 1.0):
        super().__init__()
        self.convs = nn.ModuleList(
            [PNConv(metadata, hidden, init_tau) for _ in range(num_layers)])

    def forward(self, x_dict, edge_index_dict):
        h = dict(x_dict)
        for i, conv in enumerate(self.convs):
            new = conv(h, edge_index_dict)
            if i == 0:
                h = new
            else:
                h = {k: (new[k] + h[k] if k in h and h[k].shape == new[k].shape
                         else new[k]) for k in new}
        return h
