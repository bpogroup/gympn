"""Flat (homogeneous) graph observations.

A GymProblem observation is a HeteroData with one node type per place and one
edge type per arc. PyG's per-type bookkeeping (batching, cloning) and the
per-type loops of HGT scale with the number of types, which grows with the
size of the net. The TypeEmbedStack / AEPNStack encoders only ever work on the
flattened graph, so with GymProblem.flat_obs the observation is converted once,
when it is built, into a FlatGraph:

    x       [N, F]   node features, zero-padded to the widest type of the net
    ntype   [N]      node-type id (index into metadata[0])
    edge_index [2, E], etype [E]   edges and edge-type ids (metadata[1])
    a_idx   [nA]     positions of the a_transition nodes, in action order
    p_idx   [nP]     positions of the postpone nodes, in action order

Nodes keep the order of the HeteroData dict (types in insertion order, nodes in
order within a type), so the actor's output order [a_transition ; postpone] is
unchanged. Per-sample training labels (y, advantage, logpis_a, ...) are added
by TrajectoryBuffer.get() as plain attributes.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class FlatGraph(Data):
    """A Data whose action-node index vectors shift with the node offset when
    graphs are batched (like edge_index)."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ('a_idx', 'p_idx'):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def is_flat(graph) -> bool:
    """True for a FlatGraph or a Batch of FlatGraphs."""
    return isinstance(graph, FlatGraph) or (hasattr(graph, 'ntype') and hasattr(graph, 'a_idx'))


def feature_width(x_dict) -> int:
    """Widest node-feature width among the types present (empty types included,
    since an empty type still carries its [0, W] width)."""
    w = 0
    for x in x_dict.values():
        if x is not None and x.dim() == 2:
            w = max(w, x.size(-1))
    return w


def flatten_hetero(x_dict, edge_index_dict, ntype_idx: Dict[str, int], etype_idx: Dict[tuple, int],
                   width: int):
    """Heterogeneous dicts -> (xs, tidx, eis, eidx, spans): per-type padded
    feature blocks, per-type node-type ids, offset edge blocks, edge-type ids,
    and per-type (offset, count). Concatenate the lists to get the flat graph."""
    xs, tidx, spans = [], [], {}
    off = 0
    for nt, x in x_dict.items():
        if nt not in ntype_idx:
            raise KeyError(f"node type {nt!r} not in the network's metadata")
        n = 0 if x is None else x.size(0)
        spans[nt] = (off, n)
        if n == 0:
            continue
        w = x.size(-1)
        if w > width:
            raise ValueError(f"node type {nt!r} has {w} features > width={width}")
        xs.append(F.pad(x, (0, width - w)) if w < width else x)
        tidx.append(torch.full((n,), ntype_idx[nt], dtype=torch.long, device=x.device))
        off += n
    eis, eidx = [], []
    for et, ei in edge_index_dict.items():
        if ei is None or ei.numel() == 0:
            continue
        et = tuple(et)
        if et not in etype_idx:
            raise KeyError(f"edge type {et!r} not in the network's metadata")
        src, _, dst = et
        if spans.get(src, (0, 0))[1] == 0 or spans.get(dst, (0, 0))[1] == 0:
            continue
        shift = ei.new_tensor([[spans[src][0]], [spans[dst][0]]])
        eis.append(ei + shift)
        eidx.append(torch.full((ei.size(1),), etype_idx[et], dtype=torch.long, device=ei.device))
    return xs, tidx, eis, eidx, spans


def cat_flat(xs, tidx, eis, eidx, width: int, device=None, dtype=torch.float32):
    """Concatenate flatten_hetero's blocks into (x, ntype, edge_index, etype)."""
    x = torch.cat(xs, 0).to(device=device, dtype=dtype) if xs else torch.zeros((0, width), dtype=dtype, device=device)
    t = torch.cat(tidx, 0) if tidx else torch.zeros((0,), dtype=torch.long, device=device)
    if eis:
        edge_index, et = torch.cat(eis, 1), torch.cat(eidx, 0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        et = torch.zeros((0,), dtype=torch.long, device=device)
    return x, t, edge_index, et


def vocab(metadata) -> Tuple[Dict[str, int], Dict[tuple, int]]:
    node_types: List[str] = list(metadata[0])
    edge_types = [tuple(e) for e in metadata[1]]
    return ({nt: i for i, nt in enumerate(node_types)},
            {et: i for i, et in enumerate(edge_types)})


def hetero_to_flat(graph, metadata, width: int = None) -> FlatGraph:
    """Convert a HeteroData observation into a FlatGraph (same node order)."""
    nidx, eidx_map = vocab(metadata)
    x_dict, edge_index_dict = graph.x_dict, graph.edge_index_dict
    w = feature_width(x_dict) if width is None else width
    xs, tidx, eis, eidx, spans = flatten_hetero(x_dict, edge_index_dict, nidx, eidx_map, w)
    x, t, edge_index, et = cat_flat(xs, tidx, eis, eidx, w)

    def positions(nt):
        o, n = spans.get(nt, (0, 0))
        return torch.arange(o, o + n, dtype=torch.long)

    return FlatGraph(x=x, ntype=t, edge_index=edge_index, etype=et,
                     a_idx=positions('a_transition'), p_idx=positions('postpone'),
                     num_nodes=int(x.size(0)))
