import numpy as np
import warnings
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import HANConv
from torch_geometric.utils import softmax as pyg_softmax
from torch.nn.functional import softmax
from torch_geometric.nn import global_max_pool


class ActorCritic(torch.nn.Module):
    def save_weights(self, filename):
        torch.save(self.state_dict(), filename)

    def load_weights(self, filename):
        self.load_state_dict(torch.load(filename))

class HeteroActor(ActorCritic):
    def __init__(self, input_size=-1, hidden_size=128, output_size=64, metadata=None, num_heads=1):
        super(HeteroActor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.metadata = metadata

        # Encoder: HANConv layer
        self.encoder = HANConv(self.input_size, self.output_size, heads=self.num_heads, metadata=self.metadata)

        # Decoder: MLP with a 1-dimensional output
        self.decoder = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(self, data):
        if 'graph' in data.keys():
            graph = data['graph']
            # try to obtain per-node batch indices when the graph is a batched HeteroData
            try:
                idx_a = graph['a_transition']['batch']
                if 'postpone' in graph.x_dict.keys() and graph.x_dict['postpone'] is not None:
                    idx_p = graph['postpone']['batch']
                    index = torch.cat((idx_a, idx_p), dim=0)
                else:
                    index = idx_a
            except Exception:
                # fallback to single-graph index (all zeros)
                index = torch.zeros(graph['a_transition']['x'].shape[0], dtype=torch.int64, device=next(self.parameters()).device)
        else:
            graph = data
            index = data['a_transition']['batch']
            if 'postpone' in graph.x_dict.keys()  and graph.x_dict['postpone'] is not None:
                index_postpone = data['postpone']['batch']
                index = torch.cat((index, index_postpone), dim=0)

            #print(f"Index unique values {len(set(index.tolist()))}")

        x_dict = graph.x_dict

        edge_index_dict = graph.edge_index_dict

        # Encode 'a_transition' nodes
        x_dict = self.encoder(x_dict, edge_index_dict)

        # Check if there are nan values in the encoded features
        for key, value in x_dict.items():
            if value is None:
                continue
            if torch.isnan(value).any():
                print(f"NaN values found in {key} after encoding.")
                # Handle NaN values if necessary (e.g., replace with zeros, etc.)
                x_dict[key] = torch.nan_to_num(value)


        # Decode 'a_transition' nodes one by one
        x_dict['a_transition'] = self.decoder(x_dict['a_transition'])

        # check if graph.x_dict contains 'postpone', if so it needs to be decoded as well
        if 'postpone' in x_dict.keys() and x_dict['postpone'] is not None:
            x_dict['postpone'] = self.decoder(x_dict['postpone'])
            relevant_x_dict = torch.cat((x_dict['a_transition'], x_dict['postpone']), dim=0)
        else:
            relevant_x_dict = x_dict['a_transition']

        # Apply softmax per-sample using the node index when available. For unbatched inputs
        # fall back to a global softmax.
        # Build a per-node index that matches the concatenation order used for relevant_x_dict
        # (we concatenated [a_transition, postpone] above).
        # Prefer batch vectors provided by the batched HeteroData; if absent, synthesize zeros.
        idx_parts = []
        try:
            # graph is the (possibly batched) HeteroData
            if 'a_transition' in graph.x_dict and hasattr(graph['a_transition'], 'batch'):
                idx_parts.append(graph['a_transition'].batch)
            else:
                # synthesize zeros for a_transition nodes
                nA = x_dict['a_transition'].size(0) if 'a_transition' in x_dict else 0
                idx_parts.append(torch.zeros(nA, dtype=torch.int64, device=relevant_x_dict.device))

            if 'postpone' in graph.x_dict and graph.x_dict.get('postpone') is not None:
                if hasattr(graph['postpone'], 'batch'):
                    idx_parts.append(graph['postpone'].batch)
                else:
                    nP = x_dict['postpone'].size(0) if 'postpone' in x_dict and x_dict['postpone'] is not None else 0
                    idx_parts.append(torch.zeros(nP, dtype=torch.int64, device=relevant_x_dict.device))

            if len(idx_parts) > 0:
                index = torch.cat(idx_parts, dim=0)
            else:
                index = None
        except Exception:
            # Fall back to None so we use a global softmax below and surface the mismatch upstream
            index = None

        if index is None:
            # No per-node index available: apply global softmax over the concatenated logits
            return softmax(relevant_x_dict, dim=0)

        # Sanity-check: index length should equal number of concatenated nodes.
        total_nodes = relevant_x_dict.size(0)
        if index.numel() != total_nodes:
            # Adjust by truncation or padding with last index (safer than crashing)
            key = (int(index.numel()), int(total_nodes))
            # Warn once per unique pattern to avoid flooding logs
            if not hasattr(self, '_warned_index_mismatches'):
                self._warned_index_mismatches = set()
            if key not in self._warned_index_mismatches:
                warnings.warn(f"Softmax index length {index.numel()} != nodes {total_nodes}. Truncating/padding to match.")
                self._warned_index_mismatches.add(key)

            if index.numel() > total_nodes:
                index = index[:total_nodes]
            else:
                pad_val = int(index[-1].item() if index.numel() > 0 else 0)
                pad = index.new_full((total_nodes - index.numel(),), pad_val)
                index = torch.cat((index, pad), dim=0)

        # Use torch_geometric softmax with the per-node batch index
        return pyg_softmax(relevant_x_dict, index)




class HeteroCritic(ActorCritic):
    def __init__(self, input_size=-1, hidden_size=128, output_size=64, metadata=None, num_heads=1):
        super(HeteroCritic, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.metadata = metadata

        # Encoder layers
        self.conv1 = HANConv(self.input_size, self.hidden_size, heads=self.num_heads, metadata=self.metadata)

        # Linear layer for final output
        self.lin = nn.Linear(self.hidden_size * self.num_heads, 1)

    def forward(self, data):
        if 'graph' in data.keys():
            x, metadata = data['graph'], data['graph'].metadata()
            # Use per-node batch indices for pooling when available
            try:
                index = x['a_transition']['batch']
            except Exception:
                index = torch.zeros(x['a_transition']['x'].shape[0], dtype=torch.int64, device=next(self.parameters()).device)
        else:
            x = data
            index = data['a_transition']['batch']

        x_dict = x.x_dict
        edge_index_dict = x.edge_index_dict

        # First convolution
        x_dict = self.conv1(x_dict, edge_index_dict)

        # If postpone nodes exist, include them in the pooled representation by
        # concatenating a_transition and postpone features and building a matching
        # index vector for pooling (same ordering as in the actor: [a_transition, postpone]).
        if 'postpone' in x_dict and x_dict['postpone'] is not None:
            concat_nodes = torch.cat((x_dict['a_transition'], x_dict['postpone']), dim=0)
            # build index matching the concatenation
            try:
                idx_parts = []
                if hasattr(x['a_transition'], 'batch'):
                    idx_parts.append(x['a_transition'].batch)
                else:
                    idx_parts.append(torch.zeros(x_dict['a_transition'].size(0), dtype=torch.int64, device=concat_nodes.device))
                if hasattr(x['postpone'], 'batch'):
                    idx_parts.append(x['postpone'].batch)
                else:
                    idx_parts.append(torch.zeros(x_dict['postpone'].size(0), dtype=torch.int64, device=concat_nodes.device))
                pool_index = torch.cat(idx_parts, dim=0)
            except Exception:
                pool_index = torch.zeros(concat_nodes.size(0), dtype=torch.int64, device=concat_nodes.device)

            x = global_max_pool(concat_nodes, pool_index)
        else:
            x = global_max_pool(x_dict['a_transition'], index)

        # Final linear layer
        x = self.lin(x)
        return x




class HeteroActorCriticEncoder(ActorCritic):
    """
    Actor-Critic encoder model for heterogeneous graphs using HANConv layer(s).
    """
    def __init__(self, input_size=-1, hidden_size=64, output_size=16, metadata=None, num_heads=1):
        super(ActorCritic, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.metadata = metadata

        # Encoder layers
        self.conv1 = HANConv(self.input_size, self.hidden_size, heads=self.num_heads, metadata=self.metadata)


    def forward(self, data):
        if 'graph' in data.keys():
            x, metadata = data['graph'], data['graph'].metadata()
        else:
            x = data
            index = data['a_transition']['batch']

        x_dict = x.x_dict
        edge_index_dict = x.edge_index_dict

        # First convolution
        x_dict = self.conv1(x_dict, edge_index_dict)

        return x_dict


class HeteroActorDecoder(ActorCritic):
    """
    Actor-Critic decoder model for heterogeneous graphs using HANConv layer(s).
    """
    def __init__(self, encoder, input_size=-1, hidden_size=64, output_size=16, metadata=None, num_heads=1):
        super(HeteroActorDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.metadata = metadata

        # Store the encoder
        self.encoder = encoder

        # Decoder layers
        self.lin1 = nn.Linear(self.input_size, self.hidden_size)
        #self.lin2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, data):
        if 'graph' in data.keys():
            x, metadata = data['graph'], data['graph'].metadata()
        else:
            x = data
            index = data['a_transition']['batch']

        x_dict = self.encoder(data)

        # First convolution
        x_dict = self.lin1(x_dict['a_transition'])

        return x_dict


class HeteroCriticDecoder(ActorCritic):
    def __init__(self, encoder, input_size=-1, hidden_size=64, output_size=16, metadata=None, num_heads=1):
        super(ActorCritic, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.metadata = metadata

        # Store the encoder
        self.encoder = encoder

        # Decoder layers
        self.lin1 = nn.Linear(self.input_size, self.hidden_size)
        # self.lin2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, data):
        if 'graph' in data.keys():
            x, metadata = data['graph'], data['graph'].metadata()
        else:
            x = data
            index = data['a_transition']['batch']

        x_dict = x.x_dict
        edge_index_dict = x.edge_index_dict

        # Aggregation
        if 'graph' in data.keys():
            x_dict = {k: v.sum(dim=0, keepdim=True) for k, v in x_dict.items() if v is not None}
            x = sum(x_dict.values())  # simple aggregation
        else:
            x_dict = scatter(x_dict['a_transition'], index, dim=0, reduce='sum')
            x = x_dict

        # Final linear layer
        x = self.lin(x)
        return x