import numpy as np
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
    def __init__(self, input_size=-1, hidden_size=64, output_size=16, metadata=None, num_heads=1):
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
        else:
            graph = data
            index = data['a_transition']['batch']
            print(f"Index unique values {len(set(index.tolist()))}")

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

        if 'graph' in data.keys():
            x_dict = softmax(x_dict['a_transition'], dim=0)
        else:
            x_dict = pyg_softmax(x_dict['a_transition'], index)


        return x_dict




class HeteroCritic(ActorCritic):
    def __init__(self, input_size=-1, hidden_size=64, output_size=16, metadata=None, num_heads=1):
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
            index = torch.zeros(data['graph']['a_transition']['x'].shape[0]).to(torch.int64)
        else:
            x = data
            index = data['a_transition']['batch']

        x_dict = x.x_dict
        edge_index_dict = x.edge_index_dict

        # First convolution
        x_dict = self.conv1(x_dict, edge_index_dict)

        # Aggregation
        #if 'graph' in data.keys():
        #    x_dict = {k: v.max(dim=0, keepdim=True) for k, v in x_dict.items() if v is not None}
        #    x = sum(x_dict.values())  # simple aggregation
        #else:
        #    x_dict = scatter(x_dict['a_transition'], index, dim=0, reduce='max')
        #    x = x_dict

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