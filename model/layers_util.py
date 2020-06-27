import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, dense_to_sparse
from torch_sparse import coalesce

from config import FLAGS
from utils.data.representation_node_feat import encode_node_features


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the "
                "same as the lengh of hidden_channels")

        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(
            list(map(
                self.weight_init,
                [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                 for i in range(len(self.layer_channels) - 1)])
            ))
        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([torch.nn.BatchNorm1d(dim)
                                     for dim in self.layer_channels[1:-1]])

    @staticmethod
    def weight_init(m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for i, layer in enumerate(self.layers):
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                if self.bn:
                    output = self.activation(self.bn[i](layer(input)))
                else:
                    output = self.activation(layer(input))
                layer_inputs.append(output)
        # models.store_layer_output(self, layer_inputs[-1])
        return layer_inputs[-1]


def create_act(act, num_parameters=None):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity':
        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def add_negative_edges(edge_index, batch_data):
    interaction_edges = batch_data.batch_interaction_inds
    edge_index_ind_dict = batch_data.merge_higher_level['pair_inds_to_edge_index_ind']
    additional_edges = []
    ind = edge_index.shape[1]
    for i in range(0, len(interaction_edges), 2):
        edge = (interaction_edges[i], interaction_edges[i+1])
        if edge not in edge_index_ind_dict:
            edge_index_ind_dict[edge] = ind
            edge_index_ind_dict[(edge[1], edge[0])] = ind + 1
            ind += 2
            additional_edges.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
    if batch_data.is_train and FLAGS.negative_sample and hasattr(batch_data, "negative_pair_gids"):
        assert(int(len(additional_edges) / 2) == len(batch_data.negative_pair_gids))
    if additional_edges:
        new_edge_index = torch.tensor(additional_edges, device=FLAGS.device).view(2,len(additional_edges))
        return torch.cat((edge_index, new_edge_index), dim=1)
    else:
        return edge_index


def convert_nx_to_pyg_graph(g):
    """converts_a networkx graph to a PyGSingleGraphData."""
    # Reference: https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/ppi.py

    if type(g) is not nx.Graph:
        raise ValueError('Input graphs must be undirected nx.Graph,'
                         ' NOT {}'.format(type(g)))

    if type(g.init_x) is not torch.Tensor:
        g_x = torch.tensor(g.init_x, dtype=torch.float32,
                           device=FLAGS.device)
    else:
        g_x = g.init_x

    if hasattr(g, 'edge_attr_x'):
        if type(g.edge_attr_x) is not torch.Tensor:
            # assume edge attr includes self loops and is one way ie. (ni, nj) but not (nj, ni)
            edge_attr_x = torch.tensor(g.edge_attr_x, dtype=torch.float32,
                               device=FLAGS.device)
        else:
            edge_attr_x = g.edge_attr_x
        edge_index, edge_attr_x = create_edge_index(g, edge_attr_x)
        assert edge_attr_x.shape[0] == edge_index.shape[1]
    else:
        edge_index, edge_attr_x = create_edge_index(g)

    data = Data(
        x=g_x,
        edge_index=edge_index,
        edge_attr=edge_attr_x,
        y=None)

    data, nf_dim = encode_node_features(pyg_single_g=data)
    assert data.is_undirected()
    assert data.x.shape[1] == nf_dim
    return data


def convert_adj_x_to_pyg_graph(adj, x):
    edge_index, edge_weight = dense_to_sparse(adj)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=None)
    return data


def create_edge_index(g, edge_attr_x=None):
    if g.edges:
        edge_index = torch.tensor(sorted(list(g.edges)),
                                  device=FLAGS.device).t().contiguous()
    else:
        edge_index = torch.tensor([], device=FLAGS.device)

    edge_attr = None
    if edge_attr_x is not None:
        row, col = edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index_temp = torch.stack([row, col], dim=0)
        edge_attr_x = torch.cat((edge_attr_x, edge_attr_x), dim=0)
        edge_index, edge_attr = coalesce(edge_index_temp, edge_attr_x,
                                         g.number_of_nodes(),
                                         g.number_of_nodes(),
                                         op='max')
    else:
        edge_index = to_undirected(edge_index, num_nodes=g.number_of_nodes())
    return edge_index, edge_attr