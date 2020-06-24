import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GCNConv, GATConv

from config import FLAGS
from model.layers_util import MLP, create_act


class MetaLayerWrapper(nn.Module):
    def __init__(self, input_dim, edge_dim, output_dim, edge_model, node_model, act,
                 num_edge_types, higher_level=True):
        super().__init__()
        self.higher_level = higher_level
        self.activation = create_act(act)
        self.num_edge_types = num_edge_types
        if edge_model == 'mlp_concat':
            self.edge_model = EdgeModel(input_dim, edge_dim, edge_dim)
        elif edge_model == 'none':
            self.edge_model = None
        else:
            raise NotImplementedError

        if node_model == 'mlp_concat':
            self.node_model = NodeModel(input_dim, edge_dim, output_dim)
        elif node_model == 'gcn_concat':
            self.node_model = NodeModelGNN('gcn', input_dim, edge_dim, output_dim)
        elif node_model == 'gat_concat':
            self.node_model = NodeModelGNN('gat', input_dim, edge_dim, output_dim)
        elif 'multi_edge_aggr' in node_model:
            assert self.edge_model is None
            assert higher_level
            self.node_model = NodeModelAggrByEdge(node_model.split('_')[0], input_dim, num_edge_types, output_dim )
        self.meta_layer = MetaLayer(self.edge_model, self.node_model)

    def forward(self, ins, batch_data, model):
        if self.higher_level:
            merge = batch_data.merge_higher_level['merge']
        else:
            merge = batch_data.merge_data['merge']

        edge_index = merge.edge_index
        edge_attr = merge.edge_attr

        out, merge.edge_attr, _ = self.meta_layer(ins, edge_index, edge_attr, u=batch_data)
        return self.activation(out)


class EdgeModel(nn.Module):
    def __init__(self, input_dim, edge_dim, output_dim):
        super().__init__()
        self.edge_mlp = MLP(input_dim * 2 + edge_dim, output_dim, num_hidden_lyr=1)

    def forward(self, src, dest, edge_attr, *args):
        # u is global features
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModelAggrByEdge(nn.Module):
    def __init__(self, type, input_dim, num_edge_types, output_dim):
        super().__init__()
        self.type = type
        self.input_dim = input_dim
        self.output_dim = output_dim
        if type == 'gcn':
            self.GNNS = nn.ModuleList([GCNConv(input_dim, output_dim) for _ in range(num_edge_types -1)])
        elif type == 'gat':
            self.GNNS = nn.ModuleList([GATConv(input_dim, output_dim) for _ in range(num_edge_types - 1)])
        else:
            raise ValueError

    def forward(self, x, edge_index, edge_attr, u):
        batch_data = u
        outs = torch.zeros(x.shape[0], self.output_dim, device=FLAGS.device)
        for i, edge_index in enumerate(batch_data.merge_higher_level['edges'].values()):
            outs = outs + self.GNNS[i](x, edge_index)
        return outs


class NodeModel(nn.Module):
    def __init__(self, input_dim, edge_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.node_mlp_1 = MLP(input_dim + edge_dim, output_dim, num_hidden_lyr=1)
        # self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

    def forward(self, x, edge_index, edge_attr, *args):
        row, col = edge_index
        out = torch.cat((x[row], edge_attr), dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        return out


class NodeModelGNN(nn.Module):
    def __init__(self, type, input_dim, edge_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        if type == 'gcn':
            self.GNN = GCNConvEdge(input_dim, edge_dim, output_dim)
        elif type == 'gat':
            self.GNN = GATConvEdge(input_dim, edge_dim, output_dim)
        else:
            raise ValueError

    def forward(self, x, edge_index, edge_attr, *args):
        return self.GNN(x, edge_index, edge_attr)


class GCNConvEdge(GCNConv):
    def __init__(self, input_dim, edge_dim, output_dim):
        super().__init__(input_dim  + edge_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConvEdge.norm(edge_index, x.size(0), edge_weight,
                                            self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def message(self, x_j, edge_index, edge_attr=None, norm=None):
        assert edge_attr is not None and norm is not None
        x_j = torch.cat((x_j, edge_attr), dim=1)
        x_j = torch.matmul(x_j, self.weight)
        return norm.view(-1, 1) * x_j


class GATConvEdge(GATConv):
    def __init__(self, input_dim, edge_dim, output_dim):
        super().__init__(input_dim + edge_dim, output_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """"""
        # x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, edge_attr=None, num_nodes=None):
        x_i = torch.cat((x_i, edge_attr), dim=1)
        x_j = torch.cat((x_j, edge_attr), dim=1)
        x_i = torch.mm(x_i, self.weight).view(-1, self.heads, self.out_channels)
        x_j = torch.mm(x_j, self.weight).view(-1, self.heads, self.out_channels)

        return super().message(x_i, x_j, edge_index, num_nodes)
