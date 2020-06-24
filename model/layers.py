from config import FLAGS
from model.layers_util import create_act

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv


class NodeEmbedding(nn.Module):
    def __init__(self, type, in_dim, out_dim, act, bn,
                 normalize, higher_level=False,
                 use_edge_attr=None):

        super(NodeEmbedding, self).__init__()
        self.normalize = normalize
        self.type = type
        self.out_dim = out_dim
        self.higher_level = higher_level
        assert higher_level \
            if use_edge_attr and use_edge_attr is not 'none' else True
        self.use_edge_attr = use_edge_attr
        if type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == 'gin':
            self.act = create_act(act, out_dim)
            mlps = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                self.act,
                nn.Linear(out_dim, out_dim))
            self.conv = GINConv(mlps)
        elif type == 'gat':
            self.conv = GATConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        else:
            raise ValueError(
                'Unknown node embedding layer type {}'.format(type))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, ins, batch_data, model):
        x = ins
        if self.higher_level:
            edge_index = batch_data.merge_higher_level['merge'].edge_index
        else:
            edge_index = batch_data.merge_data['merge'].edge_index
        edge_weight = None
        if hasattr(batch_data.merge_data['merge'], "edge_attr"):
            edge_weight = batch_data.merge_data['merge'].edge_attr
        if self.type == "gcn":
            x = self.conv(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv(x, edge_index)
        x = self.act(x)
        if self.bn:
            x = self.bn(x)
        model.store_layer_output(self, x)

        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x


class Loss(nn.Module):
    def __init__(self, type):
        super(Loss, self).__init__()
        self.type = type
        if type == 'BCE':
            self.loss = nn.BCELoss()
        elif type == 'BCEWithLogits':  # contains a sigmoid
            self.loss = nn.BCEWithLogitsLoss()
        elif type == 'CE':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown loss layer type {}'.format(type))

    def forward(self, ins, batch_data, _):
        if self.type != 'CE':
            y_pred = ins.view(len(batch_data.batch_gids))
            y_true = torch.tensor([pair.true_label for pair in batch_data.pair_list],
                                  dtype=torch.float, device=FLAGS.device)
        else:
            y_pred = ins
            y_true = torch.tensor([pair.true_label for pair in batch_data.pair_list],
                                  dtype=torch.long, device=FLAGS.device)
        loss = self.loss(y_pred, y_true)
        return loss


def get_prev_layer(this_layer, model):
    for i, layer in enumerate(model.layers):
        j = i + 1
        if j < len(model.layers) and this_layer == model.layers[j]:
            return layer



