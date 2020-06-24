import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add

from config import FLAGS
from model.layers_util import MLP



class NodeAggregationPairs(nn.Module):
    def __init__(self, style, concat_multi_scale=False,
                 in_dim=None, out_dim=None, num_mlp_layers=None):
        super().__init__()
        self.style = style
        self.concat_multi_scale = concat_multi_scale
        if self.style == "avg_pool":
            self.agg_func = scatter_mean
        elif self.style == "sum":
            self.agg_func = scatter_add
        elif self.style == 'deepsets':
            self.agg_func = DeepSets(in_dim, out_dim, num_mlp_layers)
        elif self.style == 'gmn_aggr':
            self.agg_func = GMNAggregatorPairs(in_dim, out_dim, bn=True)
        else:
            raise NotImplementedError("{} is not implemented".format(style))

    def forward(self, ins, batch_data, model, pair_batch=False):
        if pair_batch and hasattr(batch_data, "pair_batch"):
            batch = batch_data.pair_batch
        else:
            batch = batch_data.merge_data['merge'].batch
        if self.concat_multi_scale:
            if 'gmn' not in FLAGS.model:
                out = torch.cat([self.agg_func(x, batch, dim=0)
                                 for x in model.acts[1:]], dim=1)
            else:
                out = torch.cat([self.agg_func(x, batch, dim=0)
                                 for i, x in enumerate(model.acts[1:])
                                 if i % 2 == 1], dim=1)
        else:
            out = self.agg_func(ins, batch, dim=0)
        return out


class DeepSets(nn.Module):
    def __init__(self, in_dim, out_dim, num_mlp_layers):
        super(DeepSets, self).__init__()
        self.phi = MLP(in_dim, out_dim, num_hidden_lyr=num_mlp_layers-1)
        self.rho = MLP(out_dim, out_dim, num_hidden_lyr=num_mlp_layers-1)

    def forward(self, ins, batch, dim):
        # not sure whether to divide h_ins by graph size
        h_ins = self.phi(ins)
        h_ins = scatter_mean(h_ins, batch, dim=dim)
        h_out = self.rho(h_ins)
        return h_out


class NodeAggregation(NodeAggregationPairs):
    """aggregates node embeddings for a graph embedding (one unique embedding per graph)"""

    def __init__(self, style, is_last_layer, **kwargs):
        super(NodeAggregation, self).__init__(style, **kwargs)
        self.is_last_layer = is_last_layer

    def forward(self, x, batch_data, model, pair_batch=False):
        out = super().forward(x, batch_data, model, pair_batch)
        if not FLAGS.higher_level_layers:
            return out
        gids_to_out_ind = batch_data.merge_data['gids_to_batch_ind']
        gs_map = batch_data.dataset.gs_map
        for gid, ind in gids_to_out_ind.items():
            single_graph_x = out[ind]
            batch_data.interaction_combo_nxgraph.init_x[gs_map[gid]] = single_graph_x
        return out


class GMNAggregatorPairs(nn.Module):
    def __init__(self, input_dim, output_dim, bn):
        super().__init__()
        self.out_dim = output_dim
        self.sigmoid = nn.Sigmoid()
        self.weight_func = MLP(input_dim, output_dim, num_hidden_lyr=1,
                               hidden_channels=[output_dim], bn=bn)
        self.gate_func = MLP(input_dim, output_dim, num_hidden_lyr=1,
                             hidden_channels=[output_dim], bn=bn)
        self.mlp_graph = MLP(output_dim, output_dim, num_hidden_lyr=1,
                             hidden_channels=[output_dim], bn=bn)

    def forward(self, x, batch, dim):
        weighted_x = self.weight_func(x)  # shape N by input_dim
        gated_x = self.sigmoid(self.gate_func(x))  # shape N by input_dim
        hammard_prod = gated_x * weighted_x
        graph_embeddings = scatter_add(hammard_prod, batch, dim=dim)  # shape G by output_dim
        return self.mlp_graph(graph_embeddings)

