from config import FLAGS
from model.layers_util import MLP

import numpy as np
import torch
import torch.nn as nn


class LinkPred(nn.Module):
    def __init__(self, type, mlp_dim, num_labels, weight_dim=None, batch_unique_graphs=True,
                 multi_label_pred=False):
        super().__init__()
        self.type = type
        self.num_labels = num_labels
        self.batch_unique_graphs = batch_unique_graphs
        self.weight_dim = weight_dim
        self.multi_label_pred = multi_label_pred
        if weight_dim:
            self.weight_matrix = nn.Parameter(torch.zeros((weight_dim, weight_dim)))
            torch.nn.init.xavier_normal_(self.weight_matrix, gain=nn.init.calculate_gain('relu'))
        if self.type not in ["dot_product", "mlp_concat"]:
            raise NotImplementedError
        if self.multi_label_pred:
            assert self.type == 'ntn' or self.type == 'mlp_concat'
        if self.type == 'mlp_concat':
            if multi_label_pred:
                dims = self._calc_mlp_dims(mlp_dim * 2, num_labels, division=8)
                self.mlp_concat = MLP(mlp_dim * 2, num_labels, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)
            else:
                dims = self._calc_mlp_dims(mlp_dim * 2, division=8)
                self.mlp_concat = MLP(mlp_dim * 2, 1, num_hidden_lyr=len(dims), hidden_channels=dims, bn=False)

    @staticmethod
    def _calc_mlp_dims(mlp_dim, output_dim=1, division=2):
        dim = mlp_dim
        dims = []
        while dim > output_dim:
            dim = dim // division
            dims.append(dim)
        dims = dims[:-1]
        return dims

    def forward(self, ins, batch_data, model):
        ins = torch.nn.functional.normalize(ins, p=2, dim=1)

        batch_gids = batch_data.batch_gids.cpu().detach().numpy()\
            if type(batch_data.batch_gids) is torch.Tensor else batch_data.batch_gids
        if self.batch_unique_graphs:
            if FLAGS.higher_level_layers:
                hyper_level_ids = np.vectorize(batch_data.dataset.gs_map.get)(batch_gids)
            else:
                hyper_level_ids = np.vectorize(batch_data.merge_data['gids_to_batch_ind'].get)(batch_gids)
        else:
            hyper_level_ids = np.arange(batch_gids.shape[0] * batch_gids.shape[1]).reshape(batch_gids.shape)

        ids1 = hyper_level_ids[:, 0]
        ids2 = hyper_level_ids[:, 1]
        g1x = ins[ids1]
        g2x = ins[ids2]

        if self.type == "mlp_concat":
            if self.multi_label_pred:
                pair_preds = self.mlp_concat(torch.cat((g1x, g2x), dim=1).float())
            else:
                pair_preds = torch.sigmoid(self.mlp_concat(torch.cat((g1x, g2x), dim=1).float()))
        elif self.type == "dot_product":
            pair_preds = torch.sigmoid(torch.diagonal(torch.matmul(g1x, torch.t(g2x))))

        for i in range(len(batch_data.pair_list)):
            batch_data.pair_list[i].assign_link_pred(pair_preds[i])
        return pair_preds

