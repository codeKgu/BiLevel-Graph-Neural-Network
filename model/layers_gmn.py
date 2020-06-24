from model.layers_util import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class GMNPropagator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 distance_metric='cosine', f_node='MLP'):
        super().__init__()
        self.out_dim = output_dim
        if distance_metric == 'cosine':
            self.distance_metric = nn.CosineSimilarity()
        elif distance_metric == 'euclidean':
            self.distance_metric = nn.PairwiseDistance()
        # 2*input_dim because in_dim = dim(g1) + dim(g2)
        self.f_messasge = MLP(2 * input_dim, 2 * input_dim, num_hidden_lyr=1,
                              hidden_channels=[2 * input_dim])
        self.f_node_name = f_node
        if f_node == 'MLP':
            # 2 * input_dim for m_sum, 1 * input_dim
            # for u_sum and 1 * input_dim for x
            self.f_node = MLP(4 * input_dim, output_dim,
                              num_hidden_lyr=1, bn=True)
        elif f_node == 'GRU':
            # 2 * input_dim for m_sum, 1 * input_dim for u_sum
            self.f_node = nn.GRUCell(3 * input_dim,
                                     input_dim)
        else:
            raise ValueError("{} for f_node has not been implemented".format(f_node))

    def forward(self, ins, batch_data, model, edge_index=None):
        x = ins  # x has shape N(gs) by D
        edge_index = edge_index if edge_index is not None \
                        else batch_data.merge_data['merge'].edge_index
        row, col = edge_index
        m = torch.cat((x[row], x[col]), dim=1)  # E by (2 * D)
        m = self.f_messasge(m)
        # N(gs) by (2 * D)
        m_sum = scatter_add(m, row, dim=0, dim_size=x.size(0))
        # u_sum has shape N(gs) by D
        u_sum = self.f_match(x, batch_data)
        if self.f_node_name == 'MLP':
            in_f_node = torch.cat((x, m_sum, u_sum), dim=1)
            out = self.f_node(in_f_node)
        elif self.f_node_name == 'GRU':
            in_f_node = torch.cat((m_sum, u_sum), dim=1)  # N by 3*D
            out = self.f_node(in_f_node, x)

        model.store_layer_output(self, out)
        return out

    def f_match(self, x, batch_data):
        '''from the paper https://openreview.net/pdf?id=S1xiOjC9F7'''
        ind_list = batch_data.merge_data['ind_list']
        u_all_l = []

        for i in range(0, len(ind_list), 2):
            g1_ind = i
            g2_ind = i + 1
            g1x = x[ind_list[g1_ind][0]: ind_list[g1_ind][1]]
            g2x = x[ind_list[g2_ind][0]: ind_list[g2_ind][1]]

            u1 = self._f_match_helper(g1x, g2x)  # N(g1) by D tensor
            u2 = self._f_match_helper(g2x, g1x)  # N(g2) by D tensor

            u_all_l.append(u1)
            u_all_l.append(u2)

        return torch.cat(u_all_l, dim=0).view(x.size(0), -1)

    def _f_match_helper(self, g1x, g2x):
        g1_norm = torch.nn.functional.normalize(g1x, p=2, dim=1)
        g2_norm = torch.nn.functional.normalize(g2x, p=2, dim=1)
        g1_sim = torch.matmul(g1_norm, torch.t(g2_norm))

        # N_1 by N_2 tensor where a1[x][y] is the softmaxed a_ij of the yth node of g2 to the xth node of g1
        a1 = F.softmax(g1_sim, dim=1)

        sum_a1_h = torch.sum(g2x * a1[:, :, None],
                             dim=1)  # N1 by D tensor where each row is sum_j(a_j * h_j)
        return g1x - sum_a1_h


