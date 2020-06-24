import torch
from torch_scatter import scatter_add
from torch_geometric.data import Data as PyGSingleGraphData

from config import FLAGS

"""
Reference: 
https://github.com/rusty1s/pytorch_geometric/blob/71edd874f6056942c7c1ebdae6854da34f68aeb7/torch_geometric/data/batch.py
"""


class MergedGraphData(PyGSingleGraphData):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.

    """

    def __init__(self, batch=None, **kwargs):
        super(MergedGraphData, self).__init__(**kwargs)
        self.batch = batch
        self.anchor_info = None


    @staticmethod
    def from_data_list(graphs_in_batch, unique_graphs=True):  # merge
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        """
        assert (unique_graphs and type(graphs_in_batch) == dict) \
               or (not unique_graphs and type(graphs_in_batch) == list)

        data_list = list(graphs_in_batch.values()) \
            if unique_graphs else graphs_in_batch

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = MergedGraphData()

        for key in keys:
            batch[key] = []
        batch.batch = []

        node_indices_list = []
        edge_indices_list = []

        cum_node_sum = 0
        cum_edge_sum = 0
        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            num_edges = data.edge_index.shape[1]
            batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long,
                                          device=FLAGS.device))
            for key in data.keys:
                item = data[key]
                item = item + cum_node_sum if data.__cumsum__(key, item) else item
                batch[key].append(item)

            node_indices_list.append((cum_node_sum, cum_node_sum + num_nodes))
            edge_indices_list.append((cum_edge_sum, cum_edge_sum + num_edges))
            cum_edge_sum += num_edges
            cum_node_sum += num_nodes
            gids_to_node_ind = None
            if unique_graphs:
                gids_to_node_ind = {gid: i
                                    for i, gid in enumerate(graphs_in_batch.keys())}

        for key in keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key],
                                          device=FLAGS.device)
            else:
                raise ValueError('Unsupported attribute type.')
        batch.batch = torch.cat(batch.batch, dim=-1)
        if unique_graphs:
            graph_sizes = scatter_add(
                torch.ones(batch.batch.size(0), device=FLAGS.device),
                batch.batch,
                dim=0).type(torch.long).cpu().detach().numpy()

        else:
            gids_to_node_ind = graph_sizes = None

        return {'merge': batch.contiguous(),
                'ind_list': node_indices_list,
                'edge_ind_list': edge_indices_list,
                'gids_to_batch_ind': gids_to_node_ind,
                'graph_sizes': graph_sizes}

