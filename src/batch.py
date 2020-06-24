import copy

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_undirected

from config import FLAGS
from merged_graph import MergedGraphData
from model.layers_util import convert_nx_to_pyg_graph
from utils.data.graph import GraphPair


class BatchData(object):
    """Mini-batch.

    We assume the following sequential models architecture: Merge --> Split.

        Merge: For efficiency, first merge graphs in a batch into a large graph.
            This is only done for the first several `NodeEmbedding` layers.

        Split: For flexibility, split the merged graph into individual pairs.
            The `gen_list_view_by_split` function should be called immediately
            after the last `NodeEmbedding` layer.
    """

    def __init__(self, batch_gids, dataset,
                 sampled_gids=None, curr_sample_edge_type=None,
                 is_train=True, ignore_pairs=False,
                 enforce_negative_sampling=True,
                 unique_graphs=True, subgraph=None):
        self.dataset = dataset
        self.is_train = is_train

        if type(batch_gids) is torch.Tensor:
            self.batch_gids = batch_gids.cpu().detach().numpy().astype('int')
        else:
            self.batch_gids = batch_gids
        if hasattr(dataset, "interaction_combo_nxgraph"):
            if is_train:
                self.interaction_combo_nxgraph = dataset.interaction_combo_nxgraph
            else:
                self.interaction_combo_nxgraph = self.copy_nx_graph(dataset.interaction_combo_nxgraph)

        self.unique_graphs = unique_graphs
        self.positive_pair_gids = self.batch_gids
        self.ignore_pairs = ignore_pairs
        self.curr_sample_edge_type = curr_sample_edge_type
        if FLAGS.negative_sample and is_train:
            assert(sampled_gids is not None)
            self.sampled_gids = sampled_gids
            self.sample_negative_pairs(enforce_negative_sampling)
        self.merge_data, self.pair_list = self._merge_into_one_graph(self.batch_gids)
        self.batch_interaction_inds = [self.dataset.gs_map[gid] for gid in self.batch_gids.flatten()]
        self.merge_higher_level = {}
        if subgraph is not None:
            subgraph = nx.relabel_nodes(subgraph, dataset.id_map)  # now nodes labeled by gid
            subgraph = nx.relabel_nodes(subgraph, self.merge_data["gids_to_batch_ind"])
            self.subgraph = subgraph

    def sample_negative_pairs(self, enforce_negative=True):
        # node_degree_dist = self.dataset.node_degree_dists[self.curr_sample_edge_type]
        # node_degree_dist = self.interaction_combo_degree_dist
        negative_pair_gids = set()
        gid_ind = 0
        num_gids = len(self.sampled_gids)
        # number of pairs in a complete graph minus positive pairs
        if FLAGS.enforce_sampling_amongst_same_graphs:
            max_pairs = ((num_gids * (num_gids - 1)) / 2) - len(self.batch_gids)
        else:
            max_pairs = float('inf')
        gs_map = self.dataset.gs_map
        pos_edges = set(self.interaction_combo_nxgraph.edges).union(
            set([(gs_map[edge[0]], gs_map[edge[1]]) for edge in self.positive_pair_gids]))

        if FLAGS.enforce_sampling_amongst_same_graphs:
            negative_gids = self.sampled_gids
        else:
            negative_gids = list(self.dataset.gs_map.keys())

        while True:
            if len(negative_pair_gids) == min(max_pairs, (len(self.batch_gids) * (FLAGS.num_negative_samples))):
                break
            orig_gid = self.sampled_gids[gid_ind % len(self.sampled_gids)]
            negative_id = np.random.choice(negative_gids, size=1)[0]

            gid_ind += 1
            while enforce_negative and \
                    (((orig_gid, negative_id) in negative_pair_gids)
                     or ((negative_id, orig_gid) in negative_pair_gids)
                     or (orig_gid == negative_id)
                     or ((gs_map[orig_gid], gs_map[negative_id]) in pos_edges)
                     or ((gs_map[negative_id], gs_map[orig_gid]) in pos_edges)):
                orig_gid = self.sampled_gids[gid_ind % len(self.sampled_gids)]
                gid_ind += 1
                negative_id = np.random.choice(negative_gids, size=1)[0]

            negative_pair_gids.add((orig_gid, negative_id))

        if len(negative_pair_gids) > 0:
            self.negative_pairs = {negative_pair_gid: GraphPair(true_label=0) for negative_pair_gid in negative_pair_gids}
            self.negative_pair_gids = np.asarray(list(self.negative_pairs.keys()))
            self.batch_gids = np.concatenate((self.batch_gids, np.asarray(list(self.negative_pairs.keys()))))

    def _merge_into_one_graph(self, batch_gids):
        graphs_in_batch = {} if self.unique_graphs else []
        metadata_list = []
        pair_list = []
        gids1 = batch_gids[:, 0]
        gids2 = batch_gids[:, 1]
        assert gids1.shape == gids2.shape
        for (gid1, gid2) in zip(gids1, gids2):
            self._preproc_gid_pair(gid1, gid2, graphs_in_batch, metadata_list, pair_list)
        assert self.ignore_pairs or (len(pair_list) == gids1.shape[0] == gids2.shape[0])
        return MergedGraphData.from_data_list(graphs_in_batch, self.unique_graphs), pair_list

    def _preproc_gid_pair(self, gid1, gid2, graphs_in_batch, metadata_list, pair_list):
        assert gid1 - int(gid1) == 0
        assert gid2 - int(gid2) == 0
        gid1 = int(gid1)
        gid2 = int(gid2)
        g1 = self.dataset.look_up_graph_by_gid(gid1)
        g2 = self.dataset.look_up_graph_by_gid(gid2)
        try:
            pair = self.dataset.look_up_pair_by_gids(g1.gid(), g2.gid())
        except ValueError:
            if not self.ignore_pairs:
                pair = self.negative_pairs[(gid1, gid2)]
            else:
                pair = None
        if self.unique_graphs:
            if not gid1 in graphs_in_batch.keys():
                graphs_in_batch[gid1] = \
                    convert_nx_to_pyg_graph(g1.get_nxgraph())
            if not gid2 in graphs_in_batch.keys():
                graphs_in_batch[gid2] = convert_nx_to_pyg_graph(g2.get_nxgraph())
        else:
            graphs_in_batch.append(convert_nx_to_pyg_graph(g1.get_nxgraph()))
            graphs_in_batch.append(convert_nx_to_pyg_graph(g2.get_nxgraph()))

        # metadata_list.extend(this_metadata_list)
        if not self.ignore_pairs:
            pair.assign_g1_g2(g1, g2)
            pair_list.append(pair)

    def copy_nx_graph(self, nx_graph):
        if hasattr(nx_graph, "init_x") and type(nx_graph.init_x) is torch.Tensor:
            self.save_init_x = nx_graph.init_x
            nx_graph.init_x = self.save_init_x.clone()
        return nx_graph

    def restore_interaction_nxgraph(self):
        if type(self.dataset.interaction_combo_nxgraph.init_x) == torch.Tensor:
            if hasattr(self, "save_init_x"):
                self.dataset.interaction_combo_nxgraph.init_x = self.save_init_x.detach()
            else:
                self.dataset.interaction_combo_nxgraph.init_x = self.dataset.interaction_combo_nxgraph.init_x.detach()

        if hasattr(self, "added_edges"):
            self.interaction_combo_nxgraph.remove_edges_from(self.added_edges)
        if hasattr(self, "edges_deleted"):
            self.dataset.interaction_combo_nxgraph.add_edges_from(self.edges_deleted)


def create_edge_index(g):
    if g.edges:
        edge_index = torch.tensor(list(g.edges),
                                  device=FLAGS.device).t().contiguous()
    else:
        edge_index = torch.tensor([], device=FLAGS.device)

    edge_index = to_undirected(edge_index, num_nodes=g.number_of_nodes())
    return edge_index


def copy_nx_graph(nx_graph):
    if hasattr(nx_graph, "init_x") and type(nx_graph.init_x) is torch.Tensor:
        old_init_x = nx_graph.init_x
        nx_graph.init_x = None
        ret = copy.deepcopy(nx_graph)
        nx_graph.init_x = old_init_x
        ret.init_x = old_init_x.clone()
    else:
        ret = copy.deepcopy(nx_graph)
    return ret