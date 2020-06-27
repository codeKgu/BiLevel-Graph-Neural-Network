from collections import defaultdict, OrderedDict
import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from utils.data.graph import GraphPair
from utils.util import Timer, random_w_replacement

DATASET_CONFIG = {
    'drugbank': {
        'natts': ['atom_type', 'aromatic', 'acceptor', 'donor',
                  'hybridization', 'num_h'],
        'eatts': []
    },
    'drugcombo': {
        'natts': ['atom_type', 'aromatic', 'acceptor', 'donor',
                  'hybridization', 'num_h'],
        'eatts': []
    }
}


class TorchBiGNNDataset(Dataset):

    def __init__(self, dataset, num_node_feat, device, num_hyper_edge_feat=0):
        self.dataset, self.num_node_feat = dataset, num_node_feat
        self.num_hyper_edge_feat = num_hyper_edge_feat
        self.device = device
        self.gid_pairs = list(self.dataset.train_pairs.keys())
        self.data_items = torch.tensor(sorted(self.gid_pairs), device=self.device)

    def __len__(self):
        return len(self.gid_pairs)

    def __getitem__(self, idx):
        return self.data_items[idx]

    def get_pairs_as_list(self):
        return [self.dataset.look_up_pair_by_gids(gid1.item(), gid2.item())
                for (gid1, gid2) in self.data_items]


class BiGNNDataset(object):
    def __init__(self,  name, graphs, natts, interaction_edge_labels,
                 eatts, pairs, tvt, sparse_node_feat, loaded_dict=None):
        if loaded_dict is not None:  # restore from content loaded from disk
            self.__dict__ = loaded_dict
            self._check_invariants()
            return
        self.name = name
        self.gs = graphs
        self.gs_map = self._gen_gs_map()  # a dict that maps gid to id, id used for enumerating the dataset
        self.id_map = self._gen_id_map()  # a dict that maps id to gid
        self.natts = natts
        self.eatts = eatts
        self.pairs = pairs  # a dict that maps (gid1, gid2) to GraphPair
        self.tvt = tvt
        self._check_invariants()
        self.interaction_edge_labels = interaction_edge_labels
        self.interaction_nxgraphs = None
        self.sparse_interaction_node_feats = sparse_node_feat
        self._gen_interaction_combo_graph()
        self.stats = self._gen_stats()
        self.graph_feats = None
        self.init = False
        self.train_pairs = self.pairs

    def _gen_interaction_combo_graph(self):
        # interaction nx graphs are generated from self.pairs
        if not hasattr(self, 'num_labels'):
            self.num_labels = len({pair.true_label for pair in self.pairs.values()})
        interaction_edge_list = self.edges_to_edge_list(self.pairs.keys())
        adj_mat = get_sparse_mat(interaction_edge_list, self.gs_map, self.gs_map)
        self.interaction_combo_nxgraph = nx.from_scipy_sparse_matrix(adj_mat)
        for node in self.interaction_combo_nxgraph.nodes:
            self.interaction_combo_nxgraph.nodes[node]['gid'] = self.id_map[node]

        edge_type_edge_list = self.edge_type_edge_list_from_pairs()
        self.interaction_nxgraphs = self.create_interaction_nxgraphs(edge_type_edge_list)
        if len(edge_type_edge_list) > 1:
            edge_attrs = {}
            repeats = set()
            for etype, edges in edge_type_edge_list.items():
                for node_i, edges in edges.items():
                    for node_j in edges:
                        edge_attrs[(self.gs_map[node_i], self.gs_map[node_j])] = {'etype': etype}
                        if (self.gs_map[node_j], self.gs_map[node_i]) in edge_attrs.keys():
                            repeats.add((self.gs_map[node_j], self.gs_map[node_i]))

            self_edges = [(i, i) for i in range(self.interaction_combo_nxgraph.number_of_nodes())]
            for i in range(self.interaction_combo_nxgraph.number_of_nodes()):
                if (i, i) not in edge_attrs.keys():
                    edge_attrs[(i, i)] = {'etype': 'none'}
            self.interaction_combo_nxgraph.add_edges_from(self_edges)
            nx.set_edge_attributes(self.interaction_combo_nxgraph, edge_attrs)

        self.interaction_combo_degree_dist, self.interaction_combo_degrees = \
            get_degree_dist(self.interaction_combo_nxgraph)
        self.interaction_combo_nxgraph.init_x = len(self.interaction_combo_nxgraph) * [None]

    def create_interaction_nxgraphs(self, edge_types_edge_list):
        gid_to_adj_idx = {gid: i for i, gid in enumerate(sorted(self.gs_map.keys()))}
        adj_idx_to_gid = {adj_ind: gid for gid, adj_ind in gid_to_adj_idx.items()}
        adj_mats = {edge_type: get_sparse_mat(edge_list, gid_to_adj_idx, gid_to_adj_idx)
                    for edge_type, edge_list in edge_types_edge_list.items()}
        interaction_edge_types_nxgraphs = {}
        for edge_type, adj_mat in adj_mats.items():
            nx_graph = nx.from_scipy_sparse_matrix(adj_mat)
            nx.set_node_attributes(nx_graph, adj_idx_to_gid, 'gid')
            interaction_edge_types_nxgraphs[edge_type] = nx_graph
        return interaction_edge_types_nxgraphs

    def _check_invariants(self):
        self._assert_nonempty_str(self.name)
        assert self.gs and type(self.gs) is list, type(self.gs)
        assert self.gs_map and type(self.gs_map) is dict
        assert len(self.gs) == len(self.gs_map)
        assert type(self.natts) is list
        for natt in self.natts:
            self._assert_nonempty_str(natt)
        assert type(self.eatts) is list
        for eatt in self.eatts:
            self._assert_nonempty_str(eatt)
        self._check_pairs()
        self._assert_nonempty_str(self.tvt)

    def _check_pairs(self):
        assert type(self.pairs) is dict  # may have zero pairs
        for (gid1, gid2), pair in self.pairs.items():
            assert gid1 in self.gs_map and gid2 in self.gs_map, \
                '{} {}'.format(gid1, gid2)
            assert isinstance(pair, GraphPair)

    def print_stats(self):
        print('{} Summary of {}'.format('-' * 10, self.name))
        self._print_stats_helper(self.stats, 1)
        print('{} End of summary of {}'.format('-' * 10, self.name))

    def edge_type_edge_list_from_pairs(self):
        edge_type_edge_list = defaultdict(lambda: defaultdict(set))
        for (gid1, gid2), pair in self.pairs.items():
            assert(hasattr(pair, "edge_types"))
            for edge_type in pair.edge_types:
                edge_type_edge_list[edge_type][gid1].add(gid2)
        return edge_type_edge_list

    @staticmethod
    def edges_to_edge_list(edges):
        edge_list = defaultdict(set)
        for (gid1, gid2) in edges:
            edge_list[gid1].add(gid2)
        return edge_list

    @staticmethod
    def _assert_nonempty_str(s):
        assert s is None or (s and type(s) is str)

    @staticmethod
    def assert_valid_nid(nid, g):
        assert type(nid) is int and (0 <= nid < g.number_of_nodes())

    def init_interaction_graph_embds(self, device):
        if self.init is True:
            return
        assert hasattr(self, "interaction_combo_nxgraph")
        if type(self.interaction_combo_nxgraph.init_x[0]) is torch.Tensor:
            x = self.interaction_combo_nxgraph.init_x
            self.interaction_combo_nxgraph.init_x = torch.cat(x, dim=0).view(len(x), -1)
        else:
            assert self.interaction_combo_nxgraph.init_x[0] is None
            self.interaction_combo_nxgraph.init_x = self.graph_feats
        self.interaction_combo_nxgraph.init_x = self.interaction_combo_nxgraph.init_x.to(device)
        self.init = True

    def init_interaction_graph_feats(self, init_method, device, d_init, feat_size):
        self.interaction_num_node_feat = d_init
        if "graph_feats" in init_method:
            if type(self.sparse_interaction_node_feats) == dict:
                self.graph_feats = torch.tensor(self.sparse_interaction_node_feats[str(feat_size)].todense(),
                                                device=device, dtype=torch.float, requires_grad=False)
                self.interaction_num_node_feat = feat_size
            else:
                self.graph_feats = torch.tensor(self.sparse_interaction_node_feats.todense(),
                                                device=device, dtype=torch.float, requires_grad=False)

        elif "rand_init" in init_method:
            num_graphs = len(self.gs_map)
            init_node_embd = torch.empty(num_graphs, d_init, requires_grad=True)
            self.graph_feats = torch.nn.init.xavier_normal_(init_node_embd,
                                                            gain=torch.nn.init.calculate_gain('relu'))
            self.graph_feats.to(device)
        elif "ones_init" in init_method:
            num_graphs = len(self.gs_map)
            self.graph_feats = torch.ones(num_graphs, d_init, requires_grad=False, device=device)
        elif 'one_hot_init' in init_method:
            num_graphs = len(self.gs_map)
            self.graph_feats = torch.matrix_power(torch.zeros(num_graphs, num_graphs, device=device, requires_grad=False), 0)
            self.interaction_num_node_feat = num_graphs

        elif init_method == "model_init":
            pass
        elif init_method == "no_init":
            num_graphs = len(self.gs_map)
            self.graph_feats = torch.zeros(num_graphs, d_init, requires_grad=False, device=device)
        else:
            raise NotImplementedError

    def add_negative_samples(self, num_samples, with_replacement=True, pairs=None,
                             enforce_negative=True, unique_graphs=None):
        """
        adds negative samples from self.pairs in self if pairs is None
        if pairs is not None adds negative samples from pairs
        if eval add negative samples based on corrupt gids in train_dataset_gids
        """
        print("Generating {} negative samples for each graph pair".format(num_samples))
        random.seed(1)
        rand_func = random_w_replacement if with_replacement else random.sample
        unique_graphs = list(self.gs_map.keys()) if unique_graphs is None else unique_graphs
        new_pairs = {}
        edge_type_pairs = defaultdict(set)
        if pairs is None:
            pairs = self.pairs
        for gids, gpair in pairs.items():
            for edge_type in gpair.edge_types:
                for _ in range(num_samples):
                    seed_val = 0
                    while True:
                        seed_val += 1
                        random.seed(seed_val)
                        corrupt_gid = rand_func(unique_graphs, k=1)[0]
                        if unique_graphs is None:
                            if bool(random.getrandbits(1)):
                                corrupt_gids = (gids[0], corrupt_gid)
                            else:
                                corrupt_gids = (corrupt_gid, gids[1])
                        else:
                            # node in first index of edge is from eval dataset
                            corrupt_gids = (corrupt_gid, gids[1])

                        if not(enforce_negative and corrupt_gids in pairs or corrupt_gids in edge_type_pairs[edge_type]
                               or corrupt_gids in self.pairs.keys()
                               or (corrupt_gids[1], corrupt_gids[0]) in self.pairs.keys()
                               or corrupt_gids[0] == corrupt_gids[1]
                               or corrupt_gids in new_pairs.keys()):
                            break
                    edge_type_pairs[edge_type].add(corrupt_gids)
                    if corrupt_gids not in new_pairs:
                        new_pairs[corrupt_gids] = GraphPair(true_label=0, edge_types=set([edge_type]))
                    else:
                        new_pairs[corrupt_gids].edge_types.add(edge_type)
                if self.name != "ddi_decagon":
                    break
        prev_pairs_len = len(self.pairs)
        new_pairs_len = len(new_pairs)
        self.pairs.update(new_pairs)
        assert(prev_pairs_len + new_pairs_len == len(self.pairs))
        self.stats = self._gen_stats()
        if pairs is not None:
            self.pairs.update(pairs)
            pairs.update(new_pairs)
            return pairs, new_pairs

    def _gen_stats(self):
        stats = OrderedDict()
        stats['#graphs'] = len(self.gs)
        if self.tvt is not None:
            stats['tvt'] = self.tvt
        nn = []
        dens = []
        # natts_stats:
        # node attrib name --> { node attrib value : (count, freq) }
        natts_stats, eatts_stats = OrderedDict(), OrderedDict()
        stats['natts_stats'], stats['eatts_stats'] = natts_stats, eatts_stats
        disconnected = set()
        self._iter_gen_stats(nn, dens, natts_stats, eatts_stats, disconnected)
        # Transform node attrib value count to frequency.
        self._gen_attrib_freq(natts_stats)
        self._gen_attrib_freq(eatts_stats)
        if len(natts_stats) != len(self.natts):
            raise ValueError('Found {} node attributes != specified {}'.format(
                len(natts_stats), len(self.natts)))
        if len(eatts_stats) != len(self.eatts):
            raise ValueError('Found {} edge attributes != specified {}'.format(
                len(natts_stats), len(self.eatts)))
        stats['#disconnected graphs'] = len(disconnected)
        sn = OrderedDict()
        stats['#Nodes'] = sn
        sn['Avg'] = np.mean(nn)
        sn['Std'] = np.std(nn)
        sn['Min'] = np.min(nn)
        sn['Max'] = np.max(nn)
        stats['Avg density'] = np.mean(dens)
        stats['#pairwise results'] = len(self.pairs)
        stats['sqrt(#pairwise results)'] = np.sqrt(len(self.pairs))

        stats["# Interaction Graph Nodes"] = len(self.interaction_combo_nxgraph)
        stats["# Interaction Graph Edges"] = self.interaction_combo_nxgraph.number_of_edges()
        stats["Interaction Graph Density"] = nx.density(self.interaction_combo_nxgraph)

        return stats

    def _iter_gen_stats(self, nn, dens, natts_stats, eatts_stats, disconnected):
        gids = set()
        for g in self.gs:
            gid = g.gid()
            if gid in gids:
                raise ValueError('Graph IDs must be unique. '
                                 'Found two {}s'.format(gid))
            gids.add(gid)
            g = g.get_nxgraph()  # may contain image data; just get nxgraph
            nn.append(g.number_of_nodes())
            dens.append(nx.density(g))
            for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
                BiGNNDataset.assert_valid_nid(n, g)
                assert i == n  # 0-based consecutive node ids
                for k, v in ndata.items():
                    self._add_attrib(natts_stats, k, v)
            for i, (n1, n2, edata) in enumerate(sorted(g.edges(data=True))):
                BiGNNDataset.assert_valid_nid(n1, g)
                BiGNNDataset.assert_valid_nid(n2, g)
                for k, v in edata.items():
                    self._add_attrib(eatts_stats, k, v)
            if not nx.is_connected(g):
                disconnected.add(g)
        assert len(gids) == len(self.gs)

    @staticmethod
    def _add_attrib(d, attr_name, attr_value):
        if attr_name not in d:
            d[attr_name] = defaultdict(int)
        d[attr_name][attr_value] += 1

    @staticmethod
    def _gen_attrib_freq(d):
        for k, dic in d.items():
            new_dic = {}
            sum = np.sum(list(dic.values()))
            for v, count in dic.items():
                new_dic[v] = count / sum
            sorted_li = sorted(new_dic.items(), key=lambda x: x[1],
                               reverse=True)  # sort by decreasing freq
            sorted_li = [(x, '{:.2%}'.format(y)) for (x, y) in sorted_li]
            d[k] = sorted_li

    def _print_stats_helper(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key), end='')
            if type(value) is dict \
                    or type(value) is defaultdict \
                    or type(value) is OrderedDict:
                print()
                self._print_stats_helper(value, indent + 1)
            else:
                pre = '\t' * (indent + 1)
                if type(value) is list:
                    post = ' ({})'.format(len(value))
                    if len(value) > 6:
                        print(pre + str(value[0:3]) +
                              ' ... ' + str(value[-1:]) + post)
                    else:
                        print(pre + str(value) + post)
                else:
                    print(pre + str(value))

    def _gen_gs_map(self):
        rtn = {}
        for i, g in enumerate(self.gs):
            rtn[g.gid()] = i
        return rtn

    def _gen_id_map(self):
        assert (hasattr(self, "gs_map"))
        return {id: gid for gid, id in self.gs_map.items()}

    def num_graphs(self):
        return len(self.gs)

    def get_all_pairs(self):
        return self.pairs

    def look_up_graph_by_gid(self, gid):
        self._check_gid_type(gid)
        id = self.gs_map.get(gid)
        if id is None:
            raise ValueError('Cannot find graph w/ gid {} out of {} graphs'.format(
                gid, len(self.gs_map)))
        assert 0 <= id < len(self.gs)
        return self.gs[id]

    def look_up_pair_by_gids(self, gid1, gid2):
        self._check_gid_type(gid1)
        self._check_gid_type(gid2)
        pair = self.pairs.get((gid1, gid2))
        if pair is None:
            pair = self.pairs.get((gid2, gid1))
            if not pair:
                raise ValueError('Cannot find ({},{}) out of {} pairs'.format(
                    gid1, gid2, len(self.pairs)))
        return pair

    def _check_gid_type(self, gid):
        assert type(gid) is int or type(gid) is np.int64, type(gid)

    def _select_with_gids(self, want_gids):
        t = Timer()
        want_gids = set(want_gids)
        graphs = [g for g in self.gs if g.gid() in want_gids]
        print('Done graphs', t.time_and_clear())
        pairs = {}
        for (gid1, gid2), pair in self.pairs.items():
            # Both g1 and g2 need to be in the (one) train/test/... set.
            if gid1 in want_gids and gid2 in want_gids:
                pairs[(gid1, gid2)] = pair
        print('Done pairs', t.time_and_clear())
        return graphs, pairs


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.items():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            if n == m:
                assoc[a2idx[a], b2idx[b]] = assoc[b2idx[b], a2idx[a]] = 1.
            else:
                assoc[a2idx[a], b2idx[b]] = 1
    assoc = sp.csr_matrix(assoc)
    return assoc


def get_degree_dist(nx_graph):
    degrees = np.asarray([nx_graph.degree(n) for n in nx_graph.nodes()])
    degree_dist = degrees / degrees.sum()
    return degree_dist, degrees