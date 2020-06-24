from collections import defaultdict
from glob import glob
from os.path import join, basename

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils.data.dataset import BiGNNDataset
from utils.data.graph import Graph, GraphPair
from utils.util import get_data_path, load, sorted_nicely


def load_raw_interaction_data(name, natts, eatts, tvt):
    if 'drugbank' in name:
        dir_name = 'DrugBank'
        drugbank_dir = join(get_data_path(), dir_name)
        interaction_dir = join(drugbank_dir, 'ddi_data')
        graph_data = load(join(drugbank_dir, 'klepto', 'graph_data.klepto'))
        fname_to_gid_func = lambda fname: int(fname[2:])
        interaction_fname = 'ddi_snap.tsv'
        parse_edge_func = parse_edges_biosnap
        data_dir = join(interaction_dir, 'drugs_snap')
        if 'small' in name:
            data_dir = join(interaction_dir, 'drugs_small')
        interaction_file_path = join(interaction_dir, interaction_fname)
        edge_types_edge_list, nodes = get_interaction_edgelist(
            interaction_file_path,
            parse_edge_func,
            False,
            fname_to_gid_func)

    elif 'drugcombo' in name:
        dir_name = 'DrugCombo'
        drugcombo_dir = join(get_data_path(), dir_name)
        graph_data = load(join(drugcombo_dir, "klepto", 'graph_data.klepto'))
        data_dir = drugcombo_dir
        interaction_dir = join(drugcombo_dir, 'ddi_data')
        interaction_file_path = join(interaction_dir, 'Syner&Antag_voting.csv')
        drugname_to_cid = load(join(drugcombo_dir, 'klepto', 'drug_name_to_cid'))
        edge_to_gid_func = lambda x: int(drugname_to_cid[x.lower()][4:])
        fname_to_gid_func = lambda x: int(x[4:])

        num_pairs_synergy_antagonism = count_pairs_synergy_antagonism(
            interaction_file_path)

        edge_types_edge_list, nodes = get_interaction_edgelist(
            interaction_file_path,
            parse_edges_drugcombo,
            True,
            edge_to_gid_func,
            skip_first_line=True,
            num_pairs_synergy_antagonism=num_pairs_synergy_antagonism)
    else:
        raise NotImplementedError

    graphs = iterate_get_graphs(data_dir, graph_data, nodes,
                                fname_to_gid_func, natts=natts)
    pairs, graph_ids, edge_types_edge_list_filtered = get_graph_pairs(
        edge_types_edge_list, graphs)
    hyper_edge_labels = {'interaction': 1, 'no_interaction': 0}
    sparse_node_feat, gid_to_idx = get_molecular_node_feats(graph_data,
                                                            graph_ids,
                                                            fname_to_gid_func)
    if 'drugcombo' in name:
        for pair in pairs.values():
            if next(iter(pair.edge_types)) == 'antagonism':
                pair.true_label = 2
        hyper_edge_labels = {'antagonism': 2, 'synergy': 1, 'no_interaction': 0}

    graphs = [graphs[gid] for gid in sorted(graph_ids)]
    return BiGNNDataset(name, graphs, natts, hyper_edge_labels, eatts,
                        pairs, tvt, sparse_node_feat)


def get_molecular_node_feats(graph_data, gids, fname_to_gid_func):
    gid_to_idx = {gid: i for i, gid in enumerate(sorted(list(gids)))}
    gid_graph_data = {fname_to_gid_func(id): g_data for id, g_data in graph_data.items()}
    mats = {}
    for feat_shape in list(gid_graph_data.values())[0]['drug_feat']:
        mat = np.zeros((len(gids), int(feat_shape)))
        for gid in gids:
            mat[gid_to_idx[gid]] = gid_graph_data[gid]["drug_feat"][feat_shape]
        mats[feat_shape] = csr_matrix(mat)

    return mats, gid_to_idx


def get_interaction_edgelist(file_path, parse_edges_func, has_interaction_eatts,
                             edge_to_gid_func, skip_first_line=False, **kwargs):
    # assume each line in file is an edge, parse it using parse_edge_func
    edge_types_edge_list = defaultdict(lambda: defaultdict(list))
    nodes = set()
    skipped = set()
    with open(file_path, 'r') as f:
        readlines = f.readlines() if not skip_first_line else list(f.readlines())[1:]
        for i, line in enumerate(readlines):
            edge, edge_type = parse_edges_func(line, **kwargs)
            if edge:
                try:
                    e1 = edge_to_gid_func(edge[0])
                    e2 = edge_to_gid_func(edge[1])
                except KeyError as e:
                    skipped.add(str(e))
                    continue
                if has_interaction_eatts and edge_type:
                    edge_types_edge_list[edge_type][e1].append(e2)
                else:
                    edge_types_edge_list['default'][e1].append(e2)
                nodes.add(e1)
                nodes.add(e2)
    print("number skipped: ", len(skipped))
    return edge_types_edge_list, nodes


def parse_edges_biosnap(line):
    return line.rstrip('\n').split('\t'), None


def count_pairs_synergy_antagonism(file_path):
    count_syn_ant = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r') as f:
        for i, line in enumerate(list(f.readlines())[1:]):
            line = line.split(',')
            count_syn_ant[tuple(sorted([line[1], line[2]]))][line[-1]] += 1
    return count_syn_ant


def parse_edges_drugcombo(line, **kwargs):
    label_counts = kwargs['num_pairs_synergy_antagonism']
    line = line.split(',')
    drugs = [line[1], line[2]]
    if label_counts[tuple(sorted(drugs))]['synergy\n']\
            >= label_counts[tuple(drugs)]['antagonism\n']:
        label = 'synergy'
    else:
        label = 'antagonism'
    return drugs, label


def get_graph_pairs(edge_types_edge_list, graphs):
    graph_pairs = {}
    no_graph_structures = set()
    final_graphs = set()
    edge_types_edge_list_filtered = defaultdict(lambda: defaultdict(set))
    for edge_type, edge_list in tqdm(edge_types_edge_list.items()):
        for gid1, gid2s in edge_list.items():
            if gid1 not in graphs.keys():
                no_graph_structures.add(gid1)
                continue
            graph1 = graphs[gid1]
            for gid2 in gid2s:
                gid_pair = tuple(sorted([gid1, gid2]))
                if gid_pair not in graph_pairs.keys():
                    if gid2 not in graphs.keys():
                        no_graph_structures.add(gid2)
                        continue
                    graph2 = graphs[gid2]
                    final_graphs.add(gid1)
                    final_graphs.add(gid2)
                    graph_pairs[gid_pair] = GraphPair(true_label=1,
                                                      g1=graph1, g2=graph2,
                                                      edge_types=set(edge_type))
                else:
                    graph_pairs[gid_pair].edge_types.add(edge_type)
                edge_types_edge_list_filtered[edge_type][gid1].add(gid2)
    return graph_pairs, final_graphs, edge_types_edge_list_filtered


def iterate_get_graphs(dir, graph_data, nodes, fname_to_gid_func,
                       check_connected=False, natts=(), eatts=()):
    graphs = {}
    not_connected = []
    no_edges = []
    graphs_not_in_edge_list = []
    for file in tqdm(sorted_nicely(glob(join(dir, '*.gexf')))):
        fname = basename(file).split('.')[0]
        gid = fname_to_gid_func(fname)
        if gid not in nodes:
            graphs_not_in_edge_list.append(fname)
            continue
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        if not nx.is_connected(g):
            msg = '{} not connected'.format(gid)
            if check_connected:
                raise ValueError(msg)
            else:
                not_connected.append(fname)
        # assumes default node mapping to convert_node_labels_to_integers

        nlist = sorted(g.nodes())
        g.graph['node_label_mapping'] = dict(zip(nlist,
                                                 range(0, g.number_of_nodes())))
        add_graph_data_to_nxgraph(g, graph_data[fname])
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        if len(g.edges) == 0:
            no_edges.append(fname)
            continue
        # # Must use sorted_nicely because otherwise may result in:
        # # ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'].
        # # Be very cautious on sorting a list of strings
        # # which are supposed to be integers.
        for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
            BiGNNDataset.assert_valid_nid(n, g)
            assert i == n
            remove_entries_from_dict(ndata, natts)
        for i, (n1, n2, edata) in enumerate(sorted(g.edges(data=True))):
            BiGNNDataset.assert_valid_nid(n1, g)
            BiGNNDataset.assert_valid_nid(n2, g)
            remove_entries_from_dict(edata, eatts)
        graphs[gid] = Graph(g)
    print("total graphs with edges: {}\nnon connected graphs: {}"
          .format(len(graphs), len(not_connected)))
    print("not connected ids: ", not_connected)
    print("num no edges: ", len(no_edges), "\nno edges ids: ", no_edges)
    if not graphs:
        raise ValueError('Loaded 0 graphs from {}\n'
                         'Please download the gexf-formated dataset'
                         ' from Google Drive and extract under:\n{}'.
                         format(dir, get_data_path()))
    return graphs


def remove_entries_from_dict(d, keeps):
    for k in set(d) - set(keeps):
        del d[k]


def add_graph_data_to_nxgraph(g, graph_data):
    if graph_data:
        for k,v in graph_data.items():
            g.graph[k] = v