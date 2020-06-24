from collections import defaultdict
import gc
from os.path import join

import networkx as nx
import torch

from config import FLAGS
from utils.data.dataset import BiGNNDataset, TorchBiGNNDataset
from utils.data.load_raw_data import load_raw_interaction_data
from utils.data.interaction_edge_feat import encode_edge_features
from utils.data.representation_node_feat import encode_node_features
from utils.data.data_split import interaction_pair_tvt_split
from utils.util import get_save_path, load, save, \
    get_flags_with_prefix_as_list


def load_dataset(dataset_name, tvt, node_feats, edge_feats):
    if tvt not in ['train', 'val', 'test', 'all']:
        raise ValueError('Unknown tvt specifier {}'.format(tvt))
    name_list = list((dataset_name, tvt))
    name_list.append('_'.join([node_feat.replace('_', '')
                               for node_feat in node_feats]))
    f_name = '_'.join(name_list)
    f_path = join(get_save_path(), 'dataset', f_name)
    ld = load(f_path)

    if ld:
        dataset = BiGNNDataset(None, None, None, None, None, None,
                               None, None, ld)
    else:
        try:
            dataset = load_raw_interaction_data(dataset_name, node_feats,
                                                edge_feats, tvt)
        except Exception as e:
            print(e)
            raise FileNotFoundError(f'Please get {f_name} from google drive')

        gc.collect()
        save(dataset.__dict__, f_path)

    return dataset


def load_pair_tvt_splits():
    dir = join(get_save_path(), 'pairs_tvt_split')
    train_ratio = int(FLAGS.tvt_ratio[0] * 100)
    val_ratio = int(FLAGS.tvt_ratio[1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    ensure_train_connectivity_str = 'ensure_train_connectivity_{}'\
        .format(str(FLAGS.ensure_train_connectivity).lower())

    num_folds = 1 if FLAGS.cross_val is None else FLAGS.num_folds

    sfn = '{}_{}_seed_{}_folds_{}_train_{}_val_{}_test_{}_num_negative_pairs_' \
          '{}_{}_feat_size_{}_{}'.format(
            FLAGS.dataset,
            FLAGS.random_seed,
            num_folds,
            train_ratio,
            val_ratio,
            test_ratio,
            ensure_train_connectivity_str,
            FLAGS.num_negative_samples if FLAGS.negative_sample else 0,
            '_'.join(get_flags_with_prefix_as_list('node_fe', FLAGS)),
            FLAGS.feat_size,
            '_'.join([node_feat.replace('_', '') for node_feat in FLAGS.node_feats])
    )

    tp = join(dir, sfn)
    rtn = load(tp)
    if rtn:
        tvt_pairs_dict = rtn
    else:
        tvt_pairs_dict = _load_pair_tvt_splits_helper()
        save(tvt_pairs_dict, tp)
    return tvt_pairs_dict


def _load_pair_tvt_splits_helper():
    orig_dataset = load_dataset(FLAGS.dataset, 'all', FLAGS.node_feats, FLAGS.edge_feats)
    orig_dataset, num_node_feat = encode_node_features(dataset=orig_dataset)
    num_higherlevel_edge_feat = encode_edge_features(
        orig_dataset.interaction_combo_nxgraph,
        FLAGS.hyper_eatts)

    orig_dataset.name = FLAGS.dataset
    num_folds = FLAGS.num_folds if FLAGS.cross_val else None

    tvt_pairs_dict = interaction_pair_tvt_split(orig_dataset,
                                                FLAGS.tvt_ratio,
                                                FLAGS.ensure_train_connectivity,
                                                FLAGS.random_seed,
                                                folds=num_folds)

    tvt_pairs_dict['num_node_feat'] = num_node_feat
    tvt_pairs_dict['num_higherlevel_edge_feat'] = num_higherlevel_edge_feat
    return tvt_pairs_dict


def load_pairs_to_dataset(num_node_feat, num_higherlevel_edge_feat,
                          train_pairs, val_pairs, test_pairs, dataset,
                          color_nodes=False):

    nx_graph = dataset.interaction_combo_nxgraph
    dataset.pairs = {**train_pairs}
    dataset._gen_interaction_combo_graph()
    encode_edge_features(dataset.interaction_combo_nxgraph,
                         FLAGS.hyper_eatts)
    dataset.train_pairs = {**dataset.pairs}
    dataset.pairs.update({**val_pairs, **test_pairs})
    train_dataset = dataset
    print("num_total_pairs: ", len(train_dataset.pairs))

    if FLAGS.hyper_eatts != 'None':
        assert (len(val_pairs) + len(test_pairs) + len(dataset.train_pairs)) \
               == len(train_dataset.pairs)
    else:
        assert (len(val_pairs) + len(test_pairs)
                + len(train_dataset.interaction_combo_nxgraph.edges)) \
               == len(train_dataset.pairs)

    print("num_val_pairs: ", len(val_pairs))
    print("num_test_pairs: ", len(test_pairs))
    print("num_training_pairs: ", len(train_dataset.interaction_combo_nxgraph.edges))

    if color_nodes:
        edge_attr_dict = defaultdict(dict)
        for n1, n2 in nx_graph.edges:
            gid1 = dataset.id_map[n1]
            gid2 = dataset.id_map[n2]
            if (gid1, gid2) in train_pairs.keys() or (gid2, gid1) in train_pairs.keys():
                edge_attr_dict[(n1, n2)]["color"] = "black"
            elif (gid1, gid2) in val_pairs or (gid2, gid1) in val_pairs:
                edge_attr_dict[(n1, n2)]["color"] = "red"
            elif (gid1, gid2) in test_pairs or (gid2, gid1) in test_pairs:
                edge_attr_dict[(n1, n2)]["color"] = "green"
            else:
                continue
        nx.set_edge_attributes(nx_graph, edge_attr_dict)

    train_data = TorchBiGNNDataset(train_dataset, num_node_feat, FLAGS.device,
                                   num_higherlevel_edge_feat)

    val_data = test_data = train_data
    val_pairs = torch.tensor(sorted(list(val_pairs.keys())),
                             device=FLAGS.device)
    test_pairs = torch.tensor(sorted(list(test_pairs.keys())),
                              device=FLAGS.device)

    train_data.dataset.init_interaction_graph_feats(
        FLAGS.init_embds, device=FLAGS.device,
        d_init=FLAGS.d_init, feat_size=FLAGS.feat_size)

    val_data.dataset.init_interaction_graph_feats(
        FLAGS.init_embds, device=FLAGS.device,
        d_init=FLAGS.d_init, feat_size=FLAGS.feat_size)
    test_data.dataset.init_interaction_graph_feats(
        FLAGS.init_embds, device=FLAGS.device,
        d_init=FLAGS.d_init, feat_size=FLAGS.feat_size)
    return val_data, train_data, test_data, val_pairs, test_pairs, nx_graph