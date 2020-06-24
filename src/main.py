#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

from copy import deepcopy
from os.path import basename
from time import time

import torch

from config import FLAGS, COMET_EXPERIMENT
from load_data import load_pair_tvt_splits,  load_pairs_to_dataset, load_dataset
from model.model import Model
from train import train, test, model_forward, _get_initial_embd
from utils.data.interaction_edge_feat import encode_edge_features
from utils.data.representation_node_feat import encode_node_features
from utils.saver import Saver
from utils.util import get_model_info_as_str, convert_long_time_to_str,\
    aggregate_comet_results_from_folds, set_seed

metric_names = []
if 'drugbank' in FLAGS.dataset:
    metric_names = ['_ROC-AUC', '_PR-AUC']
    for i in range(len(FLAGS.eval_performance_by_degree) + 1):
        metric_names.append('degree_bin_{}_num_nodes'.format(i))
        metric_names.append('degree_bin_{}_pr_auc'.format(i))
        metric_names.append('degree_bin_{}_roc_auc'.format(i))


def main():
    tvt_pairs_dict = load_pair_tvt_splits()

    orig_dataset = load_dataset(FLAGS.dataset, 'all',
                                FLAGS.node_feats, FLAGS.edge_feats)

    orig_dataset, num_node_feat = encode_node_features(dataset=orig_dataset)
    num_interaction_edge_feat = encode_edge_features(
        orig_dataset.interaction_combo_nxgraph,
        FLAGS.hyper_eatts)

    for i, (train_pairs, val_pairs, test_pairs) in \
            enumerate(zip(tvt_pairs_dict['train'],
                          tvt_pairs_dict['val'],
                          tvt_pairs_dict['test'])):
        fold_num = i+1
        if FLAGS.cross_val and FLAGS.run_only_on_fold != -1 and FLAGS.run_only_on_fold != fold_num:
            continue

        set_seed(FLAGS.random_seed + 2)
        print(f'======== FOLD {fold_num} ========')
        saver = Saver(fold=fold_num)
        dataset = deepcopy(orig_dataset)
        train_data, val_data, test_data, val_pairs, test_pairs, _ = \
            load_pairs_to_dataset(num_node_feat, num_interaction_edge_feat,
                                  train_pairs, val_pairs, test_pairs,
                                  dataset)
        print('========= Training... ========')
        if FLAGS.load_model is not None:
            print('loading models: {}'.format(FLAGS.load_model))
            trained_model = Model(train_data).to(FLAGS.device)
            trained_model.load_state_dict(torch.load(FLAGS.load_model,
                                                     map_location=FLAGS.device), strict=False)
            print('models loaded')
            print(trained_model)
        else:
            train(train_data, val_data, val_pairs, saver, fold_num=fold_num)
            trained_model = saver.load_trained_model(train_data)
            if FLAGS.save_model:
                saver.save_trained_model(trained_model)

        print('======== Testing... ========')

        if FLAGS.lower_level_layers and FLAGS.higher_level_layers:
            _get_initial_embd(test_data, trained_model)
            test_data.dataset.init_interaction_graph_embds(device=FLAGS.device)
        elif FLAGS.higher_level_layers and not FLAGS.lower_level_layers:
            test_data.dataset.init_interaction_graph_embds(device=FLAGS.device)

        if FLAGS.save_final_node_embeddings:
            with torch.no_grad():
                trained_model = trained_model.to(FLAGS.device)
                trained_model.eval()
                if FLAGS.higher_level_layers:
                    batch_data = model_forward(trained_model, test_data,
                                               is_train=False)
                    trained_model.use_layers = "higher_no_eval_layers"
                    outs = trained_model(batch_data)
                else:
                    outs = _get_initial_embd(test_data, trained_model)
                    trained_model.use_layers = 'all'
            saver.save_graph_embeddings_mat(outs.cpu().detach().numpy(),
                                            test_data.dataset.id_map,
                                            test_data.dataset.gs_map)
            if FLAGS.higher_level_layers:
                batch_data.restore_interaction_nxgraph()

        test(trained_model, test_data, test_pairs, saver, fold_num)
        overall_time = convert_long_time_to_str(time() - t)
        print(overall_time)
        print(saver.get_log_dir())
        print(basename(saver.get_log_dir()))
        saver.save_overall_time(overall_time)
        saver.close()
    if FLAGS.cross_val and COMET_EXPERIMENT:
        results = aggregate_comet_results_from_folds(COMET_EXPERIMENT,
                                                     FLAGS.num_folds,
                                                     metric_names)
        COMET_EXPERIMENT.log_metrics(results, prefix='aggr')


if __name__ == '__main__':
    t = time()
    print(get_model_info_as_str(FLAGS))
    main()

