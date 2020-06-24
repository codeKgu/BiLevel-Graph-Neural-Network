from collections import OrderedDict, defaultdict

import numpy as np
from pprint import pprint
from sklearn import metrics

from config import FLAGS, COMET_EXPERIMENT
from utils import rank_metrics


dataset_splits = ["training", "validation", "test"]


class Eval(object):
    def __init__(self, trained_model, eval_data, pairs,
                 set_name="validation", saver=None):
        self.trained_model = trained_model
        self.eval_data = eval_data
        self.saver = saver
        self.global_result_dict = OrderedDict()
        self.pairs = pairs
        if set_name not in dataset_splits:
            raise ValueError("set_name is not in {}".format(dataset_splits))
        self.set_name  = set_name

    def eval(self, round=None, fold_str=''):
        if round is None:
            info = 'final_test'
            d = OrderedDict()
            self.global_result_dict[info] = d
        else:
            raise NotImplementedError()

        d['pairwise'], supplement = self._eval_pairs(info)

        if self.saver:
            self.saver.save_global_eval_result_dict(self.global_result_dict)
        if COMET_EXPERIMENT:
            with COMET_EXPERIMENT.test():
                COMET_EXPERIMENT.log_metrics(self.global_result_dict['final_test']['pairwise'], prefix=fold_str)
                confusion_matrix = self.global_result_dict['final_test']['pairwise'].get('confusion_matrix')
                if 'fine_grained_by_degree' in supplement:
                    for k, v in supplement['fine_grained_by_degree'].items():
                        COMET_EXPERIMENT.log_metrics(v, prefix=fold_str + 'degree_bin_' + str(k))
                if confusion_matrix is not None:
                    labels = [k for k, v in sorted(self.eval_data.dataset.interaction_edge_labels.items(), key=lambda item: item[1])]
                    COMET_EXPERIMENT.log_confusion_matrix(matrix=confusion_matrix, labels=labels)
        return d

    def _eval_pairs(self, info):
        print('Evaluating pairwise results...')
        pair_list = self.pairs
        result_dict, supplement = self.eval_pair_list(pair_list, FLAGS,
                                                      self.eval_data.dataset)
        pprint(result_dict)
        if self.saver:
            self.saver.save_eval_result_dict(result_dict, 'pairwise')
            self.saver.save_pairs_with_results(self.pairs, info, self.set_name)
        return result_dict, supplement

    @staticmethod
    def _prepare_metrics(y_trues, y_preds, multi_class_pred):
        if len(y_trues) == 0 or len(y_preds) == 0:
            return {}
        if not multi_class_pred:
            fpr, tpr, _ = metrics.roc_curve(y_trues, y_preds)
            roc_auc = metrics.auc(fpr, tpr)
            precision, recall, thresholds = metrics.precision_recall_curve(
                y_trues,
                y_preds)
            pr_auc = metrics.auc(recall, precision)
            results = {'roc_auc': roc_auc,
                       'pr_auc': pr_auc}
        else:
            y_pred_label = [np.argmax(pred) for pred in y_preds]
            y_pred_probs = [np.exp(x) / sum(np.exp(x)) for x in y_preds]
            accuracy = metrics.accuracy_score(y_trues, y_pred_label)
            f1 = metrics.f1_score(y_trues, y_pred_label, average='weighted')
            roc_auc_score = metrics.roc_auc_score(y_trues, y_pred_probs,
                                                  multi_class='ovo')
            results = {'roc_auc': roc_auc_score,
                       'f1': f1,
                       'accuracy': accuracy}
        return results

    @staticmethod
    def _eval_per_node(degrees, y_true, y_pred, edge_list_pairs, dataset):
        results_per_degree_bin = defaultdict(lambda: defaultdict(list))
        count_pairs_per_degree = defaultdict(int)
        for node, degree in degrees.items():
            y_trues = [y_true[ind] for ind in edge_list_pairs[dataset.id_map[node]]]
            y_preds = [y_pred[ind] for ind in edge_list_pairs[dataset.id_map[node]]]

            if FLAGS.eval_per_node:
                results = Eval._prepare_metrics(y_trues, y_preds, FLAGS.multi_class_pred)

            for i, degree_bound in enumerate(FLAGS.eval_performance_by_degree):
                if degree <= degree_bound:
                    if FLAGS.eval_per_node:
                        for k, v in results.items():
                            results_per_degree_bin[i][k].append(v)
                    else:
                        results_per_degree_bin[i]['y_trues'].extend(y_trues)
                        results_per_degree_bin[i]['y_preds'].extend(y_preds)
                    results_per_degree_bin[i]['num_results_per_node'].append(len(y_trues))
                    count_pairs_per_degree[i] += 1
                    break
                else:
                    if i == len(FLAGS.eval_performance_by_degree) - 1:
                        if FLAGS.eval_per_node:
                            for k, v in results.items():
                                results_per_degree_bin[i + 1][k].append(v)
                        else:
                            results_per_degree_bin[i+1]['y_trues'].extend(y_trues)
                            results_per_degree_bin[i+1]['y_preds'].extend(y_preds)
                        results_per_degree_bin[i + 1]['num_results_per_node'].append(len(y_trues))
                        count_pairs_per_degree[i + 1] += 1

        if not FLAGS.eval_per_node:
            for degree_bin, res in results_per_degree_bin.items():
                results = Eval._prepare_metrics(res['y_trues'], res['y_preds'], FLAGS.multi_class_pred)
                results_per_degree_bin[degree_bin] = results
                res.pop('y_trues', None)
                res.pop('y_pred', None)
        else:
            for degree_bin, res in results_per_degree_bin.items():
                for metric, list_results in res.items():
                    list_results = [x for x in list_results if str(x) != 'nan']
                    res[metric] = sum(list_results) / len(list_results)
        for degree_bin, num_pairs in count_pairs_per_degree.items():
            results_per_degree_bin[degree_bin]['num_nodes'] = num_pairs

        return results_per_degree_bin

    @staticmethod
    def eval_pair_list(pair_list, FLAGS, dataset=None):
        y_pred = [pair.get_link_pred() for pair in pair_list]
        y_true = [pair.true_label for pair in pair_list]
        if dataset is not None:
            pair_gids = [pair.get_pair_id() for pair in pair_list]
            edge_list_pairs = defaultdict(list)
            for gid, id in dataset.gs_map.items():
                for i, pair_gid in enumerate(pair_gids):
                    if pair_gid[0] == gid or pair_gid[1] == gid:
                        edge_list_pairs[gid].append(i)
            degrees = dict(dataset.interaction_combo_nxgraph.degree())
            results_per_degree_bin = Eval._eval_per_node(degrees, y_true, y_pred, edge_list_pairs, dataset)

        if not FLAGS.multi_class_pred:
            fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
            roc_auc = metrics.auc(fpr, tpr)
            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
            pr_auc = metrics.auc(recall, precision)

            actual_pairs = [pair.get_pair_id()
                            for pair in pair_list if pair.true_label == 1]
            predicted_pairs = [pair.get_pair_id()
                               for pair in sorted(pair_list, key=lambda pair: pair.get_link_pred(), reverse=True)]
            aupr = metrics.average_precision_score(y_true, y_pred)

            apk = rank_metrics.apk(actual_pairs, predicted_pairs, k=FLAGS.ap_at_k)
            eval_results = {"ROC-AUC": float(roc_auc), "PR-AUC": float(pr_auc),
                            "AVG-PR": float(aupr),
                            "AP@{}".format(FLAGS.ap_at_k): float(apk)}
            supplement = {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall, "y_pred": y_pred,
                          "y_true": y_true}

        else:
            y_pred_label = [np.argmax(pred) for pred in y_pred]
            y_pred_probs = [np.exp(x)/sum(np.exp(x)) for x in y_pred]
            accuracy = metrics.accuracy_score(y_true, y_pred_label)
            f1 = metrics.f1_score(y_true, y_pred_label, average='weighted')
            recall = metrics.recall_score(y_true, y_pred_label, average='weighted')
            precision = metrics.precision_score(y_true, y_pred_label, average='weighted')
            roc_auc_score = metrics.roc_auc_score(y_true, y_pred_probs, multi_class='ovo')
            confusion_matrix = metrics.confusion_matrix(y_true, y_pred_label)

            eval_results = {"ROC-AUC": float(roc_auc_score), "F1": f1,
                            "ACCURACY": float(accuracy)}
            supplement = {"precision": precision, "recall": recall, "y_pred": y_pred_probs,
                          "y_true": y_true, 'confusion_matrix': confusion_matrix}

        if dataset is not None:
            supplement['fine_grained_by_degree'] = results_per_degree_bin

        return eval_results, supplement