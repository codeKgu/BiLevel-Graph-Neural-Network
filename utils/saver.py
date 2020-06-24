from collections import OrderedDict
import glob
from os.path import join, getctime

import ntpath
from pprint import pprint
from tensorboardX import SummaryWriter
import torch

from config import FLAGS
from model.model import Model
from utils.util import (
    create_dir_if_not_exists,
    extract_config_code,
    get_ts,
    get_root_path,
    get_model_info_as_str,
    get_model_info_as_command,
    save,
)


class Saver(object):
    def __init__(self, fold=None):
        model_str = self.get_model_str()
        fold_str = '' if fold is None else '_FOLD_{}'.format(fold)
        self.logdir = join(
            get_root_path(),
            'logs',
            '{}{}_{}'.format(model_str, fold_str, get_ts()))
        create_dir_if_not_exists(self.logdir)
        self.writer = SummaryWriter(self.logdir)
        self.model_info_f = self._open('model_info.txt')
        self._log_model_info()
        self._save_conf_code()
        print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

    def close(self):
        self.writer.close()
        if hasattr(self, 'train_log_f'):
            self.train_log_f.close()
        if hasattr(self, 'results_f'):
            self.results_f.close()

    def get_f_name(self):
        return ntpath.basename(self.logdir)

    def get_log_dir(self):
        return self.logdir

    def log_model_architecture(self, model):
        self.model_info_f.write('{}\n'.format(model))
        self.model_info_f.close()

    def log_tvt_info(self, s):
        print(s)
        if not hasattr(self, 'train_log_f'):
            self.train_log_f = self._open('train_log.txt')
        self.train_log_f.write('{}\n'.format(s))

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(extract_config_code())
        p = join(self.get_log_dir(), 'FLAGS')
        print("in _save_conf_code")
        save({'FLAGS': FLAGS}, p, print_msg=False)

    def save_trained_model(self, trained_model, iter=None):
        iter = "_iter_{}".format(iter) if iter is not None else ""
        p = join(self.logdir, 'trained_model{}.pt'.format(iter))
        torch.save(trained_model.state_dict(), p)
        print('Trained models saved to {}'.format(p))

    def load_trained_model(self, train_data):
        p = join(self.logdir, 'trained_model*')
        files = glob.glob(p)
        best_trained_model_path = max(files, key=getctime)
        trained_model = Model(train_data)
        trained_model.load_state_dict(
            torch.load(best_trained_model_path, map_location=FLAGS.device))
        trained_model.to(FLAGS.device)
        return trained_model

    def save_eval_result_dict(self, result_dict, label):
        self._save_to_result_file(label)
        self._save_to_result_file(result_dict)

    def save_pairs_with_results(self, pairs, info, set_name="validation"):
        p = join(self.get_log_dir(), '{}_pairs'.format(info))
        print("in save_pairs_with_results")
        save({'{}_data_pairs'.format(set_name):
                  self._shrink_space_pairs(pairs),
              },
             p, print_msg=False)

    def save_ranking_mat(self, true_m, pred_m, info):
        p = join(self.get_log_dir(), '{}_ranking_mats'.format(info))
        print("in save_ranking_mat")
        save({'true_m': true_m.__dict__, 'pred_m': pred_m.__dict__},
             p, print_msg=False)

    def save_graph_embeddings_mat(self, init_x, id_map, gs_map):
        assert(init_x.shape[0] == len(gs_map))
        p = join(self.get_log_dir(), "graph_embeddings")
        save({"init_x": init_x, "id_map": id_map, "gs_map": gs_map}, p)

    def save_global_eval_result_dict(self, global_result_dict):
        p = join(self.get_log_dir(), 'global_result_dict')
        print("in save_global_eval_result_dict")
        save(global_result_dict, p, print_msg=False)

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    def save_exception_msg(self, msg):
        with self._open('exception.txt') as f:
            f.write('{}\n'.format(msg))

    @staticmethod
    def get_model_str():
        li = []
        key_flags = [FLAGS.model, FLAGS.dataset]
        for f in key_flags:
            li.append(str(f))
        return '_'.join(li)

    def _log_model_info(self):
        s = get_model_info_as_str(FLAGS)
        c = get_model_info_as_command(FLAGS)
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.write(c)
        self.model_info_f.write('\n\n')
        self.writer.add_text('model_info_str', s)
        self.writer.add_text('model_info_command', c)

    def _save_to_result_file(self, obj, name=None):
        if not hasattr(self, 'results_f'):
            self.results_f = self._open('results.txt')
        if type(obj) is dict or type(obj) is OrderedDict:
            pprint(obj, stream=self.results_f)
        elif type(obj) is str:
            self.results_f.write('{}\n'.format(obj))
        else:
            self.results_f.write('{}: {}\n'.format(name, obj))

    def _shrink_space_pairs(self, pairs):
        ret_pairs = {}
        for pair in pairs:
            gid_pair = pair.get_pair_id()
            pair.shrink_space_for_save()
            ret_pairs[gid_pair] = pair
        return ret_pairs
