from collections import OrderedDict
import datetime
from os import makedirs, environ
from os.path import dirname, abspath, exists, join, isfile, expanduser
import random
import re
from socket import gethostname
from statistics import mean, stdev
import sys
from time import time

import klepto
import networkx as nx
import numpy as np
import pickle
import pytz
import torch


def check_nx_version():
    nxvg = '2.2'
    nxva = nx.__version__
    if nxvg != nxva:
        raise RuntimeError(
            'Wrong networkx version! Need {} instead of {}'.format(nxvg, nxva))

# Always check the version first.
check_nx_version()


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return join(get_root_path(), 'data')


def get_save_path():
    return join(get_root_path(), 'save')


def get_src_path():
    return join(get_root_path(), 'src')


def get_model_path():
    return join(get_root_path(), 'models')


def get_temp_path():
    return join(get_root_path(), 'temp')


def create_dir_if_not_exists(dir):
    if not exists(dir):
        makedirs(dir)


def save(obj, filepath, print_msg=True, use_klepto=True):
    if type(obj) is not dict and type(obj) is not OrderedDict:
        raise ValueError('Can only save a dict or OrderedDict'
                         ' NOT {}'.format(type(obj)))
    fp = proc_filepath(filepath, ext='.klepto' if use_klepto else '.pickle')
    if use_klepto:
        create_dir_if_not_exists(dirname(filepath))
        save_klepto(obj, fp, print_msg)
    else:
        save_pickle(obj, fp, print_msg)


def load(filepath, print_msg=True):
    fp = proc_filepath(filepath)
    if isfile(fp):
        return load_klepto(fp, print_msg)
    elif print_msg:
        print('Trying to load but no file {}'.format(fp))


def save_klepto(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    klepto.archives.file_archive(filepath, dict=dic).dump()


def load_klepto(filepath, print_msg):
    rtn = klepto.archives.file_archive(filepath)
    rtn.load()
    if print_msg:
        print('Loaded from {}'.format(filepath))
    return rtn


def save_pickle(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    with open(filepath, 'wb') as handle:
        if sys.version_info.major < 3:  # python 2
            pickle.dump(dic, handle)
        elif sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError()


def load_pickle(filepath, print_msg=True):
    fp = proc_filepath(filepath, '.pickle')
    if isfile(fp):
        with open(fp, 'rb') as handle:
            pickle_data = pickle.load(handle)
            return pickle_data
    elif print_msg:
        print('No file {}'.format(fp))


def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp


def sorted_nicely(l, reverse=False):
    def try_int(s):
        try:
            return int(s)
        except ValueError as e:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [try_int(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn


TIME_STAMP = None


def get_ts():
    global TIME_STAMP
    if not TIME_STAMP:
        TIME_STAMP = get_current_ts()
    return TIME_STAMP


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')


def get_user():
    try:
        home_user = expanduser("~").split('/')[-1]
    except:
        home_user = 'user'
    return home_user


def get_host():
    host = environ.get('HOSTNAME')
    if host is not None:
        return host
    return gethostname()


class C(object):  # counter
    def __init__(self):
        self.count = 0

    def c(self):  # count and increment itself
        self.count += 1
        return self.count

    def t(self):  # total
        return self.count

    def reset(self):
        self.count = 0


class Timer(object):
    def __init__(self):
        self.t = time()
        self.durations_log = OrderedDict()

    def time_and_clear(self, log_str='', only_seconds=False):
        duration = self._get_duration_and_reset()
        if log_str:
            if log_str in self.durations_log:
                raise ValueError('log_str {} already in log {}'.format(
                    log_str, self.durations_log))
            self.durations_log[log_str] = duration
        if only_seconds:
            rtn = duration
        else:
            rtn = format_seconds(duration)
        return rtn

    def start_timing(self):
        self.t = time()

    def print_durations_log(self):
        print('Timer log', '*' * 50)
        rtn = []
        tot_duration = sum([sec for sec in self.durations_log.values()])
        print('Total duration:', format_seconds(tot_duration))
        lss = np.max([len(s) for s in self.durations_log.keys()])
        for log_str, duration in self.durations_log.items():
            s = '{0}{1} : {2} ({3:.2%})'.format(
                log_str, ' ' * (lss - len(log_str)), format_seconds(duration),
                         duration / tot_duration)
            rtn.append(s)
            print(s)
        print('Timer log', '*' * 50)
        return rtn

    def _get_duration_and_reset(self):
        now = time()
        duration = now - self.t
        self.t = now
        return duration

    def get_duration(self):
        now = time()
        duration = now - self.t
        return duration

    def reset(self):
        self.t = time()


def format_seconds(seconds):
    """
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('min', 60),
        ('sec', 1)
    ]

    if seconds <= 1:
        return '{:.3f} secs'.format(seconds)

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            if period_name == 'sec':
                period_value = seconds
                has_s = 's'
            else:
                period_value, seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
            strings.append('{:.3f} {}{}'.format(period_value, period_name, has_s))

    return ', '.join(strings)


def random_w_replacement(input_list, k=1):
    return [random.choice(input_list) for _ in range(k)]


def get_model_info_as_str(config_flags):
    rtn = []
    d = vars(config_flags)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def get_model_info_as_command(config_flags):
    rtn = []
    d = vars(config_flags)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '--{}={}'.format(k, v)
        rtn.append(s)
    return 'python {} {}'.format(join(get_src_path(), 'main.py'), '  '.join(rtn))


def extract_config_code():
    with open(join(get_src_path(), 'config.py')) as f:
        return f.read()


def get_flags_with_prefix_as_list(prefix, config_flags):
    rtn = []
    d = vars(config_flags)
    i_check = 1  # one-based
    for k in sorted_nicely(d.keys()):
        v = d[k]
        sp = k.split(prefix)
        if len(sp) == 2 and sp[0] == '' and sp[1].startswith('_'):
            id = int(sp[1][1:])
            if i_check != id:
                raise ValueError('Wrong flag format {}={} '
                                 '(should start from _1'.format(k, v))
            rtn.append(v)
            i_check += 1
    return rtn


def convert_long_time_to_str(sec):
    def _give_s(num):
        return '' if num == 1 else 's'

    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} day{} {} hour{} {} min{} {:.1f} sec{}'.format(
        int(day), _give_s(int(day)), int(hour), _give_s(int(hour)),
        int(minutes), _give_s(int(minutes)), seconds, _give_s(seconds))


def parse_str_list(strl):
    if strl == 'None':
        return None
    return strl.split(',')


def get_metric_names(dataset_name, eval_performance_by_degree_bins):
    metric_names = []
    if 'drugbank' in dataset_name:
        metric_names = ['_ROC-AUC', '_PR-AUC']
        for i in range(len(eval_performance_by_degree_bins) + 1):
            metric_names.append('degree_bin_{}_num_nodes'.format(i))
            metric_names.append('degree_bin_{}_pr_auc'.format(i))
            metric_names.append('degree_bin_{}_roc_auc'.format(i))

    elif 'drugcombo' in dataset_name:
        metric_names = ['_ROC-AUC', '_ACCURACY', '_F1']
        for i in range(len(eval_performance_by_degree_bins) + 1):
            metric_names.append('degree_bin_{}_num_nodes'.format(i))
            metric_names.append('degree_bin_{}_accuracy'.format(i))
            metric_names.append('degree_bin_{}_f1'.format(i))
            metric_names.append('degree_bin_{}_roc_auc'.format(i))

    return metric_names


def aggregate_comet_results_from_folds(comet_obj, num_folds, dataset_name,
                                       eval_performance_by_degree_bins):
    metric_names = get_metric_names(dataset_name,
                                    eval_performance_by_degree_bins)
    total_results = {}
    for metric_name in metric_names:
        res_list = []
        for fold in range(1, num_folds+1):
            name = 'Fold_{}_{}'.format(fold, metric_name)
            try:
                metric = comet_obj.get_metric(name)
            except KeyError:
                continue
            res_list.append(metric)
        if len(res_list) <= 1:
            continue
        metric_avg = mean(res_list)
        metric_stdev = stdev(res_list)
        total_results[metric_name + '_mean'] = metric_avg
        total_results[metric_name + '_stdev'] = metric_stdev
    return total_results


def set_seed(seed, all_gpus=True):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if all_gpus:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MovingAverage(object):
    def __init__(self, window, want_increase=True):
        self.moving_avg = [float('-inf')] if want_increase else [float('inf')]
        self.want_increase = want_increase
        self.results = []
        self.window = window

    def mean_of_last_k(self, k):
        if k > len(self.results):
            k = len(self.results)
        return mean(self.results[-k:]), k

    def add_to_moving_avg(self, x):
        self.results.append(x)
        if len(self.results) >= self.window:
            next_val = sum(self.results[-self.window:]) / self.window
            self.moving_avg.append(next_val)

    def stop(self):
        print(self.results)
        if len(self.moving_avg) < 2:
            return False
        if self.want_increase:
            return (self.moving_avg[-1] + 1e-7) < self.moving_avg[-2]
        else:
            return (self.moving_avg[-2] + 1e-7) < self.moving_avg[-1]
