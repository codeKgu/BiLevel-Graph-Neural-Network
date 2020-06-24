import random
from math import ceil

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from batch import BatchData
from config import FLAGS, COMET_EXPERIMENT
from eval import Eval
from model.model import Model
from sampler import NeighborSampler, RandomSampler, EverythingSampler
from utils.util import Timer, MovingAverage


def train(train_data, val_data, val_pairs, saver, fold_num=None):
    print('creating models...')
    model = Model(train_data)
    model = model.to(FLAGS.device)
    print(model)

    if "model_init" in FLAGS.init_embds:
        print('initial embedding models:')
        print(model.init_layers)
        _get_initial_embd(train_data, model)

    train_data.dataset.init_interaction_graph_embds(device=FLAGS.device)
    val_data.dataset.init_interaction_graph_embds(device=FLAGS.device)

    val_pairs = list(val_pairs)
    random.shuffle(val_pairs)
    val_pairs = torch.stack(val_pairs)
    saver.log_model_architecture(model)
    model.train_data = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, )
    num_iters_total = 0
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.train():
            val_results = _train(num_iters_total, train_data, val_data,
                                 val_pairs, model, optimizer, saver, fold_num)
    else:
        val_results = _train(num_iters_total, train_data, val_data,
                             val_pairs, model, optimizer, saver, fold_num)

    return model, val_results


def _get_initial_embd(train_data, model):
    model_initial_embd = model
    model_initial_embd.train_data = train_data
    model_initial_embd.use_layers = 'lower_layers'
    gids = list(train_data.dataset.gs_map.keys())
    gid_pairs = []
    for i in range(0, len(gids)-2, 2):
        gid_pairs.append(torch.tensor([gids[i], gids[i+1]], dtype=torch.long))
    gid_pairs.append(torch.tensor([gids[len(gids)-2],
                                   gids[len(gids)-1]],
                                  dtype=torch.long))
    all_batch_gids = torch.stack(gid_pairs)
    batch_size = int(len(gid_pairs)/2) if FLAGS.batch_size * 2 >= len(gid_pairs) else FLAGS.batch_size
    out = []
    for i in range(0, all_batch_gids.shape[0]-batch_size, batch_size):
        batch_gids = all_batch_gids[i:i+batch_size]
        batch_data = BatchData(batch_gids, train_data.dataset,
                               is_train=False, ignore_pairs=True)
        out.append(model_initial_embd(batch_data))

    batch_gids = all_batch_gids[i + batch_size:]
    batch_data = BatchData(batch_gids, train_data.dataset,
                           is_train=False, ignore_pairs=True)
    out.append(model_initial_embd(batch_data))
    return torch.cat(out, dim=0)


def model_forward(model, data, sampler=None, is_train=True):
    if sampler is None:
        sampler = RandomSampler(data, FLAGS.batch_size, FLAGS.sample_induced)
    if FLAGS.lower_level_layers and FLAGS.higher_level_layers:
        if "model_init" in FLAGS.init_embds and is_train:
            _get_initial_embd(data, model)
            data.dataset.init_interaction_graph_embds(device=FLAGS.device)
            model.init_x = data.dataset.interaction_combo_nxgraph.init_x.cpu().detach().numpy()

        batch_gids, sampled_gids, subgraph = sampler.sample_next_training_batch()
        batch_data = BatchData(
            batch_gids,
            data.dataset,
            is_train=is_train,
            sampled_gids=sampled_gids,
            enforce_negative_sampling=FLAGS.enforce_negative_sampling,
            unique_graphs=FLAGS.batch_unique_graphs,
            subgraph=subgraph)

        if FLAGS.pair_interaction:
            model.use_layers = 'lower_layers'
            model(batch_data)
        model.use_layers = 'higher_layers'
    else:
        batch_gids, sampled_gids, subgraph = sampler.sample_next_training_batch()
        batch_data = BatchData(
            batch_gids,
            data.dataset,
            is_train=is_train,
            sampled_gids=sampled_gids,
            enforce_negative_sampling=FLAGS.enforce_negative_sampling,
            unique_graphs=FLAGS.batch_unique_graphs,
            subgraph=subgraph)
    return batch_data


def _train(num_iters_total, train_data, val_data,
           train_val_links, model, optimizer, saver, fold_num,
           retry_num=0):
    fold_str = '' if fold_num is None else 'Fold_{}_'.format(fold_num)
    fold_str = fold_str + 'retry_{}_'.format(retry_num) if retry_num > 0 else fold_str
    if fold_str == '':
        print("here")
    epoch_timer = Timer()
    total_loss = 0
    curr_num_iters = 0
    val_results = {}
    if FLAGS.sampler == "neighbor_sampler":
        sampler = NeighborSampler(train_data, FLAGS.num_neighbors_sample, FLAGS.batch_size)
        estimated_iters_per_epoch = ceil((len(train_data.dataset.gs_map) / FLAGS.batch_size))
    elif FLAGS.sampler == "random_sampler":
        sampler = RandomSampler(train_data, FLAGS.batch_size, FLAGS.sample_induced)
        estimated_iters_per_epoch = ceil((len(train_data.dataset.train_pairs) / FLAGS.batch_size))
    else:
        sampler = EverythingSampler(train_data)
        estimated_iters_per_epoch = 1

    moving_avg = MovingAverage(FLAGS.validation_window_size)
    iters_per_validation = FLAGS.iters_per_validation \
        if FLAGS.iters_per_validation != -1 else estimated_iters_per_epoch

    for iter in range(FLAGS.num_iters):
        model.train()
        model.zero_grad()
        batch_data = model_forward(model, train_data, sampler=sampler)
        loss = _train_iter(batch_data, model, optimizer)
        batch_data.restore_interaction_nxgraph()
        total_loss += loss
        num_iters_total_limit = FLAGS.num_iters
        curr_num_iters += 1
        if num_iters_total_limit is not None and \
                num_iters_total == num_iters_total_limit:
            break
        if iter % FLAGS.print_every_iters == 0:
            saver.log_tvt_info("{}Iter {:04d}, Loss: {:.7f}".format(fold_str, iter+1, loss))
            if COMET_EXPERIMENT:
                COMET_EXPERIMENT.log_metric("{}loss".format(fold_str), loss, iter+1)
        if (iter+1) % iters_per_validation == 0:
            eval_res, supplement = validation(model, val_data, train_val_links, saver, max_num_examples=FLAGS.max_eval_pairs)
            epoch = iter / estimated_iters_per_epoch
            saver.log_tvt_info('{}Estimated Epoch: {:05f}, Loss: {:.7f} '
                               '({} iters)\t\t{}\n Val Result: {}'.format(
                fold_str, epoch, eval_res["Loss"], curr_num_iters,
                epoch_timer.time_and_clear(), eval_res))
            if COMET_EXPERIMENT:
                COMET_EXPERIMENT.log_metrics(eval_res, prefix="{}validation".format(fold_str), step=iter+1)
                COMET_EXPERIMENT.log_histogram_3d(supplement['y_pred'], name="{}y_pred".format(fold_str), step=iter+1)
                COMET_EXPERIMENT.log_histogram_3d(supplement['y_true'], name='{}y_true'.format(fold_str), step=iter+1)
                confusion_matrix = supplement.get('confusion_matrix')
                if confusion_matrix is not None:
                    labels = [k for k, v in sorted(batch_data.dataset.interaction_edge_labels.items(), key=lambda item: item[1])]
                    COMET_EXPERIMENT.log_confusion_matrix(matrix=confusion_matrix, labels=labels, step=iter+1)
            curr_num_iters = 0
            val_results[iter+1] = eval_res
            if len(moving_avg.results) == 0 or (eval_res[FLAGS.validation_metric] - 1e-7) > max(moving_avg.results):
                saver.save_trained_model(model, iter + 1)
            moving_avg.add_to_moving_avg(eval_res[FLAGS.validation_metric])
            if moving_avg.stop():
                break
    return val_results


def _train_iter(batch_data, model, optimizer):
    loss = model(batch_data)
    loss.backward()
    optimizer.step()
    loss = loss.item()
    return loss


def evaluate(model, data, eval_links, saver, max_num_examples=None, test=False):
    with torch.no_grad():
        model = model.to(FLAGS.device)
        model.eval()
        total_loss = 0
        all_pair_list = []
        iter_timer = Timer()
        eval_dataset = torch.utils.data.dataset.TensorDataset(eval_links)
        data_loader = DataLoader(eval_dataset, batch_size=FLAGS.batch_size,
                                 shuffle=True)

        for iter, batch_gids in enumerate(data_loader):
            if max_num_examples and len(all_pair_list) >= max_num_examples:
                break

            batch_gids = batch_gids[0]
            if len(batch_gids) == 0:
                continue
            batch_data = BatchData(batch_gids, data.dataset, is_train=False,
                                   unique_graphs=FLAGS.batch_unique_graphs)
            if FLAGS.lower_level_layers and FLAGS.higher_level_layers:
                if FLAGS.pair_interaction:
                    model.use_layers = 'lower_layers'
                    model(batch_data)
                model.use_layers = 'higher_layers'
            else:
                model.use_layers = 'all'
            loss = model(batch_data)

            batch_data.restore_interaction_nxgraph()
            total_loss += loss.item()
            all_pair_list.extend(batch_data.pair_list)
            if test:
                saver.log_tvt_info('\tIter: {:03d}, Test Loss: {:.7f}\t\t{}'.format(
                    iter + 1, loss, iter_timer.time_and_clear()))
    return all_pair_list, total_loss / (iter + 1)


def validation(model, data, val_links, saver, max_num_examples=None):
    pairs, loss = evaluate(model, data, val_links, saver,
                           max_num_examples=max_num_examples)
    res, supplement = Eval.eval_pair_list(pairs, FLAGS)
    res["Loss"] = loss
    return res, supplement


def test(model, data, test_train_links, saver, fold_num):
    print("testing...")
    fold_str = '' if fold_num is None else 'Fold_{}_'.format(fold_num)

    pairs, loss = evaluate(model, data, test_train_links, saver, test=True)
    eval = Eval(model, data, pairs, set_name="test", saver=saver)
    res = eval.eval(fold_str=fold_str)
    if COMET_EXPERIMENT:
        with COMET_EXPERIMENT.test():
            COMET_EXPERIMENT.send_notification(saver.get_f_name(),
                                               status="finished",
                                               additional_data=res)
