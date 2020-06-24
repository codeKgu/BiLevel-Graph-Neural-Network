from copy import deepcopy
import random

from sklearn.model_selection import KFold


def interaction_pair_tvt_split(orig_dataset, tvt_list, ensure_train_connectivity, seed,
                                                 folds=None):
    assert not (ensure_train_connectivity and folds)
    train_sets = []
    val_sets = []
    test_sets = []

    num_pairs_original = len(orig_dataset.pairs)
    num_val = int(tvt_list[1] * num_pairs_original)
    num_test = int((1 - tvt_list[1] - tvt_list[0]) * num_pairs_original)

    if folds is None:
        train_pairs, val_pairs, test_pairs = split_tvt_one_fold(orig_dataset, ensure_train_connectivity,
                                                                 num_val, num_test, seed)
        train_sets.append(train_pairs)
        val_sets.append(val_pairs)
        test_sets.append(test_pairs)
    else:
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        all_pairs = list(orig_dataset.pairs.keys())
        orig_dataset_pairs = deepcopy(orig_dataset.pairs)
        for i, (train_inds, test_inds) in enumerate(kf.split(all_pairs)):
            print(f'\n\n======== Generate Fold {i+1} ========')
            all_pairs_dict = deepcopy(orig_dataset_pairs)
            test_pairs_ids = [all_pairs[ind] for ind in test_inds]
            test_pairs = {pid: all_pairs_dict[pid] for pid in test_pairs_ids}
            all_pairs_dict = {k: v for k, v in all_pairs_dict.items() if k not in test_pairs.keys()}
            train_pairs_ids = [all_pairs[ind] for ind in train_inds]
            train_pairs, val_pairs = split_train_val(train_pairs_ids, all_pairs_dict, num_val, seed)
            train_sets.append(train_pairs)

            orig_dataset.pairs = {**train_pairs}
            orig_dataset.pairs.update({**val_pairs, **test_pairs})
            print("total positive pairs: ", len(orig_dataset.pairs))
            print("positive_val_pairs: ", len(val_pairs))
            print("positive_test_pairs: ", len(test_pairs))
            val_pairs, neg_val_pairs = orig_dataset.add_negative_samples(1, pairs=val_pairs)
            test_pairs, neg_test_pairs = orig_dataset.add_negative_samples(1, pairs=test_pairs)
            print("num_train_pairs: ", len(train_pairs))
            print("all_num_val_pairs: ", len(val_pairs))
            print("all_num_test_pairs: ", len(test_pairs))
            val_sets.append(val_pairs)
            test_sets.append(test_pairs)

    tvt_pairs = {"train": train_sets, "val": val_sets, "test": test_sets}
    return tvt_pairs


def split_tvt_one_fold(orig_dataset, ensure_train_connectivity, num_val, num_test, seed):
    if ensure_train_connectivity:
        pairs_to_not_remove = set()
        edge_list = orig_dataset.interaction_combo_nxgraph.adj
        id_map = orig_dataset.id_map
        for id1, neighbor_edges_dict in edge_list.items():
            random.seed(seed)
            edge = random.sample(neighbor_edges_dict.keys(), k=1)[0]
            pairs_to_not_remove.add((id_map[id1], id_map[edge]))
        pairs_could_remove = {}
        for gids, pair in orig_dataset.pairs.items():
            if gids not in pairs_to_not_remove and (gids[1], gids[0]) not in pairs_to_not_remove:
                pairs_could_remove[gids] = pair
    else:
        pairs_could_remove = orig_dataset.pairs

    sorted_pairs_to_remove = sorted(pairs_could_remove.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    random.seed(seed)
    test_pairs = random.sample(sorted_pairs_to_remove, k=num_test)
    test_pairs = {tup[0]: tup[1] for tup in test_pairs}
    orig_dataset.pairs = {k: v for k, v in orig_dataset.pairs.items() if k not in test_pairs.keys()}
    sorted_pairs_to_remove = [(k, v) for k, v in sorted_pairs_to_remove if k not in test_pairs.keys()]
    random.seed(seed)
    val_pairs = random.sample(sorted_pairs_to_remove, k=num_val)
    val_pairs = {tup[0]: tup[1] for tup in val_pairs}
    train_pairs = {k: v for k, v in orig_dataset.pairs.items() if k not in val_pairs.keys()}

    return train_pairs, val_pairs, test_pairs


def split_train_val(pairs_ids, all_pairs_dict, num_val, seed):
    random.seed(seed)
    val_pair_ids = random.sample(pairs_ids, k=num_val)
    val_pairs = {pid: all_pairs_dict[pid] for pid in val_pair_ids}
    train_pairs = {k: v for k, v in all_pairs_dict.items() if k not in val_pairs.keys()}
    return train_pairs, val_pairs


