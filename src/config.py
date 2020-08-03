import argparse
from math import isclose
import importlib
if importlib.util.find_spec('comet_ml'):
    from comet_ml import Experiment
import sys
from os.path import dirname, abspath, join

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, join(dirname(cur_folder), 'src'))
sys.path.insert(0, dirname(cur_folder))


import torch

from utils.data.dataset import DATASET_CONFIG
from utils.util import C, get_user, get_host


parser = argparse.ArgumentParser()
COMET_ML_API_KEY = ''  # YOUR_COMET_API_KEY
COMET_PROJECT_NAME = ''  # YOUR_COMET_PROJECT_NAME
"""
Most Relevant
"""

debug = False
gpu = 1  # -1 if use cpu
use_comet_ml = False if importlib.util.find_spec('comet_ml') and not debug else False

parser.add_argument('--use_comet_ml', default=use_comet_ml)
parser.add_argument('--debug', default=debug)

if use_comet_ml:
    parser.add_argument('--comet_api_key', default=COMET_ML_API_KEY)

"""
Data.
"""

""" 
dataset
"""

# dataset = 'drugbank_small'
dataset = 'drugbank'
# dataset = 'drugcombo'

parser.add_argument('--random_seed', default=3)

if 'drugcombo' in dataset:
    parser.add_argument('--hypergraph',
                        default='combo',
                        choices=['synergy', 'combo', 'antagonism'],
                        help='what edge type to use in the interaction graph. '
                             'combo is all edge type combined')

parser.add_argument('--dataset', default=dataset)
feat_size = 64
parser.add_argument('--feat_size', default=feat_size,
                    choices=[16, 32, 64, 128, 512, 1024],
                    help='feature size for molecular fingerprints')

c = C()
parser.add_argument('--node_fe_{}'.format(c.c()), default='one_hot')

dname = dataset.split('_')[0] if '_' in dataset else dataset
natts = DATASET_CONFIG[dname]['natts']
eatts = DATASET_CONFIG[dname]['eatts']


parser.add_argument('--node_feats', default=natts)
parser.add_argument('--edge_feats', default=eatts)
hyper_eatts = ['etype'] if 'drugcombo' in dataset else []
parser.add_argument('--hyper_eatts', default=hyper_eatts)

use_interaction_edge_attrs = False
different_edge_type_aggr = True

assert not (different_edge_type_aggr and use_interaction_edge_attrs)

if 'drugcombo' not in dataset:
    use_interaction_edge_attrs = different_edge_type_aggr = False
parser.add_argument('--use_hyper_edge_attrs', default=use_interaction_edge_attrs,
                    help='whether to use edge attribtues in the interaction '
                         'graph as part of message passing')
parser.add_argument('--different_edge_type_aggr', default=different_edge_type_aggr,
                    help='aggregate the messages across each edge type in the '
                         'interaction graph')


"""
Model. Pt1
"""

model = 'higher_level_gnn'                  # DECAGON
# model = 'lower_level_gnn'                 # LL-GNN
# model = 'lower_level_gmn'                 # MHCADDI
# model = 'lower_level_gnn_higher_level'    # BI-GNN
# model = 'higher_level_init_to_pred'       # FP-PRED

lower_level_layers = False
high_level_layers = False

if 'higher_level' in model:
    high_level_layers = True
if 'lower_level' in model:
    lower_level_layers = True
if 'our_model' in model or 'our_best_model' in model:
    lower_level_layers = True
    high_level_layers = True

node_model = None
edge_model = None

# CHANGE THIS WHEN INCORPORATING MULTIPLE EDGE TYPES
if use_interaction_edge_attrs:
    node_model = 'gat_concat'  # choices: gcn_concat, gat_concat, mlp_concat
    edge_model = 'mlp_concat'  # choices: mlp_concat
elif different_edge_type_aggr:
    edge_model = 'none'
    # choices: gat_multi_edge_aggr, gcn_multi_edge_aggr
    node_model = 'gat_multi_edge_aggr'

# OTHERWISE CHANGE THIS FOR DIFFERENT GNN TYPE
lower_level_gnn_type = 'gat'  # choices: gcn, gat, gin
higher_level_gnn_type = 'gat'  # choices: gcn, gat, gin

# CHANGE THIS FOR NUMBER OF LL AND HL LAYERS
lower_level_num_layers = 5 if 'gmn' not in model else 3
higher_level_num_layers = 3



"""
Sampling
"""
validation_window_size = 15

sampler = 'random_sampler'
use_sampled_subgraph = False
if sampler == 'random_sampler':
    sample_induced = False if lower_level_layers and not high_level_layers else True  #if gmn used in models and not new drug discovery
    sample_induced = False
    parser.add_argument('--sample_induced', default=sample_induced)

elif sampler == 'neighbor_sampler':
    parser.add_argument('--num_neighbors_sample', default=5)
    use_sampled_subgraph = False

parser.add_argument('--use_sampled_subgraph', default=use_sampled_subgraph)
parser.add_argument('--sampler', default=sampler, 
                    choices=['neighbor_sampler', 'random_sampler',
                             'everything_sampler'])
parser.add_argument('--negative_sample', default=True)
parser.add_argument('--num_negative_samples', default=1)
parser.add_argument('--enforce_negative_sampling', default=True)
parser.add_argument('--enforce_sampling_amongst_same_graphs', default=True)

"""
Validation
"""
parser.add_argument('--validation_window_size', default=validation_window_size)
parser.add_argument('--validation_metric', default='ROC-AUC',
                    choices=['ROC-AUC', 'PR-AUC', 'AVG-PR'])
# iters_per_validation = -1
iters_per_validation = 100 if not debug else 5
parser.add_argument('--iters_per_validation', default=iters_per_validation) # if -1 then based on epochs

use_best_val_model_for_inference = True
parser.add_argument('--use_best_val_model_for_inference', default=use_best_val_model_for_inference)

"""
Evaluation.
"""
save_final_node_embeddings = True
parser.add_argument('--save_final_node_embeddings', default=save_final_node_embeddings)


batch_size = 128 if 'gmn' in model else 64
batch_size = batch_size if 'small' not in dataset else 14


assert not(dataset == 'ddi_small_drugbank' and batch_size > 20)


cross_val = True
parser.add_argument('--cross_val', default=cross_val)

if cross_val:
    num_folds = 4
    # tvt_ratio = [0.3, 0.2, 0.5]
    # tvt_ratio = [0.7, 0.1, 0.2]
    tvt_ratio = [0.7, 0.05, 0.25]
    # tvt_ratio = [0.5, 0.25, 0.25]
    # tvt_ratio = [0.3, 0.45, 0.25]
    # tvt_ratio = [0.1, 0.65, 0.25]
    assert isclose((tvt_ratio[0] + tvt_ratio[1]), 1.0 - 1.0/num_folds)  #test set is the size of the fold
    parser.add_argument('--num_folds', default=num_folds)
    parser.add_argument('--run_only_on_fold', default=-1, help='if -1 run on all folds, otherwise run on the specified fold')
else:
    # tvt_ratio = [0.7, 0.1, 0.2]
    tvt_ratio = [0.5, 0.1, 0.4]
    # tvt_ratio = [0.3, 0.6, 0.1]
    # tvt_ratio = [0.1, 0.8, 0.1]

parser.add_argument('--tvt_ratio', default=tvt_ratio)

# max_eval_pairs = None
max_eval_pairs = 2000
parser.add_argument('--max_eval_pairs', default=max_eval_pairs)

parser.add_argument('--ensure_train_connectivity', default=False)

eval_performance_by_degrees = [0, 1, 2, 3, 4, 5, 10, 20, 30]
parser.add_argument('--eval_performance_by_degree', default=eval_performance_by_degrees)

parser.add_argument('--eval_per_node', default=False)

parser.add_argument('--ap_at_k', default=50)
multi_label = True if 'drugcombo' in dataset else False
parser.add_argument('--multi_label', default=multi_label)

multi_class_pred = True if 'drugcombo' in dataset else False
parser.add_argument('--multi_class_pred', default=multi_class_pred)

"""
Model Pt2
"""

pair_interaction = False if 'our_model' not in model else True
parser.add_argument('--pair_interaction', default=pair_interaction) # whether the models has lower level pair interaction
parser.add_argument('--model', default=model)

c = C()

D_lower = 64  # dim of lower level gnn
D_higher = 64  # dim of higher level gnn

# node_aggr = 'sum'
# node_aggr = 'avg_pool'
# node_aggr = 'deepsets'
# node_aggr = 'gmn_aggr'
# node_aggr = 'multi_scale'
# node_aggr = 'combine'

# see layers_aggregation for each node_aggr method
if lower_level_layers and high_level_layers:
    node_aggr = 'multi_scale'
elif lower_level_layers and not high_level_layers:
    node_aggr = 'multi_scale' if 'gmn' in model else 'multi_scale'
else:
    node_aggr = 'avg_pool'
style = 'avg_pool'

bn = True
# whether the batch contains unique graphs across the batch
batch_unique_graphs = False if model == 'lower_level_gmn' else True
parser.add_argument('--batch_unique_graphs', default=batch_unique_graphs)

agg_bn = False
gnn_normalize = False

in_channels_lower_level = in_channels_graph_feats = in_channels_init_embeddings = latent_channels = out_channels = None

aggregator_append = 'Pairs' if (not batch_unique_graphs and model == 'lower_level_gmn') or use_sampled_subgraph else ""

if lower_level_layers and high_level_layers:
    if node_aggr != 'combine':
        init_embds = 'model_init'
    else:
        init_embds = 'one_hot_init_and_model_init'
elif lower_level_layers:
    init_embds = 'no_init'
elif high_level_layers:
    # CHANGE THIS FOR DIFFERENT HL INIT
    # choices: ones_init, rand_init, one_hot_init,
    # graph_feats (ECFP fingerprints)
    init_embds = 'rand_init'

parser.add_argument('--init_embds', default=init_embds,
                    choices=['graph_feats', 'rand_init', 'model_init',
                             'ones_init', 'no_init', 'one_hot_init'])

assert init_embds == 'model_init' if use_sampled_subgraph else True
assert lower_level_layers if 'model_init' in init_embds else True
assert 'gmn' not in model if 'model_init' in init_embds else True
assert 'model_init' not in model if not lower_level_layers and high_level_layers else True

parser.add_argument('--lower_level_layers', default=lower_level_layers)
parser.add_argument('--higher_level_layers', default=high_level_layers)

D_init = feat_size
parser.add_argument('--d_init', default=D_init)


if lower_level_layers and 'gnn' in model:
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type={},output_dim={},act=relu,bn={},normalize={}'\
        .format(lower_level_gnn_type, D_lower, bn, gnn_normalize)
    parser.add_argument(n, default=s)

    for i in range(lower_level_num_layers-1):
        n = '--layer_{}'.format(c.c())
        act = 'relu' if i != lower_level_num_layers - 2 else 'identity'
        s = 'NodeEmbedding:type={},input_dim={},output_dim={},' \
            'act={},bn={},normalize={}'. \
            format(lower_level_gnn_type, D_lower, D_lower, act, bn, gnn_normalize)
        parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    if node_aggr == 'avg_pool':
        s = 'NodeAggregation{}:style=avg_pool'.format(aggregator_append)
    elif node_aggr == 'deepsets':
        s = 'NodeAggregation:style=deepsets,in_dim={},' \
            'out_dim={},num_mlp_layers={}'.format(D_lower, D_lower, 2)
    elif node_aggr == 'gmn_aggr':
        s = 'NodeAggregation:style=gmn_aggr,' \
            'in_dim={},out_dim={}'.format(D_lower, D_lower)
    elif node_aggr == 'multi_scale':
        s = 'NodeAggregation:style={},concat_multi_scale={},' \
            'in_dim={},out_dim={}'.format(style, True, D_lower, D_lower)
        D_lower = D_lower * lower_level_num_layers
    else:
        raise NotImplementedError

    parser.add_argument(n, default=s)

elif lower_level_layers and 'gmn' in model:
    f_node = 'MLP'
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type={},output_dim={},act=relu,bn={},normalize={}'\
        .format(lower_level_gnn_type, D_lower, bn, gnn_normalize)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},output_dim={},act=relu,bn={},f_node={}'.format(D_lower, D_lower, bn, f_node)
    parser.add_argument(n, default=s)

    for i in range(lower_level_num_layers-1):
        n = '--layer_{}'.format(c.c())
        act = 'relu' if i != lower_level_num_layers - 2 else 'identity'
        s = 'NodeEmbedding:type={},input_dim={},output_dim={},' \
            'act={},bn={},normalize={}'. \
            format(lower_level_gnn_type, D_lower, D_lower, act, bn, gnn_normalize)
        parser.add_argument(n, default=s)

        n = '--layer_{}'.format(c.c())
        s = 'GMNPropagator:input_dim={},output_dim={},act=relu,bn={},f_node={}'.format(D_lower, D_lower, bn, f_node)
        parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())

    if node_aggr == 'avg_pool':
        s = 'NodeAggregation{}:style=avg_pool'.format(aggregator_append)
    elif node_aggr == 'deepsets':
        s = 'NodeAggregation:style=deepsets,in_dim={},' \
            'out_dim={},num_mlp_layers={}'.format(D_lower, D_lower, 2)
    elif node_aggr == 'gmn_aggr':
        s = 'NodeAggregation:style=gmn_aggr,' \
            'in_dim={},out_dim={}'.format(D_lower, D_lower)
    elif node_aggr == 'multi_scale':
        s = 'NodeAggregation:style={},concat_multi_scale={},' \
            'in_dim={},out_dim={}'.format(style, True, D_lower, D_lower)
        D_lower = D_lower * lower_level_num_layers
    else:
        raise NotImplementedError

    parser.add_argument(n, default=s)


if lower_level_layers and 'model_init' in init_embds:
    parser.add_argument('--last_lower_lyr_num', default=c.t() - 1)

if high_level_layers:
    n = '--layer_{}'.format(c.c())
    s = 'LoadInteractionLayer'
    parser.add_argument(n, default=s)

    if model != 'higher_level_init_to_pred':
        n = '--layer_{}'.format(c.c())
        if not lower_level_layers:
            if use_interaction_edge_attrs or different_edge_type_aggr:
                s = 'MetaLayer:output_dim={},act=relu,higher_level={},' \
                    'edge_model={},node_model={}'.format(
                    D_higher, True, edge_model, node_model)
            else:
                s = 'NodeEmbedding:type={},output_dim={},act=relu,bn={},' \
                    'higher_level={},normalize={}'.format(
                    higher_level_gnn_type, D_higher, bn, True, gnn_normalize)
        else:
            if use_interaction_edge_attrs or different_edge_type_aggr:
                s = 'MetaLayer:input_dim={},output_dim={},act=relu,' \
                    'higher_level={},edge_model={},node_model={}'.format(
                    D_lower, D_higher, True, edge_model, node_model)
            else:
                s = 'NodeEmbedding:type={},input_dim={},output_dim={},' \
                    'act=relu,bn={},higher_level={},normalize={}'.format(
                    higher_level_gnn_type, D_lower, D_higher, bn, True, gnn_normalize)

        parser.add_argument(n, default=s)
        higher_level_num_layers -= 1

        for i in range(higher_level_num_layers):
            n = '--layer_{}'.format(c.c())
            act = 'relu' if i != higher_level_num_layers - 1 else 'identity'
            if use_interaction_edge_attrs or different_edge_type_aggr:
                s = 'MetaLayer:input_dim={},output_dim={},act={},' \
                    'higher_level={},edge_model={},node_model={}'.format(
                    D_higher, D_higher, act, True, edge_model, node_model)
            else:
                s = 'NodeEmbedding:type={},input_dim={},output_dim={},' \
                    'act={},bn={},higher_level={},normalize={}'. \
                    format(higher_level_gnn_type, D_higher, D_higher, act, bn, True, gnn_normalize)
            parser.add_argument(n, default=s)

if not high_level_layers:
    D_higher = D_lower

n = '--layer_{}'.format(c.c())
link_predictor_type = 'mlp_concat'
s = 'LinkPredictor:type={},multi_label_pred={},' \
    'mlp_dim={},batch_unique_graphs={}'.\
    format(link_predictor_type, multi_class_pred, D_higher, batch_unique_graphs)
parser.add_argument(n, default=s)
parser.add_argument('--link_predictor_type', default=link_predictor_type)

n = '--layer_{}'.format(c.c())
loss_type = 'BCE' if not multi_class_pred else 'CE'
s = 'Loss:type={}'.format(loss_type)
parser.add_argument(n, default=s)

parser.add_argument('--layer_num', type=int, default=c.t())

"""
Optimization.
"""

lr = 1e-3
parser.add_argument('--lr', type=float, default=lr)


device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)

num_iters = 10000 if 'small' not in dataset else 100
num_iters = 10 if debug else num_iters
parser.add_argument('--num_iters', type=int, default=num_iters)

print_every_iters = 5
parser.add_argument('--print_every_iters', type=int, default=print_every_iters)

save_model = False
parser.add_argument('--save_model', type=bool, default=save_model)
load_model = None
parser.add_argument('--load_model', default=load_model)
parser.add_argument('--batch_size', type=int, default=batch_size)

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()

COMET_EXPERIMENT = None
if FLAGS.use_comet_ml:
    hyper_params = vars(FLAGS)
    COMET_EXPERIMENT = Experiment(api_key=COMET_ML_API_KEY,
                                  project_name=COMET_PROJECT_NAME,
                                  auto_metric_logging=False)
    COMET_EXPERIMENT.log_parameters(hyper_params)
    print('Experiment url, ', COMET_EXPERIMENT.url)
    COMET_EXPERIMENT.add_tag(FLAGS.dataset)
    COMET_EXPERIMENT.add_tag(FLAGS.model)
    if 'drugcombo' in FLAGS.dataset:
        COMET_EXPERIMENT.add_tag(FLAGS.hypergraph)


