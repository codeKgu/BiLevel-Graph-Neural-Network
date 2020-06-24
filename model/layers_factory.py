import torch.nn as nn

from config import FLAGS
from model.layers import NodeEmbedding, Loss
from model.layers_aggregation import (
    NodeAggregation, NodeAggregationPairs,
    GMNAggregatorPairs
)
from model.layers_gmn import GMNPropagator
from model.layers_link_pred import LinkPred
from model.layers_load_interaction_graph import LoadInteractionGraph
from model.layers_meta import MetaLayerWrapper

def create_layers(model, pattern, num_layers):
    layers = nn.ModuleList()
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = vars(FLAGS)['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = '='.join(ssp[1:])  # could have '=' in layer_info
        if name in layer_ctors:
            layers.append(layer_ctors[name](layer_info, model, i,
                                            layers, num_layers))
        else:
            raise ValueError('Unknown layer {}'.format(name))
    return layers


def create_node_embedding_layer(lf, model, layer_id, layers, *unused):
    _check_spec([4, 5, 6, 7], lf, 'NodeEmbedding')
    input_dim, higher_level = get_input_dim_higher_level(lf, NodeEmbedding,
                                                         layers, layer_id,
                                                         model)

    return NodeEmbedding(
        type=lf['type'],
        in_dim=input_dim,
        out_dim=int(lf['output_dim']),
        act=lf['act'],
        bn=_parse_as_bool(lf['bn']),
        normalize=_parse_as_bool(lf["normalize"]),
        higher_level=higher_level,
    )


def create_meta_wrapper_layer(lf, model, layer_id, layers, *unused):
    _check_spec([5, 6], lf, 'MetaLayerWrapper')
    input_dim, higher_level = get_input_dim_higher_level(lf, MetaLayerWrapper,
                                                         layers, layer_id,
                                                         model)
    edge_dim = model.num_hyper_edge_feat

    return MetaLayerWrapper(
        input_dim=input_dim,
        edge_dim=edge_dim,
        output_dim=int(lf['output_dim']),
        edge_model=lf['edge_model'],
        node_model=lf['node_model'],
        higher_level=higher_level,
        num_edge_types=model.num_hyper_edge_feat,
        act=lf['act']
    )


def get_input_dim_higher_level(lf, lyr_class, layers, layer_id, model):

    input_dim = lf.get('input_dim')
    higher_level = lf.get('higher_level')
    higher_level = _parse_as_bool(higher_level) if higher_level else False
    if input_dim is None:
        if lyr_class in [type(layer) for layer in layers]:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))

        if higher_level:
            input_dim = model.interaction_num_node_feat
        else:
            input_dim = model.num_node_feat
    else:
        input_dim = int(input_dim)

    return input_dim, higher_level


def create_load_interaction_graph_layer(lf, *unused):
    _check_spec([0], lf, 'LoadInteractionGraph')
    return LoadInteractionGraph()


def create_node_aggregation_layer(lf, model, layer_id, layers, num_layers,
                                  *unused):
    _check_spec([1, 3, 4], lf, "NodeAggregation")
    is_last_layer = False
    # if last layer always followed by link predictor and loss
    if (layer_id + 2) == num_layers:
        is_last_layer = True
    in_dim = lf.get('in_dim')
    in_dim = int(in_dim) if in_dim is not None else None
    out_dim = lf.get('out_dim')
    out_dim = int(out_dim) if out_dim is not None else None
    num_mlp_layers = lf.get('num_mlp_layers')
    num_mlp_layers = int(num_mlp_layers) if num_mlp_layers is not None else None
    concat_multi_scale = lf.get('concat_multi_scale')
    concat_multi_scale = _parse_as_bool(concat_multi_scale) \
        if concat_multi_scale is not None else False

    return NodeAggregation(
        style=lf["style"],
        is_last_layer=is_last_layer,
        concat_multi_scale=concat_multi_scale,
        in_dim=in_dim,
        out_dim=out_dim,
        num_mlp_layers=num_mlp_layers
    )


def create_node_aggregation_pairs_layer(lf, *unused):
    _check_spec([1], lf, "NodeAggregationPairs")
    return NodeAggregationPairs(style=lf["style"])


def create_link_pred_layer(lf, model, *unused):
    _check_spec([3, 4, 5], lf, "LinkPred")
    weight_dim = lf.get('weight_dim')
    weight_dim = int(weight_dim) if weight_dim else weight_dim
    mlp_dim = lf.get('mlp_dim')
    mlp_dim = int(mlp_dim) if mlp_dim else mlp_dim

    multi_label_pred = lf['multi_label_pred']
    multi_label_pred = _parse_as_bool(multi_label_pred) if multi_label_pred else False
    return LinkPred(
        type=lf["type"],
        mlp_dim=mlp_dim,
        weight_dim=weight_dim,
        batch_unique_graphs=_parse_as_bool(lf["batch_unique_graphs"]),
        multi_label_pred=multi_label_pred,
        num_labels=model.num_labels + 1)


def create_loss_layer(lf, *unused):
    _check_spec([1], lf, 'Loss')
    return Loss(
        type=lf['type'])


def create_gmn_propagator_layer(lf, model, layer_id, layers, *unused):
    _check_spec([3, 4, 5], lf, 'GMNPropagator')
    input_dim = lf.get('input_dim')
    if not input_dim:
        if GMNPropagator in [type(layer) for layer in layers]:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.num_node_feat
    f_node = lf.get('f_node')
    if not f_node:
        f_node = 'MLP'

    return GMNPropagator(
        input_dim=int(input_dim),
        output_dim=int(lf['output_dim']),
        f_node=f_node,
    )


def create_gmn_aggregator_pairs_layer(lf, *unused):
    _check_spec([2], lf, 'GMNAggregatorPairs')
    return GMNAggregatorPairs(
        input_dim=int(lf['input_dim']),
        output_dim=int(lf['output_dim'])
    )

"""
Register the constructor caller function here.
"""
layer_ctors = {
    'NodeEmbedding': create_node_embedding_layer,
    'NodeAggregation': create_node_aggregation_layer,
    'NodeAggregationPairs': create_node_aggregation_pairs_layer,
    'Loss': create_loss_layer,
    'GMNPropagator': create_gmn_propagator_layer,
    'GMNAggregatorPairs': create_gmn_aggregator_pairs_layer,
    'LinkPredictor': create_link_pred_layer,
    'MetaLayer': create_meta_wrapper_layer,
    'LoadInteractionLayer': create_load_interaction_graph_layer

}


def _check_spec(allowed_nums, lf, ln):
    if len(lf) not in allowed_nums:
        raise ValueError('{} layer must have {} specs NOT {} {}'.
                         format(ln, allowed_nums, len(lf), lf))


def _parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))


def _parse_as_int_list(il):
    rtn = []
    for x in il.split('_'):
        x = int(x)
        rtn.append(x)
    return rtn