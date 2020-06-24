import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.transforms import LocalDegreeProfile

from config import FLAGS
from utils.util import get_flags_with_prefix_as_list


class NodeFeatureEncoder(object):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        if node_feat_name is None:
            return
        # Go through all the graphs in the entire dataset
        # and create a set of all possible
        # labels so we can one-hot encode them.
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in
                             enumerate(sorted(inputs_set))}
        self._fit_onehotencoder()

    def _fit_onehotencoder(self):
        self.oe = OneHotEncoder(categories='auto').fit(
            np.array(sorted(self.feat_idx_dic.values())).reshape(-1, 1))

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in
                sorted(g.nodes())]  # sort nids just to make sure
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


def encode_node_features(dataset=None, pyg_single_g=None):
    if dataset:
        assert pyg_single_g is None
        input_dim = 0
    else:
        assert pyg_single_g is not None
        input_dim = pyg_single_g.x.shape[1]
    node_feat_encoders = get_flags_with_prefix_as_list('node_fe', FLAGS)
    natts = FLAGS.node_feats
    if 'one_hot' not in node_feat_encoders:
        raise ValueError('Must have one hot node feature encoder!')
    if dataset:
        if len(natts) == 0:
            # if no node feat return 1
            for g in dataset.gs:
                g.init_x = np.ones((nx.number_of_nodes(g), 1))
            return 1
        else:
            for node_feat_name in natts:
                for nfe in node_feat_encoders:
                    if nfe == 'one_hot':
                        if dataset:
                            input_dim = _one_hot_encode(dataset, input_dim,
                                                        node_feat_name)
                    elif nfe == 'local_degree_profile':
                        pass
                    else:
                        raise ValueError(
                            'Unknown node feature encoder {}'.format(nfe))
            # Handle ldf.
            if 'local_degree_profile' in node_feat_encoders:
                input_dim += 5  # only update dim for model weight initilization

        if input_dim <= 0:
            raise ValueError('Must have at least one node feature encoder '
                             'so that input_dim > 0')

        return dataset, input_dim
    else:
        if 'local_degree_profile' in node_feat_encoders:
            pyg_single_g = LocalDegreeProfile()(pyg_single_g)
        return pyg_single_g, input_dim


def _one_hot_encode(dataset, input_dim, node_feat_name):
    gs = [g.get_nxgraph() for g in dataset.gs]

    nfe = NodeFeatureEncoder(gs, node_feat_name)
    for g in gs:
        x = nfe.encode(g)
        if hasattr(g, "init_x"):
            g.init_x = np.concatenate((g.init_x, x), axis=1)
        else:
            g.init_x = x  # assign the initial features
    input_dim += nfe.input_dim()
    return input_dim
