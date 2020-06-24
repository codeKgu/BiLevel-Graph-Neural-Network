import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def encode_edge_features(nx_graph, attr_keys=None):
    if not attr_keys:
        return None
    input_dim = 0
    results = []
    for attr in attr_keys:
        res, input_dim = one_hot_encode_edges(nx_graph, input_dim, attr)
        results.append(res)
    nx_graph.edge_attr_x = np.concatenate(results, axis=1)
    return input_dim


def one_hot_encode_edges(nx_graph, input_dim, attr):
    edge_attr = nx.get_edge_attributes(nx_graph, attr)
    inputs_set = set()
    for edge in edge_attr.values():
        inputs_set.add(edge)
    feat_idx_dic = {feat: idx for idx, feat in enumerate(sorted(inputs_set))}
    oe = OneHotEncoder(categories='auto').fit(
                np.array(sorted(feat_idx_dic.values())).reshape(-1, 1))
    temp = [feat_idx_dic[feat] for feat in
            edge_attr.values()]  # sort nids just to make sure
    input_dim += oe.transform([[0]]).shape[1]
    return oe.transform(np.array(temp).reshape(-1, 1)).toarray(), input_dim