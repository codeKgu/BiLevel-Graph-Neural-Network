from config import FLAGS
from model.layers_util import convert_nx_to_pyg_graph, create_edge_index

import torch.nn as nn


class LoadInteractionGraph(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_data, model):
        assert(hasattr(batch_data, "interaction_combo_nxgraph"))
        batch_data.merge_higher_level['merge'] = \
            convert_nx_to_pyg_graph(batch_data.interaction_combo_nxgraph)

        if FLAGS.different_edge_type_aggr:
            batch_data.merge_higher_level['edges'] = {}
            for k, v in batch_data.dataset.interaction_nxgraphs.items():
                batch_data.merge_higher_level['edges'][k] = create_edge_index(v)[0]

        return batch_data.merge_higher_level['merge'].x