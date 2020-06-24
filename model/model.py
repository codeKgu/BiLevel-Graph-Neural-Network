import torch.nn as nn

from config import FLAGS
from model.layers_factory import create_layers


class Model(nn.Module):
    def __init__(self, data, config_layer_type='layer'):
        super(Model, self).__init__()

        self.train_data = data
        self.interaction_num_node_feat = data.dataset.interaction_num_node_feat
        self.num_node_feat = data.num_node_feat
        self.num_hyper_edge_feat = data.num_hyper_edge_feat
        self.num_labels = data.dataset.num_labels
        self.layers = create_layers(self, config_layer_type,
                                    vars(FLAGS)["{}_num".format(config_layer_type)])
        self.pred_layer = self.layers[-2]  # assume pred layer is second to last layer
        self._use_layers = 'all'
        if FLAGS.lower_level_layers and FLAGS.higher_level_layers:
            self._use_layers = 'init_model'
            self.init_layers = self.layers[:FLAGS.last_lower_lyr_num]
            self.lower_layers = self.layers[:FLAGS.last_lower_lyr_num + 1]
            self.higher_level_layers = self.layers[FLAGS.last_lower_lyr_num + 1:]
        elif FLAGS.lower_level_layers:
            self.init_layers = self.layers[:-2]
            self.lower_layers = self.layers[:-2]

        assert (len(self.layers) > 0)
        self._print_layers(None, self.layers)
        self.layer_output = {}
        self.acts = None

    def forward(self, batch_data):
        # Go through each layer except the last one.
        # acts = [self._get_ins(self.layers[0])]
        md = batch_data.merge_data['merge']
        self.acts = [md.x]
        if FLAGS.lower_level_layers and FLAGS.higher_level_layers:
            if self._use_layers == 'init_layers':
                layers = self.init_layers
            elif self._use_layers == 'lower_layers':
                layers = self.lower_layers
            elif self._use_layers == 'higher_layers':
                layers = self.higher_level_layers
            elif self._use_layers == 'higher_no_eval_layers':
                layers = self.layers[FLAGS.last_lower_lyr_num + 1:-2]
        else:
            if self._use_layers == "higher_no_eval_layers":
                layers = self.layers[:-2]
            elif self._use_layers == 'lower_layers':
                layers = self.lower_layers
            elif self._use_layers == 'init_layers':
                layers = self.init_layers
            else:
                layers = self.layers
        for k, layer in enumerate(layers):
            ins = self.acts[-1]
            outs = layer(ins, batch_data, self)
            self.acts.append(outs)
        total_loss = self.acts[-1]
        return total_loss

    def store_layer_output(self, layer, output):
        self.layer_output[layer] = output

    def get_layer_output(self, layer):
        return self.layer_output[layer]  # may get KeyError/ValueError

    def _print_layers(self, branch_name, layers):
        print('Created {} layers: {}'.format(
            len(layers), ', '.join(l.__class__.__name__ for l in layers)))

    @property
    def use_layers(self):
        return self._use_layers

    @use_layers.setter
    def use_layers(self, setting):
        assert setting in ['all', 'init_layers', 'lower_layers',
                           'higher_layers', 'higher_no_eval_layers']
        self._use_layers = setting
