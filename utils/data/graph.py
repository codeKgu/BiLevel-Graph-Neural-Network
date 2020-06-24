class Graph(object):
    def __init__(self, nxgraph):
        if 'gid' not in nxgraph.graph or type(nxgraph.graph['gid']) is not int \
                or nxgraph.graph['gid'] < 0:
            raise ValueError('Graph ID must be non-negative integers {}'.
                             format(nxgraph.graph.get('gid')))
        self.nxgraph = nxgraph

    def gid(self):
        return self.nxgraph.graph['gid']

    def get_nxgraph(self):
        return self.nxgraph


class GraphPair(object):
    def __init__(self, true_label, g1=None, g2=None, edge_types=None):
        if type(true_label) is not int or true_label < 0:
            raise ValueError('true_label must be a non-negative int {}'.
                             format(true_label))
        self.true_label = true_label
        self.edge_types = edge_types
        self.link_pred = None
        self.g1 = g1
        self.g2 = g2

    def get_pair_id(self):
        assert self.g1 is not None and self.g2 is not None, "need to assign g1 and g2"
        return self.g1.gid(), self.g2.gid()

    def assign_g1_g2(self, g1, g2):
        self.g1 = g1
        self.g2 = g2

    def assign_link_pred(self, link_pred):
        if link_pred.shape[0] == 1:
            self.link_pred = link_pred.item()
        else:
            self.link_pred = link_pred.detach().cpu().numpy()

    def get_link_pred(self):
        if self.link_pred is None:
            raise ValueError('Must call assign_link_pred before calling this')
        return self.link_pred

    def shrink_space_for_save(self):
        import torch
        self.__dict__.pop('g1', None)
        self.__dict__.pop('g2', None)
        if type(self.link_pred) == torch.Tensor:
            self.link_pred = self.link_pred.cpu().detach().numpy()