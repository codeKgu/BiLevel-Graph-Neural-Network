from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import random
import torch
from torch.utils.data import DataLoader


def generic_bfs_edges(G, source, batch_size,
                      neighbors=None, depth_limit=None):
    """
    Reference: https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/traversal/breadth_first_search.html
    added batch_size to control size of bfs subgraph
    """
    visited = {source}
    edges = []
    if depth_limit is None:
        depth_limit = len(G)
    queue = deque([(source, depth_limit, neighbors(source))])
    while queue:
        parent, depth_now, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                edges.append((parent, child))
                visited.add(child)
                if len(visited) == batch_size:
                    break
                if depth_now > 1:
                    queue.append((child, depth_now - 1, neighbors(child)))
        except StopIteration:
            queue.popleft()
    return visited, edges


class BaseSampler(ABC):
    def __init__(self, data, batch_size):
        self.id_map = data.dataset.id_map
        self.gs_map = data.dataset.gs_map
        self.interaction_combo_graph = data.dataset.interaction_combo_nxgraph
        self.batch_size = batch_size
        self.num_nodes = self.interaction_combo_graph.number_of_nodes()
        self.nodes_visited_counter = np.zeros(self.num_nodes)

    @abstractmethod
    def sample_next_training_batch(self):
        pass


class EverythingSampler(BaseSampler):
    """
    Sample all the pairs in the data at once
    """
    def __init__(self, data):
        super().__init__(data, 0)
        self.batch_size = len(data.dataset.train_pairs)
        self.data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.data_iterable = iter(self.data_loader)

    def sample_next_training_batch(self):
        try:
            sampled_pairs = next(self.data_iterable)
        except StopIteration:
            self.data_iterable = iter(self.data_loader)
            sampled_pairs = next(self.data_iterable)
        batch_gids = sampled_pairs.cpu().detach().numpy()
        sampled_gids = np.unique(batch_gids)
        return batch_gids, sampled_gids, None


class NeighborSampler(BaseSampler):
    """
    Sample pairs based on a sampled subgraph
    """
    def __init__(self, data, neighbor_size, batch_size):
        super().__init__(data, batch_size)
        self.neighbor_size = neighbor_size #float from 0 to 1 or int

    def neighbors_sampler(self, node):
        neighbors = list(self.interaction_combo_graph.neighbors(node))
        random.shuffle(neighbors)
        num_sample = self.neighbor_size if type(self.neighbor_size) == int else int(
            len(neighbors) * self.neighbor_size)
        neighbors = set(neighbors[:num_sample])
        return iter(neighbors)

    def sample_next_training_batch(self):
        sample_nodes, bfs_edges = self.recurse_sample_batch_size()
        subgraph = self.interaction_combo_graph.subgraph(sample_nodes).copy()
        batch_gids = np.asarray(list(map(lambda edge: (self.id_map[edge[0]], self.id_map[edge[1]]), subgraph.edges)))
        return batch_gids, [self.id_map[node] for node in sample_nodes], subgraph

    def recurse_sample_batch_size(self):
        sample_nodes = set()
        sample_edges = []
        while len(sample_nodes) < self.batch_size:
            batch_size = self.batch_size - len(sample_nodes)
            rand_node = random.randint(0, self.num_nodes - 1)
            while rand_node in sample_nodes or (np.any(self.nodes_visited_counter == 0) and self.nodes_visited_counter[rand_node] > 0):
                rand_node = random.randint(0, self.num_nodes - 1)
            self.nodes_visited_counter[rand_node] += 1
            bfs_nodes, bfs_edges = generic_bfs_edges(self.interaction_combo_graph, rand_node, batch_size,
                                                     neighbors=self.neighbors_sampler)
            sample_nodes = sample_nodes.union(bfs_nodes)
            sample_edges.append(bfs_edges)
        return sample_nodes, sample_edges


class RandomSampler(BaseSampler):
    """
    Randomly sampler pairs and take all induced pairs
    """
    def __init__(self, data, batch_size, sample_induced=False):
        super().__init__(data, batch_size)
        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.data_iterable = iter(self.data_loader)
        self.sample_induced = sample_induced

    def sample_next_training_batch(self):
        try:
            sampled_pairs = next(self.data_iterable)
        except StopIteration:
            self.data_iterable = iter(self.data_loader)
            sampled_pairs = next(self.data_iterable)
        if self.sample_induced:
            return self.sample_induced_pairs(sampled_pairs)
        else:
            batch_gids = sampled_pairs.cpu().detach().numpy()
            sampled_gids = np.unique(batch_gids)
            return batch_gids, sampled_gids, None

    def sample_induced_pairs(self, given_links):
        unique_gids = torch.unique(given_links).cpu().detach().numpy()
        ids = [self.gs_map[gid] for gid in unique_gids]
        subgraph = self.interaction_combo_graph.subgraph(ids)
        batch_gids = np.asarray(list(map(lambda edge: (self.id_map[edge[0]], self.id_map[edge[1]]), subgraph.edges)))
        sampled_gids = np.unique(batch_gids)
        unique_ids = np.unique(list(subgraph.edges))
        for uid in unique_ids:
            self.nodes_visited_counter[uid] += 1
        return batch_gids, sampled_gids, None


class RandomGraphSampler(BaseSampler):
    def __init__(self, data, batch_size):
        super().__init__(data, batch_size)
        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.data_iterable = iter(self.data_loader)
