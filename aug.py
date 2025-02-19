# aug.py
import torch as th
import dgl
import numpy as np

def random_aug(graph, x, feat_drop_rate=0.2, edge_drop_rate=0.2):
    """
    Returns an augmented graph and feature matrix:
      - drop edges randomly
      - drop feature columns randomly
    """
    edge_mask = mask_edge(graph, edge_drop_rate)
    x_aug = drop_feature(x, feat_drop_rate)

    # Build new graph
    aug_graph = dgl.graph([])
    aug_graph.add_nodes(graph.number_of_nodes())

    src, dst = graph.edges()
    aug_src = src[edge_mask]
    aug_dst = dst[edge_mask]
    aug_graph.add_edges(aug_src, aug_dst)

    return aug_graph, x_aug

def drop_feature(x, drop_prob):
    """
    Zero out a fraction (drop_prob) of feature columns for all nodes.
    """
    D = x.shape[1]
    mask = th.rand(D, device=x.device) < drop_prob  # True means we zero that column
    x_dropped = x.clone()
    x_dropped[:, mask] = 0
    return x_dropped

def mask_edge(graph, mask_prob):
    """
    For each edge, keep it with probability (1 - mask_prob).
    Returns a boolean index mask.
    """
    E = graph.number_of_edges()
    keep_prob = 1.0 - mask_prob
    mask = th.rand(E) < keep_prob
    return mask.nonzero().squeeze()
