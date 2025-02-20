import torch as th
import numpy as np
import dgl


def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    # Get device from input graph
    device = graph.device

    # Create new graph on the same device
    ng = dgl.graph([], device=device)  # <--- Fix here
    ng.add_nodes(graph.number_of_nodes())

    # Edge masking on correct device
    edge_mask = mask_edge(graph, edge_mask_rate)
    src, dst = graph.edges()
    nsrc = src[edge_mask].to(device)
    ndst = dst[edge_mask].to(device)
    ng.add_edges(nsrc, ndst)

    # Feature masking
    feat = drop_feature(x, feat_drop_rate)
    return ng, feat


def drop_feature(x, drop_prob):
    drop_mask = th.empty(
        (x.size(1),),
        dtype=th.float32,
        device=x.device  # <--- Ensure mask matches feature device
    ).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def mask_edge(graph, mask_prob):
    E = graph.number_of_edges()
    mask_rates = th.FloatTensor(
        np.ones(E) * mask_prob
    ).to(graph.device)  # <--- Create tensor on graph's device
    masks = th.bernoulli(1 - mask_rates)
    return masks.bool()