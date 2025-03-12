import torch as th
import dgl


def random_aug(graph, x, feat_drop_rate=0.2, edge_drop_rate=0.2):
    """Augment graph and features by dropping edges/features."""
    # Ensure device consistency: create augmented graph on the same device as input features
    device = x.device

    # Edge masking (ensure mask is on the correct device)
    edge_mask = mask_edge(graph, edge_drop_rate)  # Fixed in mask_edge()

    # Feature dropping
    x_aug = drop_feature(x, feat_drop_rate)

    # Create augmented graph ON THE SAME DEVICE as input features
    aug_graph = dgl.graph([], device=device)  # <--- KEY FIX HERE
    aug_graph.add_nodes(graph.number_of_nodes())

    # Add edges (src/dst are already on correct device via graph.edges())
    src, dst = graph.edges()
    aug_graph.add_edges(src[edge_mask], dst[edge_mask])

    return aug_graph, x_aug


def drop_feature(x, drop_prob):
    """Zero out features with probability drop_prob."""
    mask = th.rand(x.shape[1], device=x.device) > drop_prob  # Use same device as x
    x_dropped = x.clone()
    x_dropped[:, ~mask] = 0
    return x_dropped


def mask_edge(graph, mask_prob):
    """Generate edge mask on the same device as input graph."""
    E = graph.number_of_edges()
    return (th.rand(E, device=graph.device) > mask_prob).nonzero().squeeze()  # Use graph's device