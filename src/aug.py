import torch
import dgl

def random_aug(graph, x, feat_drop_rate=0.2, edge_drop_rate=0.2):
    device = x.device
    edge_mask = (torch.rand(graph.number_of_edges(), device=device) > edge_drop_rate).nonzero().squeeze()
    if edge_mask.numel() == 0:  # Ensure at least some edges remain
        edge_mask = torch.arange(graph.number_of_edges(), device=device)
    x_aug = x.clone()
    mask = torch.rand(x.shape[1], device=device) > feat_drop_rate
    x_aug[:, ~mask] = 0
    aug_graph = dgl.graph([], device=device)
    aug_graph.add_nodes(graph.number_of_nodes())
    src, dst = graph.edges()
    aug_graph.add_edges(src[edge_mask], dst[edge_mask])
    return aug_graph, x_aug