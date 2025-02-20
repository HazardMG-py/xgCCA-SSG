import torch
import numpy as np


def extract_feature_groups(embeddings, threshold=0.5):
    """
    Extract groups of correlated features from the learned embeddings.

    Parameters:
      embeddings (Tensor): Shape [N, D] where N is number of nodes and D is feature dimension.
      threshold (float): Absolute correlation threshold to group features.

    Returns:
      groups (list of lists): Each sub-list contains indices of features that are highly correlated.
      corr_matrix (ndarray): The D x D correlation matrix.
    """
    emb_mean = embeddings.mean(dim=0, keepdim=True)
    emb_std = embeddings.std(dim=0, keepdim=True) + 1e-6
    emb_norm = (embeddings - emb_mean) / emb_std  # [N, D]

    # Compute correlation matrix: D x D
    corr_matrix = torch.mm(emb_norm.t(), emb_norm) / embeddings.shape[0]
    corr_matrix_np = corr_matrix.cpu().detach().numpy()

    D = embeddings.shape[1]
    groups = []
    visited = set()
    for i in range(D):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, D):
            if j in visited:
                continue
            if abs(corr_matrix_np[i, j]) >= threshold:
                group.append(j)
                visited.add(j)
        groups.append(group)
    return groups, corr_matrix_np
