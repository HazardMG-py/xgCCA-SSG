# xgcca_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

########################################
# 1) Simple GNN
########################################
class SimpleGNN(nn.Module):
    """
    A minimal two-layer GNN for demonstration.
    """
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hid_dim, norm='both')
        self.conv2 = GraphConv(hid_dim, out_dim, norm='both')

    def forward(self, g, feat):
        x = self.conv1(g, feat)
        x = F.relu(x)
        x = self.conv2(g, x)
        return x

########################################
# 2) Soft Discrete Biclique
########################################
class SoftDiscreteBicliqueSearch(nn.Module):
    """
    Approximates discrete row/column removal with a
    differentiable gating approach:
     row_mask[i] = sigmoid(scale*( mean(|C[i,:]|) - threshold + offset[i] ))
     etc.
    """
    def __init__(self, dim, threshold=0.05, scale=50.0):
        super().__init__()
        self.dim = dim
        self.base_thresh = threshold
        self.scale = scale

        # Offsets can be learnable
        self.row_offsets = nn.Parameter(torch.zeros(dim))
        self.col_offsets = nn.Parameter(torch.zeros(dim))

    def forward(self, C):
        """
        C: [D, D], cross-correlation matrix
        Return:
          row_mask, col_mask in (0,1)
        """
        row_score = C.abs().mean(dim=1)  # shape [D]
        col_score = C.abs().mean(dim=0)  # shape [D]

        row_mask = torch.sigmoid(self.scale * (row_score + self.row_offsets - self.base_thresh))
        col_mask = torch.sigmoid(self.scale * (col_score + self.col_offsets - self.base_thresh))

        return row_mask, col_mask

########################################
# 3) xgCCA_SSG model
########################################
class XgCCA_SSG(nn.Module):
    """
    Combines:
      - a GNN for each augmented view
      - a cross-correlation step
      - a soft discrete biclique subspace search
    """
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        # GNN for both views (could define 2 if you want separate weights)
        self.gnn = SimpleGNN(in_dim, hid_dim, out_dim)

        # Subspace detection
        self.subspace = SoftDiscreteBicliqueSearch(dim=out_dim, threshold=0.05, scale=50.0)

    def forward(self, g1, x1, g2, x2):
        # produce embeddings
        z1 = self.gnn(g1, x1)
        z2 = self.gnn(g2, x2)

        # cross-correlation matrix
        C = compute_correlation_matrix(z1, z2)

        # row/col masks
        row_mask, col_mask = self.subspace(C)

        # apply them
        rm = row_mask.unsqueeze(1)  # [D,1]
        cm = col_mask.unsqueeze(0)  # [1,D]
        C_masked = C * (rm * cm)    # shape [D, D]

        return C_masked, row_mask, col_mask, z1, z2

########################################
# 4) Utilities
########################################
def compute_correlation_matrix(z1, z2):
    """
    z1, z2: [N, D]
    Return cross-correlation matrix [D, D].
    Standardize each dimension, then dot-product / N
    """
    z1_std = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2_std = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
    N = z1.size(0)
    C = z1_std.T @ z2_std / N
    return C

def cca_loss(C_masked, lambd=1e-3):
    """
    Example correlation-based objective:
    invariance = - sum(diagonal(C_masked))
    decorrelation = punishing off-diagonal from identity
    """
    d = C_masked.size(0)
    diag_sum = torch.diagonal(C_masked).sum()
    loss_inv = -diag_sum

    identity = torch.eye(d, device=C_masked.device)
    # measure how far from identity
    diff = (C_masked - identity).pow(2).sum() - (C_masked.diagonal() - 1).pow(2).sum()
    loss_dec = diff

    return loss_inv + lambd*loss_dec
