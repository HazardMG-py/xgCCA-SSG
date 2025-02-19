# xgcca-ssg_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

##########################
# 1. Gumbel Subspace (optional)
##########################
class GumbelSubspaceSelector(nn.Module):
    """
    Differentiable subspace selection for feature dims using Gumbel-Softmax.
    """
    def __init__(self, dim, tau=1.0):
        super(GumbelSubspaceSelector, self).__init__()
        self.logits = nn.Parameter(torch.zeros(dim))
        self.tau = tau

    def forward(self):
        """
        Returns a vector in [0,1]^dim, approximating binary selection.
        """
        g_noise = -torch.log(-torch.log(torch.rand_like(self.logits).clamp_min(1e-9)))
        y = (self.logits + g_noise) / self.tau
        return torch.sigmoid(y)

##########################
# 2. BicliqueAttentionLayer
##########################
class BicliqueAttentionLayer(nn.Module):
    """
    A custom GNN layer that multiplies the input by a 'mask'
    of selected feature dims, then applies attention-based edge weighting.
    """
    def __init__(self, in_dim, out_dim):
        super(BicliqueAttentionLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_param = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_uniform_(self.attn_param, gain=1.414)

    def edge_attention(self, edges):
        score = torch.matmul(edges.src['h'], self.attn_param)  # (batch, 1)
        return {'score': F.leaky_relu(score.squeeze(-1))}

    def forward(self, graph, feat, mask=None):
        if mask is None:
            mask = torch.ones(feat.shape[1], device=feat.device)
        # mask out columns in feat
        feat_masked = feat * mask.unsqueeze(0)
        h = self.linear(feat_masked)

        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.edge_attention)
            alpha = dgl.softmax(graph.edata['score'], graph.edges()[1])
            graph.edata['alpha'] = alpha
            graph.update_all(
                dgl.function.u_mul_e('h', 'alpha', 'm'),
                dgl.function.sum('m', 'h_new'))
            h_new = graph.ndata['h_new']
            return F.relu(h_new)

##########################
# 3. BicliqueGCN
##########################
class BicliqueGCN(nn.Module):
    """
    GCN with first & last GraphConv,
    optional middle layers as BicliqueAttentionLayer.
    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super(BicliqueGCN, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        if n_layers == 1:
            # single layer
            self.layers.append(GraphConv(in_dim, out_dim, norm='both'))
        else:
            # first layer
            self.layers.append(GraphConv(in_dim, hid_dim, norm='both'))
            # middle layers
            for _ in range(n_layers - 2):
                self.layers.append(BicliqueAttentionLayer(hid_dim, hid_dim))
            # last layer
            self.layers.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, mask=None):
        h = x
        for layer in self.layers:
            if isinstance(layer, BicliqueAttentionLayer):
                h = layer(graph, h, mask)
            else:
                # standard GraphConv
                h = layer(graph, h)
                h = F.relu(h)
        return h

##########################
# 4. xgCCA-SSG
##########################
class xgCCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers,
                 use_gumbel=False, tau=1.0):
        super(xgCCA_SSG, self).__init__()
        self.gcn = BicliqueGCN(in_dim, hid_dim, out_dim, n_layers)
        self.use_gumbel = use_gumbel
        if use_gumbel:
            self.selector = GumbelSubspaceSelector(out_dim, tau)

    def forward(self, g1, x1, g2, x2):
        # subspace mask
        if self.use_gumbel:
            mask = self.selector()
        else:
            mask = None

        h1 = self.gcn(g1, x1, mask)
        h2 = self.gcn(g2, x2, mask)

        # feature-wise normalization
        z1 = (h1 - h1.mean(0)) / (h1.std(0) + 1e-6)
        z2 = (h2 - h2.mean(0)) / (h2.std(0) + 1e-6)

        return z1, z2

    @torch.no_grad()
    def get_embedding(self, g, x):
        if self.use_gumbel:
            mask = self.selector()
        else:
            mask = None
        out = self.gcn(g, x, mask)
        return out.detach()

##########################
# 5. CCA-style Loss
##########################
def cca_loss(z1, z2, lambd=1e-3):
    """
    invariance term: negative diagonal of cross-correlation
    decorrelation term: (c1 - I)^2 + (c2 - I)^2
    """
    N = z1.size(0)
    c = (z1.T @ z2) / N     # cross-correlation
    c1 = (z1.T @ z1) / N
    c2 = (z2.T @ z2) / N

    loss_inv = -torch.diagonal(c).sum()

    d = c.size(0)
    I = torch.eye(d, device=z1.device)
    loss_dec1 = (c1 - I).pow(2).sum()
    loss_dec2 = (c2 - I).pow(2).sum()

    return loss_inv + lambd * (loss_dec1 + loss_dec2)
