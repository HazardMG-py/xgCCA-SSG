import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class GumbelSubspaceSelector(nn.Module):
    def __init__(self, dim, tau=1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(dim))
        self.tau = tau

    def forward(self, hard=False):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits).clamp_min(1e-9)))
        logits = self.logits + gumbel_noise
        soft_mask = torch.sigmoid(logits / self.tau)
        if hard:
            hard_mask = (soft_mask > 0.5).float()
            return hard_mask + soft_mask - soft_mask.detach()
        return soft_mask


class BicliqueAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim_per_head = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_param = nn.Parameter(torch.FloatTensor(num_heads, self.out_dim_per_head, 1))
        nn.init.xavier_uniform_(self.attn_param, gain=1.414)

    def edge_attention(self, edges):
        h_src = edges.src['h'].view(-1, self.num_heads, self.out_dim_per_head)
        score = torch.matmul(h_src, self.attn_param).squeeze(-1)
        return {'score': F.leaky_relu(score)}

    def forward(self, graph, feat, mask=None):
        if mask is None:
            mask = torch.ones(feat.shape[1], device=feat.device)
        feat_masked = feat * mask.unsqueeze(0)
        h = self.linear(feat_masked).view(-1, self.num_heads, self.out_dim_per_head)

        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.edge_attention)
            alpha = dgl.softmax(graph.edata['score'], graph.edges()[1])
            graph.edata['alpha'] = alpha
            graph.update_all(dgl.function.u_mul_e('h', 'alpha', 'm'), dgl.function.sum('m', 'h_new'))
            return F.relu(graph.ndata['h_new'].view(-1, self.num_heads * self.out_dim_per_head))


class BicliqueGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(GraphConv(in_dim, out_dim, norm='both'))
        else:
            self.layers.append(GraphConv(in_dim, hid_dim, norm='both'))
            for _ in range(n_layers - 2):
                self.layers.append(BicliqueAttentionLayer(hid_dim, hid_dim, num_heads))
            self.layers.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, mask=None):
        h = x
        for layer in self.layers:
            if isinstance(layer, BicliqueAttentionLayer):
                h = layer(graph, h, mask)
            else:
                h = layer(graph, h)
                h = F.relu(h)
        return h


class xgCCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, tau=1.0, sparsity_lambda=1e-3, num_heads=4):
        super().__init__()
        self.gcn = BicliqueGCN(in_dim, hid_dim, out_dim, n_layers, num_heads)
        self.selector = GumbelSubspaceSelector(out_dim, tau)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, g1, x1, g2, x2):
        mask = self.selector(hard=False)
        h1 = self.gcn(g1, x1, mask)
        h2 = self.gcn(g2, x2, mask)
        z1 = (h1 - h1.mean(0)) / (h1.std(0) + 1e-6)
        z2 = (h2 - h2.mean(0)) / (h2.std(0) + 1e-6)
        sparsity_loss = torch.mean(mask)
        return z1, z2, sparsity_loss

    @torch.no_grad()
    def get_embedding(self, g, x):
        mask = self.selector(hard=True)
        return self.gcn(g, x, mask)


def cca_loss(z1, z2, lambd=1e-3):
    """CCA loss with invariance and decorrelation terms."""
    N = z1.size(0)
    c = (z1.T @ z2) / N
    c1 = (z1.T @ z1) / N
    c2 = (z2.T @ z2) / N
    loss_inv = -torch.diagonal(c).sum()
    I = torch.eye(c.size(0), device=z1.device)
    loss_dec = (c1 - I).pow(2).sum() + (c2 - I).pow(2).sum()
    return loss_inv + lambd * loss_dec


class LogReg(nn.Module):
    """Linear classifier for evaluation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)