import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from gcca import select_subspace


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)
        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act_fn(x)
        x = self.layer2(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)
        return x


class BicliqueAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Additive attention instead of dot product
        self.attn_mlp = nn.Linear(2 * out_dim, 1)  # For concatenated [src||dst] features
        self.mask_weights = nn.Parameter(torch.ones(in_dim))  # Learnable soft mask

    def forward(self, graph, feat, biclique_mask=None):
        # Soft mask application (learned sigmoid weights)
        feat_masked = feat * torch.sigmoid(self.mask_weights).unsqueeze(0)

        h = self.linear(feat_masked)
        with graph.local_scope():
            graph.ndata['h'] = h
            # Compute attention scores using concatenated features
            graph.apply_edges(
                lambda edges: {'score': F.leaky_relu(
                    self.attn_mlp(torch.cat([edges.src['h'], edges.dst['h']], dim=1))
                )}
            )
            # Normalize attention scores
            graph.edata['a'] = dgl.softmax(graph.edata['score'], graph.edges()[1])
            # Message passing with attention weights
            graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_new'))
            return graph.ndata['h_new']


class BicliqueGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, norm='both'))
        for _ in range(n_layers - 2):
            self.layers.append(BicliqueAttentionLayer(hid_dim, hid_dim))
        self.layers.append(GraphConv(hid_dim, out_dim, norm='both'))  # No ReLU in last layer

    def forward(self, graph, x, biclique_mask=None):
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BicliqueAttentionLayer):
                h = F.relu(layer(graph, h, biclique_mask))
            else:
                h = layer(graph, h)
        return h


class XgCCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super(XgCCA_SSG, self).__init__()
        self.backbone = BicliqueGCN(in_dim, hid_dim, out_dim, n_layers)
        self.ema_R = None
        self.subspace_update_freq = 10
        self.register_buffer('subspace_mask', torch.ones(out_dim))
        self.subspace = list(range(out_dim))

    def update_subspace(self, current_R):
        if self.ema_R is None:
            self.ema_R = current_R
        else:
            self.ema_R = 0.9 * self.ema_R + 0.1 * current_R
        new_subspace = select_subspace(self.ema_R)
        self.subspace_mask.zero_()
        self.subspace_mask[new_subspace] = 1.0
        self.subspace = new_subspace

    def forward(self, graph1, feat1, graph2, feat2, mask=None, epoch=None):
        if epoch is not None and (epoch % self.subspace_update_freq == 0):
            with torch.no_grad():
                h1 = self.backbone(graph1, feat1)  # Graph first, then features.
                h2 = self.backbone(graph2, feat2)
                current_R = torch.mm(h1.t(), h2) / h1.shape[0]
                self.update_subspace(current_R)
        if mask is None:
            mask = self.subspace_mask
        # Swap the order: graph first, then features.
        h1 = self.backbone(graph1, feat1, mask)
        h2 = self.backbone(graph2, feat2, mask)
        z1 = F.normalize(h1, p=2, dim=1)
        z2 = F.normalize(h2, p=2, dim=1)
        return z1, z2

    def get_embedding(self, graph, feat, mask=None):
        out = self.backbone(graph, feat, mask)
        return F.normalize(out, p=2, dim=1).detach()

