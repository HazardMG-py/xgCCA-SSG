import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

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
        super(BicliqueAttentionLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # Learnable attention parameter (vector)
        self.attn_param = nn.Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_uniform_(self.attn_param, gain=1.414)

    def forward(self, graph, feat, biclique_mask):
        # feat: [N, in_dim]; biclique_mask: [in_dim] (binary mask: 1 for features in robust subspace)
        # Apply the mask to features
        feat_masked = feat * biclique_mask.unsqueeze(0)  # shape: [N, in_dim]
        h = self.linear(feat_masked)  # shape: [N, out_dim]
        # Compute edge attention scores using a simple dot product with attn_param
        with graph.local_scope():
            graph.ndata['h'] = h
            # For each edge, compute attention = LeakyReLU(dot(src['h'], attn_param))
            graph.apply_edges(lambda edges: {'score': F.leaky_relu((edges.src['h'] @ self.attn_param).squeeze(-1))})
            # Normalize attention scores with softmax over incoming edges
            graph.edata['a'] = dgl.softmax(graph.edata['score'], graph.edges()[1])
            # Message passing: weighted sum of neighbor features
            graph.update_all(message_func=dgl.function.u_mul_e('h', 'a', 'm'),
                             reduce_func=dgl.function.sum('m', 'h_new'))
            h_new = graph.ndata['h_new']
            return F.relu(h_new)


class BicliqueGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super(BicliqueGCN, self).__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        # First layer: standard GraphConv
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        # Middle layers: biclique-aware layers
        for i in range(n_layers - 2):
            self.convs.append(BicliqueAttentionLayer(hid_dim, hid_dim))
        # Last layer: standard GraphConv (if n_layers > 1)
        if n_layers > 1:
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))
        else:
            self.convs[0] = GraphConv(in_dim, out_dim, norm='both')

    def forward(self, graph, x, biclique_mask=None):
        h = x
        for i in range(self.n_layers):
            if i == 0 or i == self.n_layers - 1:
                h = self.convs[i](graph, h)
                h = F.relu(h)
            else:
                # Use biclique-aware layer; if mask not provided, use an all-ones mask.
                if biclique_mask is None:
                    mask = torch.ones(h.shape[1], device=h.device)
                else:
                    mask = biclique_mask
                h = self.convs[i](graph, h, mask)
        return h


class XgCCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp=False):
        super(XgCCA_SSG, self).__init__()
        if not use_mlp:
            self.backbone = BicliqueGCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            raise NotImplementedError("MLP version not implemented for xgCCA_SSG.")

    def get_embedding(self, graph, feat, biclique_mask=None):
        out = self.backbone(graph, feat, biclique_mask)
        return out.detach()

    def forward(self, graph1, feat1, graph2, feat2, biclique_mask=None):
        h1 = self.backbone(graph1, feat1, biclique_mask)
        h2 = self.backbone(graph2, feat2, biclique_mask)
        # Normalize along each feature dimension
        z1 = (h1 - h1.mean(0)) / (h1.std(0) + 1e-6)
        z2 = (h2 - h2.mean(0)) / (h2.std(0) + 1e-6)
        return z1, z2
