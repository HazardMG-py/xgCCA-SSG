import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


class LogReg(nn.Module):
    """Logistic Regression classifier for evaluation"""

    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    """MLP backbone alternative (not recommended for graph data)"""

    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super().__init__()
        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.bn = nn.BatchNorm1d(nhid) if use_bn else None
        self.act = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.bn: x = self.bn(x)
        return self.layer2(self.act(x))


class BicliqueAttentionLayer(nn.Module):
    """GNN layer with biclique-aware attention mechanism"""

    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Multi-head attention parameters
        self.attn_weights = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim, 1))
        self.linear = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)

        # Subspace integration parameters
        self.subspace_selector = DifferentiableSubspaceSelector(in_dim)
        nn.init.xavier_uniform_(self.attn_weights)

    def forward(self, graph, feat, temp=0.5):
        # Generate subspace mask
        mask = self.subspace_selector(temp)  # [in_dim]

        # Apply mask and transform features
        feat_masked = feat * mask.unsqueeze(0)  # [N, in_dim]
        h = self.linear(feat_masked).view(-1, self.num_heads, self.head_dim)  # [N, H, D/H]

        # Compute attention scores
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.edge_attention)
            graph.edata['a'] = dgl.softmax(graph.edata['score'], graph.edges()[1])

            # Multi-head aggregation
            graph.update_all(self.message_func, self.reduce_func)
            return graph.ndata['h_new'].view(-1, self.num_heads * self.head_dim)

    def edge_attention(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)  # [E, H, 2D/H]
        score = (h @ self.attn_weights).squeeze(-1)  # [E, H]
        return {'score': F.leaky_relu(score)}

    def message_func(self, edges):
        return {'m': edges.src['h'] * edges.data['a'].unsqueeze(-1)}

    def reduce_func(self, nodes):
        return {'h_new': torch.sum(nodes.mailbox['m'], dim=1)}


class DifferentiableSubspaceSelector(nn.Module):
    """Gumbel-softmax based feature subspace selector"""

    def __init__(self, dim):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(dim))

    def forward(self, temp=0.5, hard=False):
        return F.gumbel_softmax(self.logits, tau=temp, hard=hard)[:, 1]  # Returns differentiable mask


class BicliqueGCN(nn.Module):
    """Biclique-aware GNN encoder"""

    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([GraphConv(in_dim, hid_dim, norm='both')])

        for _ in range(n_layers - 2):
            self.layers.append(BicliqueAttentionLayer(hid_dim, hid_dim))

        if n_layers > 1:
            self.layers.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, temp=0.5):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BicliqueAttentionLayer):
                x = F.relu(layer(graph, x, temp))
            else:
                x = layer(graph, x)
                if i != len(self.layers) - 1:  # No ReLU for last layer
                    x = F.relu(x)
        return x


class XgCCA_SSG(nn.Module):
    """Main xgCCA-SSG model with automatic subspace management"""

    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()
        self.backbone = BicliqueGCN(in_dim, hid_dim, out_dim, n_layers)
        self.ema_R = None
        self.subspace_mask = nn.Parameter(torch.ones(hid_dim), requires_grad=False)
        self.subspace_update_freq = 5
        self.temp = 1.0  # Initial Gumbel temperature

        # Statistics tracking
        self.register_buffer('subspace_sizes', torch.zeros(100))
        self.curr_step = 0

    def update_subspace(self, R):
        """EMA-based subspace tracking with Gumbel relaxation"""
        if self.ema_R is None:
            self.ema_R = R.detach()
        else:
            self.ema_R = 0.9 * self.ema_R + 0.1 * R.detach()

        # Gradually decrease Gumbel temperature
        self.temp = max(0.1, 1.0 - self.curr_step * 0.01)
        self.curr_step += 1

    def forward(self, graph1, feat1, graph2, feat2, epoch=None):
        # Update subspace every few epochs
        if epoch and (epoch % self.subspace_update_freq == 0):
            with torch.no_grad():
                h1 = self.backbone(graph1, feat1, temp=self.temp)
                h2 = self.backbone(graph2, feat2, temp=self.temp)
                self.update_subspace(h1.T @ h2 / h1.size(0))

        # Forward pass with current subspace parameters
        h1 = self.backbone(graph1, feat1, temp=self.temp)
        h2 = self.backbone(graph2, feat2, temp=self.temp)

        # Masked normalization
        z1 = (h1 - h1.mean(0)) / (h1.std(0) + 1e-6)
        z2 = (h2 - h2.mean(0)) / (h2.std(0) + 1e-6)

        return z1 * self.subspace_mask.unsqueeze(0), z2 * self.subspace_mask.unsqueeze(0)

    def get_embedding(self, graph, feat):
        """Get stabilized embeddings for downstream tasks"""
        with torch.no_grad():
            return self.backbone(graph, feat, temp=0.1)  # Use low temp for hard mask