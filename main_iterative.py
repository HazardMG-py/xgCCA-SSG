import argparse
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import CCA_SSG, LogReg
from aug import random_aug
from dataset import load
from graph_group import extract_feature_groups

parser = argparse.ArgumentParser(description='Iterative Graph-Guided CCA-SSG (IG-SSL)')
parser.add_argument('--dataname', type=str, default='cora', help='Dataset name (e.g., cora)')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs per iteration')
parser.add_argument('--iterations', type=int, default=3, help='Number of iterative stages')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate for CCA-SSG')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay for CCA-SSG')
parser.add_argument('--gamma', type=float, default=1e-3, help='Graph regularization weight')
parser.add_argument('--lambd', type=float, default=1e-3, help='Trade-off ratio for decorrelation loss')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')
parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio')
parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio')
parser.add_argument('--group_thresh', type=float, default=0.5, help='Correlation threshold for grouping features')
args = parser.parse_args()

device = 'cuda' if th.cuda.is_available() else 'cpu'

# Load dataset
graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
in_dim = feat.shape[1]

# Initialize the CCA-SSG model
model = CCA_SSG(in_dim, 512, 512, n_layers=args.n_layers, use_mlp=args.use_mlp)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

print("Starting Iterative Training Process...")
for iteration in range(args.iterations):
    print(f"\nIteration {iteration + 1}/{args.iterations}")

    # Stage 2: Self-supervised training with augmented loss
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        # Generate two augmented views using random edge and feature dropout
        graph1, feat1 = random_aug(graph, feat, feat_drop_rate=args.dfr, edge_mask_rate=args.der)
        graph2, feat2 = random_aug(graph, feat, feat_drop_rate=args.dfr, edge_mask_rate=args.der)
        # Add self-loops and send graphs to device
        graph1 = graph1.add_self_loop().to(device)
        graph2 = graph2.add_self_loop().to(device)
        feat1 = feat1.to(device)
        feat2 = feat2.to(device)

        # Forward pass: Get embeddings from two views
        z1, z2 = model(graph1, feat1, graph2, feat2)

        # Compute correlation matrices (using number of nodes N)
        N = graph.number_of_nodes()
        c = th.mm(z1.t(), z2) / N
        c1 = th.mm(z1.t(), z1) / N
        c2 = th.mm(z2.t(), z2) / N

        # Invariance loss: maximize the diagonal of c (align views)
        loss_inv = -th.diagonal(c).sum()
        # Decorrelation loss: push c1 and c2 towards identity matrix
        iden = th.eye(c.shape[0]).to(device)
        loss_dec = (iden - c1).pow(2).sum() + (iden - c2).pow(2).sum()

        # Graph regularization loss: encourage neighbors (per original graph) to have similar embeddings.
        # Use the average of the two views as the final embedding.
        embeds = (z1 + z2) / 2
        src, dst = graph.edges()
        loss_graph = th.mean((embeds[src] - embeds[dst]).pow(2).sum(dim=1))

        # Total loss: invariance + decorrelation + graph regularization
        loss = loss_inv + args.lambd * loss_dec + args.gamma * loss_graph
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Iteration {iteration + 1}, Epoch {epoch:03d}: Loss = {loss.item():.4f}")

    # End of self-supervised training iteration.
    # Stage 1: Extract graph-based feature groups from updated embeddings.
    model.eval()
    with th.no_grad():
        full_graph = graph.remove_self_loop().add_self_loop().to(device)
        feat = feat.to(device)
        embeddings = model.get_embedding(full_graph, feat)  # [N, D]

    # Extract feature groups using correlation thresholding.
    groups, corr_matrix = extract_feature_groups(embeddings, threshold=args.group_thresh)
    print(f"Iteration {iteration + 1}: Extracted {len(groups)} feature groups.")
    print("Feature groups (list of feature indices):", groups)

    # (Optional: Update loss weighting or further processing based on groups before next iteration.)

# Final Evaluation: Obtain final embeddings and perform linear evaluation.
print("\n=== Final Evaluation ===")
model.eval()
with th.no_grad():
    graph_eval = graph.remove_self_loop().add_self_loop().to(device)
    feat = feat.to(device)
    final_embeds = model.get_embedding(graph_eval, feat)

# Example: Linear evaluation using logistic regression.
logreg = LogReg(final_embeds.shape[1], num_class).to(device)
opt_logreg = optim.Adam(logreg.parameters(), lr=1e-2, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

train_embs = final_embeds[train_idx]
val_embs = final_embeds[val_idx]
test_embs = final_embeds[test_idx]

train_labels = labels[train_idx].to(device)
val_labels = labels[val_idx].to(device)
test_labels = labels[test_idx].to(device)

best_val_acc = 0
for epoch in range(2000):
    logreg.train()
    opt_logreg.zero_grad()
    logits = logreg(train_embs)
    loss = loss_fn(logits, train_labels)
    loss.backward()
    opt_logreg.step()

    logreg.eval()
    with th.no_grad():
        val_logits = logreg(val_embs)
        test_logits = logreg(test_embs)
        val_preds = th.argmax(val_logits, dim=1)
        test_preds = th.argmax(test_logits, dim=1)
        val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
        test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    if epoch % 100 == 0:
        print(f"LogReg Epoch {epoch:04d}: Loss = {loss.item():.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")

print("Final Linear Evaluation Test Accuracy:", best_test_acc.item())
