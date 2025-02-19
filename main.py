import argparse
import numpy as np
import torch as th
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

from xgcca_model import XgCCA_SSG, LogReg
from aug import random_aug
from dataset import load
from gcca import select_subspace

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='xgCCA-SSG: Biclique-Aware Self-Supervised Graph Representation Learning')
    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index (use -1 for CPU).')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate for xgCCA-SSG.')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate for linear evaluator.')
    parser.add_argument('--wd1', type=float, default=0, help='Weight decay for xgCCA-SSG.')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay for linear evaluator.')
    parser.add_argument('--lambd', type=float, default=1e-3, help='Trade-off parameter for decorrelation loss.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers.')
    #parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN.')
    parser.add_argument('--der', type=float, default=0.2, help='Edge drop ratio.')
    parser.add_argument('--dfr', type=float, default=0.2, help='Feature drop ratio.')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden layer dimension.')
    parser.add_argument('--out_dim', type=int, default=512, help='Output layer dimension.')
    parser.add_argument('--gcca_thresh', type=float, default=0.05, help='Threshold for subspace selection.')
    args = parser.parse_args()

    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    print(args)

    # Load dataset (e.g., Cora)
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
    in_dim = feat.shape[1]

    # Initialize xgCCA-SSG model (using our biclique-aware encoder)
    model = XgCCA_SSG(in_dim, args.hid_dim, args.out_dim, args.n_layers)
    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    N = graph.number_of_nodes()

    # Self-supervised pretraining loop
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Generate two augmented views
        graph1, feat1 = random_aug(graph, feat, args.dfr, args.der)
        graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)
        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()
        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)
        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        # Forward pass: get embeddings from both views
        z1, z2 = model(graph1, feat1, graph2, feat2)  # z1, z2: [N, D]

        # Compute cross-correlation matrix R (D x D)
        R = th.mm(z1.t(), z2) / N

        # Apply gCCA-inspired subspace selection to obtain indices
        selected_indices = select_subspace(R, threshold=args.gcca_thresh)
        if len(selected_indices) == 0:
            selected_indices = list(range(z1.shape[1]))

        # Create a biclique mask (a binary vector) for the encoder:
        biclique_mask = th.zeros(z1.shape[1], device=z1.device)
        biclique_mask[selected_indices] = 1.0

        # Optionally, you can use this mask during the forward pass of the encoder:
        # For demonstration, we recompute the embeddings with the biclique mask.
        z1_mask, z2_mask = model(graph1, feat1, graph2, feat2, biclique_mask=biclique_mask)

        # Compute cross-correlation on the selected subspace
        z1_sel = z1_mask[:, selected_indices]
        z2_sel = z2_mask[:, selected_indices]
        c_sel = th.mm(z1_sel.t(), z2_sel) / N
        loss_inv = -th.diagonal(c_sel).sum()

        # Compute decorrelation loss on the selected subspace
        c1_sel = th.mm(z1_sel.t(), z1_sel) / N
        c2_sel = th.mm(z2_sel.t(), z2_sel) / N
        iden = th.eye(c1_sel.shape[0]).to(args.device)
        loss_dec = (iden - c1_sel).pow(2).sum() + (iden - c2_sel).pow(2).sum()

        loss = loss_inv + args.lambd * loss_dec
        loss.backward()
        optimizer.step()

        print(f"Epoch={epoch:03d}, Loss={loss.item():.4f}, Selected Dim={len(selected_indices)}")

    print("=== Evaluation ===")
    # Evaluate: obtain full embeddings
    graph = graph.to(args.device)
    graph = graph.remove_self_loop().add_self_loop()
    feat = feat.to(args.device)
    embeds = model.get_embedding(graph, feat)

    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]
    labels = labels.to(args.device)
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    logreg = LogReg(train_embs.shape[1], num_class).to(args.device)
    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        loss_cls = loss_fn(logits, train_labels)
        loss_cls.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)
            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()
            test_acc = (test_preds == test_labels).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                eval_acc = test_acc
            if epoch % 100 == 0:
                print(
                    f"Epoch:{epoch}, Train Loss:{loss_cls.item():.4f}, Val Acc:{val_acc:.4f}, Test Acc:{test_acc:.4f}")

    print(f"Final Linear Evaluation Accuracy: {eval_acc:.4f}")
