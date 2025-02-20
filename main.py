import argparse
import numpy as np
import torch as th
import torch.nn as nn
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

from xgcca_model import XgCCA_SSG, LogReg
from aug import random_aug
from dataset import load
from gcca import select_subspace

def evaluate(logreg, train_embs, val_embs, test_embs, train_labels, val_labels, test_labels, device, max_epochs=2000):
    logreg = logreg.to(device)
    optimizer = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(max_epochs):
        logreg.train()
        optimizer.zero_grad()
        logits = logreg(train_embs)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Validation
        logreg.eval()
        with th.no_grad():
            val_logits = logreg(val_embs)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

            test_logits = logreg(test_embs)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == test_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

    return best_test_acc

def main(args):
    # Device setup
    device = th.device(f'cuda:{args.gpu}' if args.gpu != -1 and th.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(args.dataname)
    in_dim = feat.shape[1]
    graph = graph.to(device).add_self_loop()
    feat = feat.to(device)
    labels = labels.to(device)

    # Initialize model
    model = XgCCA_SSG(
        in_dim=in_dim,
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        n_layers=args.n_layers
    ).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)

    # Training loop
    best_loss = float('inf')
    for epoch in tqdm(range(args.epochs), desc="Pretraining"):
        model.train()
        optimizer.zero_grad()

        # Generate augmented views
        graph1, feat1 = random_aug(graph, feat, args.dfr, args.der)
        graph2, feat2 = random_aug(graph, feat, args.dfr, args.der)
        graph1, graph2 = graph1.to(device), graph2.to(device)
        feat1, feat2 = feat1.to(device), feat2.to(device)

        # Forward pass (model handles subspace updates internally)
        z1, z2 = model(graph1, feat1, graph2, feat2, epoch=epoch)

        # Compute masked losses
        R = z1.T @ z2 / z1.shape[0]
        iden = th.eye(z1.size(1), device=device)

        # Invariance loss: Align biclique features
        loss_inv = -th.diagonal(R).sum()

        # Decorrelation loss: Minimize off-diagonal terms
        loss_dec = (z1.T @ z1 - iden).pow(2).sum() + (z2.T @ z2 - iden).pow(2).sum()

        # Total loss
        loss = loss_inv + args.lambd * loss_dec
        loss.backward()
        optimizer.step()

        tqdm.write(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Subspace Size: {model.subspace_mask.sum().item():.0f}")

    # Evaluation
    model.eval()
    with th.no_grad():
        embeds = model.get_embedding(graph, feat)
    test_acc = evaluate(
        LogReg(embeds.size(1), num_class).to(device),
        embeds[train_idx], embeds[val_idx], embeds[test_idx],
        labels[train_idx], labels[val_idx], labels[test_idx],
        device
    )
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='xgCCA-SSG: Biclique-Aware Self-Supervised Graph Representation Learning'
    )
    # ... (keep existing arguments unchanged)
    args = parser.parse_args()
    main(args)