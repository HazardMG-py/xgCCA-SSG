# main.py
import argparse
import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl

from dataset import load_data
from aug import random_aug
from xgcca_model import XgCCA_SSG, cca_loss

######################################################
# A simple logistic regression for evaluation
######################################################
class LogReg(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser(description="xgCCA-SSG Example")
    parser.add_argument('--dataname', type=str, default='cora')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3, help='LR for self-supervised')
    parser.add_argument('--wd', type=float, default=0,    help='Weight decay for self-supervised')

    # Data augmentation
    parser.add_argument('--edge_drop', type=float, default=0.2)
    parser.add_argument('--feat_drop', type=float, default=0.2)

    # GNN dims
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--lambda_', type=float, default=1e-3, help='decorrelation trade-off')

    # linear eval
    parser.add_argument('--lr2', type=float, default=1e-2, help='LR for linear eval')
    parser.add_argument('--wd2', type=float, default=1e-4, help='WD for linear eval')
    parser.add_argument('--eval_epochs', type=int, default=2000)
    args = parser.parse_args()

    device = th.device('cuda:%d'%args.gpu if (args.gpu>=0 and th.cuda.is_available()) else 'cpu')

    ######################################################
    # 1) Load data
    ######################################################
    graph, features, labels, num_classes, train_idx, val_idx, test_idx = load_data(args.dataname)
    graph = graph.remove_self_loop().add_self_loop()
    features = features.float()

    print("NumNodes:", graph.number_of_nodes())
    print("NumEdges:", graph.number_of_edges())
    print("NumFeats:", features.shape[1])
    print("NumClasses:", num_classes)
    print("NumTrainingSamples:", len(train_idx))
    print("NumValidationSamples:", len(val_idx))
    print("NumTestSamples:", len(test_idx))

    graph = graph.to(device)
    features = features.to(device)
    labels   = labels.to(device)

    ######################################################
    # 2) Build model
    ######################################################
    model = XgCCA_SSG(
        in_dim=features.shape[1],
        hid_dim=args.hid_dim,
        out_dim=args.out_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    ######################################################
    # 3) Self-supervised training
    ######################################################
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # create two augmented views
        g1, feat1 = random_aug(graph, features, feat_drop_rate=args.feat_drop, edge_drop_rate=args.edge_drop)
        g2, feat2 = random_aug(graph, features, feat_drop_rate=args.feat_drop, edge_drop_rate=args.edge_drop)
        g1 = g1.add_self_loop().to(device)
        g2 = g2.add_self_loop().to(device)

        C_masked, row_mask, col_mask, z1, z2 = model(g1, feat1, g2, feat2)
        loss = cca_loss(C_masked, lambd=args.lambda_)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            row_keep = (row_mask.detach() > 0.5).sum().item()
            col_keep = (col_mask.detach() > 0.5).sum().item()
            print(f"Epoch={epoch:03d}, Loss={loss.item():.4f}, "
                  f"SelectedRowDim={row_keep}, SelectedColDim={col_keep}")

    ######################################################
    # 4) Evaluate via linear classifier
    ######################################################
    print("=== Evaluation ===")
    model.eval()

    # Get final node embeddings with the original graph
    # (We can pass the same graph & features as both 'views',
    #  or define an identity augmentation.)
    with th.no_grad():
        # minimal identity augmentation
        ig = graph
        ifeat = features
        C_m, rm, cm, emb_all, _ = model(ig, ifeat, ig, ifeat)
        # 'emb_all' is shape [N, out_dim]

    train_embs = emb_all[train_idx]
    val_embs   = emb_all[val_idx]
    test_embs  = emb_all[test_idx]

    train_labels = labels[train_idx]
    val_labels   = labels[val_idx]
    test_labels  = labels[test_idx]

    logreg = LogReg(train_embs.shape[1], num_classes).to(device)
    opt2 = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc_at_val = 0

    for e in range(args.eval_epochs):
        logreg.train()
        opt2.zero_grad()

        logits_train = logreg(train_embs)
        loss_cls = loss_fn(logits_train, train_labels)
        loss_cls.backward()
        opt2.step()

        logreg.eval()
        with th.no_grad():
            val_logits  = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds  = th.argmax(val_logits,  dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc  = (val_preds  == val_labels).float().mean().item()
            test_acc = (test_preds == test_labels).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc_at_val = test_acc

    print(f"Best Val Acc={best_val_acc:.4f}, Test Acc at that Val={best_test_acc_at_val:.4f}")

if __name__ == '__main__':
    main()
