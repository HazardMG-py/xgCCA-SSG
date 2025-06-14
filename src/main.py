import argparse
import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import NeighborSampler, DataLoader
from ogb.linkproppred import Evaluator
import numpy as np
import random
import os
from datetime import datetime
from config import get_args
from dataset import load_data
from xgcca_model import xgCCA_SSG, cca_loss
from aug import random_aug
from utils import setup_seed, save_checkpoint, log_results


def train(args, model, graph, feat, train_edges, device, log_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    sampler = NeighborSampler([15, 10])  # Sample 15 neighbors in layer 1, 10 in layer 2
    dataloader = DataLoader(graph, graph.nodes(), sampler, batch_size=1024, shuffle=True, device=device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_nodes, _, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            batch_feat = feat[batch_nodes].to(device)
            optimizer.zero_grad()
            g1, x1 = random_aug(blocks[-1], batch_feat, args.dfr, args.der)
            g2, x2 = random_aug(blocks[-1], batch_feat, args.dfr, args.der)
            g1, g2 = g1.add_self_loop().to(device), g2.add_self_loop().to(device)
            x1, x2 = x1.to(device), x2.to(device)

            z1, z2, sparsity_loss = model(g1, x1, g2, x2)
            loss = cca_loss(z1, z2, args.lambd) + args.sparsity_lambda * sparsity_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        selected_dims = (model.selector(hard=True) > 0.5).sum().item()
        print(f"Epoch={epoch:03d}, Loss={total_loss / len(dataloader):.4f}, Selected Dim={selected_dims}")
        log_results(log_dir,
                    f"Epoch={epoch:03d}, Loss={total_loss / len(dataloader):.4f}, Selected Dim={selected_dims}")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, log_dir, epoch)

    return model


def evaluate(model, graph, feat, train_edges, val_edges, test_edges, args, device, log_dir):
    model.eval()
    graph = graph.remove_self_loop().add_self_loop().to(device)
    with torch.no_grad():
        embeds = model.get_embedding(graph, feat.to(device))
    evaluator = Evaluator(name='ogbl-biokg')

    def score_edges(edges):
        src, dst = edges[0], edges[1]
        scores = F.cosine_similarity(embeds[src], embeds[dst])
        return scores

    val_scores = score_edges(val_edges.to(device))
    test_scores = score_edges(test_edges.to(device))
    val_neg_edges = val_edges[:, torch.randperm(val_edges.size(1))]
    test_neg_edges = test_edges[:, torch.randperm(test_edges.size(1))]

    val_result = evaluator.eval({
        'y_pred_pos': val_scores.cpu(),
        'y_pred_neg': score_edges(val_neg_edges.to(device)).cpu()
    })
    test_result = evaluator.eval({
        'y_pred_pos': test_scores.cpu(),
        'y_pred_neg': score_edges(test_neg_edges.to(device)).cpu()
    })

    results = {
        'Val MRR': val_result['mrr'],
        'Test MRR': test_result['mrr'],
        'Val Hits@10': val_result['hits@10'],
        'Test Hits@10': test_result['hits@10']
    }
    print(f"Val MRR: {results['Val MRR']:.4f}, Test MRR: {results['Test MRR']:.4f}")
    print(f"Val Hits@10: {results['Val Hits@10']:.4f}, Test Hits@10: {results['Test Hits@10']:.4f}")
    log_results(log_dir, str(results))
    return results['Test MRR']


def main():
    args = get_args()
    setup_seed(42)  # Ensure reproducibility
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 and torch.cuda.is_available() else 'cpu')

    log_dir = f"logs/{args.dataname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    log_results(log_dir, str(args))

    graph, feat, _, _, train_edges, val_edges, test_edges = load_data(args.dataname)
    graph, feat = graph.to(device), feat.to(device)
    train_edges, val_edges, test_edges = train_edges.to(device), val_edges.to(device), test_edges.to(device)

    args.in_dim = feat.shape[1]
    model = xgCCA_SSG(args.in_dim, args.hid_dim, args.out_dim, args.n_layers, args.tau,
                      args.sparsity_lambda, args.num_heads).to(device)

    model = train(args, model, graph, feat, train_edges, device, log_dir)
    test_mrr = evaluate(model, graph, feat, train_edges, val_edges, test_edges, args, device, log_dir)
    return test_mrr


if __name__ == '__main__':
    main()