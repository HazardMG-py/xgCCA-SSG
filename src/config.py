import argparse

def get_args():
    parser = argparse.ArgumentParser(description='xgCCA-SSG: Biclique-Aware Self-Supervised Graph Learning')
    parser.add_argument('--dataname', type=str, default='biokg', choices=['cora', 'citeseer', 'pubmed', 'photo', 'comp', 'cs', 'physics', 'arxiv', 'reddit', 'biokg'], help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=100, help='Pretraining epochs')
    parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate for xgCCA_SSG')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate for evaluation')
    parser.add_argument('--wd1', type=float, default=0, help='Weight decay for xgCCA_SSG')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay for evaluation')
    parser.add_argument('--lambd', type=float, default=1e-3, help='Decorrelation loss weight')
    parser.add_argument('--sparsity_lambda', type=float, default=1e-3, help='Sparsity penalty weight')
    parser.add_argument('--tau', type=float, default=1.0, help='Gumbel temperature')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--der', type=float, default=0.2, help='Edge drop ratio')
    parser.add_argument('--dfr', type=float, default=0.2, help='Feature drop ratio')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--out_dim', type=int, default=512, help='Output dimension')
    args = parser.parse_args(args=[]) if 'google.colab' in str(get_ipython()) else parser.parse_args()
    args.device = f'cuda:{args.gpu}' if args.gpu != -1 and torch.cuda.is_available() else 'cpu'
    return args