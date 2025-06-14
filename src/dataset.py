import torch
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from torch_geometric.datasets import Reddit
import dgl
import os


def load_data(name='cora', path='data'):
    dataset_map = {
        'cora': CoraGraphDataset,
        'citeseer': CiteseerGraphDataset,
        'pubmed': PubmedGraphDataset,
        'photo': AmazonCoBuyPhotoDataset,
        'comp': AmazonCoBuyComputerDataset,
        'cs': CoauthorCSDataset,
        'physics': CoauthorPhysicsDataset,
        'arxiv': lambda: DglNodePropPredDataset(name='ogbn-arxiv'),
        'reddit': lambda: Reddit(root=os.path.join(path, 'Reddit')),
        'biokg': lambda: DglLinkPropPredDataset(name='ogbl-biokg')
    }

    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}")

    dataset = dataset_map[name]()

    if name == 'biokg':
        graph = dataset[0]
        features = graph.ndata['feat']
        labels = None
        num_classes = None
        split_edge = dataset.get_edge_split()
        train_edges = split_edge['train']['edge']
        val_edges = split_edge['valid']['edge']
        test_edges = split_edge['test']['edge']
        train_idx = torch.tensor(train_edges, dtype=torch.long).t()
        val_idx = torch.tensor(val_edges, dtype=torch.long).t()
        test_idx = torch.tensor(test_edges, dtype=torch.long).t()
    elif name == 'reddit':
        pyg_data = dataset[0]
        graph = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]), num_nodes=pyg_data.num_nodes)
        features = pyg_data.x
        labels = pyg_data.y
        num_classes = dataset.num_classes
        N = graph.number_of_nodes()
        idx = np.random.permutation(N)
        train_num, val_num = int(N * 0.1), int(N * 0.1)
        train_idx = torch.tensor(idx[:train_num])
        val_idx = torch.tensor(idx[train_num:train_num + val_num])
        test_idx = torch.tensor(idx[train_num + val_num:])
    elif name == 'arxiv':
        graph, labels = dataset[0]
        labels = labels.squeeze()
        features = graph.ndata['feat']
        num_classes = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
    else:
        graph = dataset[0]
        labels = graph.ndata.pop('label', None)
        features = graph.ndata.pop('feat', None)
        num_classes = dataset.num_classes
        if name in ['cora', 'citeseer', 'pubmed']:
            train_idx = torch.nonzero(graph.ndata.pop('train_mask'), as_tuple=False).squeeze()
            val_idx = torch.nonzero(graph.ndata.pop('val_mask'), as_tuple=False).squeeze()
            test_idx = torch.nonzero(graph.ndata.pop('test_mask'), as_tuple=False).squeeze()
        else:
            N = graph.number_of_nodes()
            idx = np.random.permutation(N)
            train_num, val_num = int(N * 0.1), int(N * 0.1)
            train_idx = torch.tensor(idx[:train_num])
            val_idx = torch.tensor(idx[train_num:train_num + val_num])
            test_idx = torch.tensor(idx[train_num + val_num:])

    if features is None or (labels is None and name != 'biokg'):
        raise KeyError("Features or labels not found in dataset")

    if name != 'biokg':
        graph.ndata['feat'] = features
        graph.ndata['label'] = labels

    return (graph, features, labels, num_classes, train_idx, val_idx, test_idx)