import numpy as np
import torch as th
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, AmazonCoBuyPhotoDataset, \
    AmazonCoBuyComputerDataset, CoauthorCSDataset, CoauthorPhysicsDataset


def load(name):
    name = name.lower()
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()
    else:
        raise NotImplementedError("Dataset not implemented")

    graph = dataset[0]
    if name in ['cora', 'citeseer', 'pubmed']:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    else:
        N = graph.number_of_nodes()
        train_ratio = 0.1
        val_ratio = 0.1
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)
        train_idx = th.tensor(idx[:train_num])
        val_idx = th.tensor(idx[train_num:val_num])
        test_idx = th.tensor(idx[val_num:])

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx
