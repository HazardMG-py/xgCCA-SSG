# dataset.py
import torch as th
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset

def load_data(name='cora'):
    """
    Loads a graph dataset using DGL's standard dataset classes.
    Returns:
      graph: DGLGraph
      features: FloatTensor [num_nodes, num_features]
      labels: LongTensor [num_nodes]
      num_classes: int
      train_idx, val_idx, test_idx: LongTensor of node indices
    """
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
        raise ValueError(f"Unknown dataset name: {name}")

    graph = dataset[0]

    # For citation networks, we have built-in train/val/test masks.
    if name in ['cora', 'citeseer', 'pubmed']:
        train_mask = graph.ndata.pop('train_mask')
        val_mask   = graph.ndata.pop('val_mask')
        test_mask  = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx   = th.nonzero(val_mask,   as_tuple=False).squeeze()
        test_idx  = th.nonzero(test_mask,  as_tuple=False).squeeze()

    else:
        # For co-purchase/co-author networks, create 10%/10%/80% splits at random
        N = graph.number_of_nodes()
        idx = np.arange(N)
        np.random.shuffle(idx)
        train_ratio, val_ratio = 0.1, 0.1
        train_num = int(N * train_ratio)
        val_num   = int(N * (train_ratio + val_ratio))

        train_idx = th.tensor(idx[:train_num])
        val_idx   = th.tensor(idx[train_num:val_num])
        test_idx  = th.tensor(idx[val_num:])

    features   = graph.ndata.pop('feat')
    labels     = graph.ndata.pop('label')
    num_classes = dataset.num_classes

    return graph, features, labels, num_classes, train_idx, val_idx, test_idx
