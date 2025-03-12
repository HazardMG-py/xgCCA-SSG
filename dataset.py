import torch as th
import numpy as np
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset


def load_data(name='cora'):
    """
    Loads a graph dataset.
    Args:
        name: Dataset name (e.g., 'cora', 'photo')
    Returns:
        graph, features, labels, num_classes, train_idx, val_idx, test_idx
    """
    dataset_map = {
        'cora': CoraGraphDataset, 'citeseer': CiteseerGraphDataset, 'pubmed': PubmedGraphDataset,
        'photo': AmazonCoBuyPhotoDataset, 'comp': AmazonCoBuyComputerDataset,
        'cs': CoauthorCSDataset, 'physics': CoauthorPhysicsDataset
    }
    if name not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}")

    dataset = dataset_map[name]()
    graph = dataset[0]

    if name in ['cora', 'citeseer', 'pubmed']:
        train_idx = th.nonzero(graph.ndata.pop('train_mask'), as_tuple=False).squeeze()
        val_idx = th.nonzero(graph.ndata.pop('val_mask'), as_tuple=False).squeeze()
        test_idx = th.nonzero(graph.ndata.pop('test_mask'), as_tuple=False).squeeze()
    else:
        N = graph.number_of_nodes()
        idx = np.random.permutation(N)
        train_num, val_num = int(N * 0.1), int(N * 0.2)
        train_idx = th.tensor(idx[:train_num])
        val_idx = th.tensor(idx[train_num:val_num])
        test_idx = th.tensor(idx[val_num:])

    return (graph, graph.ndata.pop('feat'), graph.ndata.pop('label'), dataset.num_classes,
            train_idx, val_idx, test_idx)