import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def _filter(data, node_idx):
    """
    presumably data_n_id and new_n_id are sorted

    """
    new_data = Data()
    N = data.num_nodes
    E = data.edge_index.size(1)
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                       value=data.edge_attr, sparse_sizes=(N, N))
    new_adj, edge_idx = adj.saint_subgraph(node_idx)
    row, col, value = new_adj.coo()
    
    for key, item in data:
        if item.size(0) == N:
            new_data[key] = item[node_idx]
        elif item.size(0) == E:
            new_data[key] = item[edge_idx]
        else:
            new_data[key] = item
    
    new_data.edge_index = torch.stack([row, col], dim=0)
    new_data.num_nodes = len(node_idx)
    new_data.edge_attr = value
    return new_data