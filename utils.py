import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from inspect import getargspec
import logging

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


class BraceMessage(object):
    def __init__(self, fmt, args, kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return str(self.fmt).format(*self.args, **self.kwargs)

class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger):
        self.logger = logger

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, log_kwargs = self.process(msg, kwargs)
            self.logger._log(level, BraceMessage(msg, args, kwargs), (), 
                    **log_kwargs)

    def process(self, msg, kwargs):
        return msg, {key: kwargs[key] 
                for key in getargspec(self.logger._log).args[1:] if key in kwargs}