import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import MessagePassing


class GateConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels=1, **kwargs):
        super(GateConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.weight_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        self.u = nn.Parameter(torch.Tensor(out_channels, out_channels))
        self.v = nn.Parameter(torch.Tensor(out_channels, out_channels))

        self.layer_norm = nn.LayerNorm(normalized_shape=out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        x = torch.matmul(x, self.weight_n)

        if len(edge_weight.shape) == 1:
            edge_weight = edge_weight.unsqueeze(dim=-1)

        edge_weight = torch.matmul(edge_weight, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, edge_index_i, x_i, x_j, edge_weight):
        x_i = torch.matmul(x_i, self.u)
        x_j = torch.matmul(x_j, self.v)

        gate = torch.sigmoid(x_i * x_j * edge_weight)

        return x_j * gate

    def update(self, aggr_out, x):
        aggr_out = torch.matmul(x, self.u) + aggr_out

        aggr_out = self.layer_norm(aggr_out)

        return x + aggr_out
