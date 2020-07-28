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

        self.lin_gate = nn.Linear(out_channels*3, out_channels)

        self.layer_norm = nn.LayerNorm(normalized_shape=out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_n)
        nn.init.xavier_uniform_(self.weight_e)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        x = torch.matmul(x, self.weight_n)

        if len(edge_weight.shape) == 1:
            edge_weight = edge_weight.unsqueeze(dim=-1)

        edge_weight = torch.matmul(edge_weight, self.weight_e)

        return self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        gate = torch.cat([x_i, x_j, edge_weight], dim=-1)
        gate = torch.tanh(self.lin_gate(gate))

        return x_j * gate

    def update(self, aggr_out):
        return aggr_out
