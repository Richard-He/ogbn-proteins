import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import MessageNorm


def clones(module, N, block_num=-1):
    """Produce N identical layers."""
    if block_num <= 0:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    else:
        assert block_num >= 1 and isinstance(block_num, int)
        modules = []
        freq = N // block_num
        for idx in range(N):
            if idx % freq == 0:
                module = copy.deepcopy(module)
            modules.append(module)
        return nn.ModuleList(modules)


def elu_fea_map(x):
    return F.elu(x) + 1


class LinearMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, eps=1e-6):
        """Take in model size and number of heads."""
        super(LinearMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.eps = eps

    def forward(self, query, key, value, mask=None):
        """Implement linear and multi-headed attention"""
        if mask is not None:
            raise Exception('Do not support mask currently')
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).reshape(nbatches, -1, self.h, self.d_k)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply linear attention, where we do not calculate attention matrix explicitly
        Q = elu_fea_map(query)
        K = elu_fea_map(key)
        KV = torch.einsum('bshd, bshm -> bhdm', K, value)
        Z = 1. / (torch.einsum('bshd, bhd -> bsh', Q, K.sum(dim=1)) + self.eps)
        V = torch.einsum('bshd, bhdm, bsh -> bshm', Q, KV, Z)

        # 3) "Concat" using a view and apply a final linear.
        x = V.reshape(nbatches, self.h * self.d_k)

        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # self.a_2 = nn.Parameter(torch.ones(features))
        # self.b_2 = nn.Parameter(torch.zeros(features))
        # fit for bert optimizer
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """"Core encoder is a stack of N layers"""

    def __init__(self, layer, N, block_num=-1):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N, block_num=block_num)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def make_transformer_encoder(num_layers, hidden_size, ff_size, num_att_heads, dropout=0.1, block_num=-1):
    dcopy = copy.deepcopy
    # mh_att = MultiHeadedAttention(num_att_heads, hidden_size, dropout=dropout)
    mh_att = LinearMultiHeadedAttention(
        num_att_heads, hidden_size, dropout=dropout
    )
    pos_ff = PositionwiseFeedForward(
        hidden_size, ff_size, hidden_size, dropout=dropout
    )

    tranformer_encoder = Encoder(
        EncoderLayer(
            hidden_size, dcopy(mh_att), dcopy(pos_ff), dropout=dropout
        ),
        num_layers,
        block_num=block_num
    )

    return tranformer_encoder


class GateConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels=1, **kwargs):
        super(GateConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels

        self.linear_n = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.linear_e = nn.Parameter(torch.Tensor(edge_channels, out_channels))

        # self.linear_attn = LinearMultiHeadedAttention(h=)
        self.tfm_encoder = make_transformer_encoder(
            num_layers=2,
            hidden_size=out_channels*2,
            ff_size=out_channels*2,
            num_att_heads=8
        )

        self.msg_norm = MessageNorm(learn_scale=True)

        self.linear_msg = nn.Linear(out_channels*2, out_channels)
        self.linear_aggr = nn.Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_n)
        nn.init.xavier_uniform_(self.linear_e)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        """"""
        x = torch.matmul(x, self.linear_n)

        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(dim=-1)

        edge_weight = torch.matmul(edge_weight, self.linear_e)

        out = self.propagate(edge_index, size=size, x=x,
                             edge_weight=edge_weight)

        return x + self.msg_norm(x, out)

    def message(self, x_j, edge_weight):
        msg = torch.cat([x_j, edge_weight], dim=-1)
        msg = self.tfm_encoder(msg, mask=None)
        return self.linear_msg(msg)

    def update(self, aggr_out):
        return self.linear_aggr(aggr_out)
