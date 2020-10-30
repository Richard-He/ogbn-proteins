import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_sparse import SparseTensor
import logging
from typing import List, Optional, Tuple, NamedTuple
from loguru import logger
import numpy as np
from mylogger import Logger

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer,pruner=None):
    model.train()

    optimizer.zero_grad()
    if pruner != None:
        edge_index = pruner.edge_index.cuda()
    else:
        edge_index = data.edge_index
    out = model(data.x, edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator,pruner=None):
    model.eval()
    if pruner != None:
        edge_index = pruner.edge_index.cuda()
    else:
        edge_index = data.edge_index
    out = model(data.x, edge_index)
    loss = F.nll_loss(out, data.y.squeeze(), reduction='none').cpu()
    if pruner != None:
        pruner.update_loss(loss)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

class Pruner(object):
    def __init__(self, edge_index, split_idx, prune_set='train', ratio=0.9):
        self.N = N = int(edge_index.max() + 1)
        self.E = edge_index.size(1)
        self.edge_index = edge_index

        self.adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           value=torch.arange(edge_index.size(1)), sparse_sizes=(N, N),
                           is_sorted=False)
        if prune_set == 'train':
            self.train_idx = split_idx['train']
        else:
            self.train_idx = torch.cat([split_idx['train'], split_idx['valid']])
        subadj, _ = self.adj.saint_subgraph(self.train_idx)
        # subadj = self.adj.to_dense()[self.train_idx][:,self.train_idx].view(-1)
        _,_,e_idx = subadj.coo()
        self.train_e_idx = e_idx.squeeze().long()
        self.train_edge_index = self.edge_index[:, self.train_e_idx] 
        self.rest_e_idx = torch.LongTensor(list(set(range(self.E))  - set(self.train_e_idx.tolist())))
        self.ratio = ratio
        self.loss = torch.zeros(N)
        self.times = 0
    
    def prune(self,naive=False):
        i = self.times        
        diff_loss = torch.abs(self.loss[self.train_edge_index[0]] - self.loss[self.train_edge_index[1]])
        # print(diff_loss.nonzero().size())
        # print(int(len(diff_loss)*ratio))
        _, mask1 = torch.topk(diff_loss, int(len(diff_loss)*self.ratio), largest=False)
       
        newE =self.train_edge_index.size(1)
        mask2 = torch.randperm(newE)[:int(newE*self.ratio)]
        torch.save(self.train_edge_index, f'./savept/pre_prune_edges_{self.ratio ** i:.4f}.pt')

        torch.save(mask1, f'./savept/smart_mask_prune_edges_{self.ratio ** i:.4f}.pt')
        torch.save(mask2, f'./savept/naive_mask_prune_edges_{self.ratio ** i:.4f}.pt')
        torch.save(self.loss, f'./savept/loss_{self.ratio ** i:.4f}.pt')
        self.times +=1
        if naive == False:
            mask = mask1
        else:
            mask = mask2
        self.train_e_idx = self.train_e_idx[mask]
        self.train_edge_index = self.train_edge_index[:, mask]
        self.edge_index = self.edge_index[:,torch.cat([self.train_e_idx, self.rest_e_idx])]
        self.train_e_idx = torch.arange(self.train_e_idx.size(0))
        self.rest_e_idx = torch.arange(self.train_e_idx.size(0),self.train_e_idx.size(0) + self.rest_e_idx.size(0))
        self.E = self.edge_index.size(1)
        print(f'****************trainE : {self.train_e_idx.size(0)} ,restE:{self.rest_e_idx.size(0)}, totalE:{self.E}')
    
    def update_loss(self, loss):
        self.loss = loss


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--prune_set', type=str, default='train')
    parser.add_argument('--ratio', type=float, default=0.95)
    parser.add_argument('--times', type=int, default=20)
    parser.add_argument('--prune_epoch', type=int, default=301)
    parser.add_argument('--reset_param',type=bool, default=False)
    parser.add_argument('--naive', type=bool, default=False)
    parser.add_argument('--data_dir',type=str,default='./data/')
    args = parser.parse_args()
    
    log_name = f'log/arxivtest_{args.prune_set}_{args.ratio}_{args.epochs}_{args.prune_epoch}_{args.times}.log'
    logger.add(log_name)
    logger.info('logname: {}'.format(log_name))
    logger.info(args)

    # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = torch.device(device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',root=args.data_dir,
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger1 = Logger(args.runs, args)
    row, col, val = data.adj_t.coo()
    N = int(row.max()+1)
    row = torch.cat([torch.arange(0, N).cuda(), row],dim=0)
    col = torch.cat([torch.arange(0, N).cuda(), col],dim=0)
    edge_index = torch.cat([row,col]).view(2, -1)
    data.edge_index = edge_index
    # print(data.edge_index)
    pruner = Pruner(edge_index.cpu(), split_idx, prune_set=args.prune_set, ratio=args.ratio)
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger1.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                logger.info(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}% Test: {100 * test_acc:.2f}%')

        logger1.print_statistics(ratio=1)
        logger1.flush()
        for i in range(1, args.times+1):
            pruner.prune(naive=args.naive)
            if args.reset_param == True:
                model.reset_parameters()
            for epoch in range(1, 1 + args.prune_epoch):
                loss = train(model, data, train_idx, optimizer,pruner=pruner)
                result = test(model, data, split_idx, evaluator,pruner=pruner)
                logger1.add_result(run, result)
                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    logger.info(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}% Test: {100 * test_acc:.2f}%')

            logger1.print_statistics(ratio=args.ratio ** i)
            logger1.flush()

if __name__ == "__main__":
    main()