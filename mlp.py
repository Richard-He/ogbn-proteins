import argparse

import torch
import torch.nn.functional as F
from torch_scatter import scatter

import numpy as np

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, x, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    train_x, train_y = x[train_idx], y_true[train_idx]
    row_idx = np.arange(train_x.size(0))
    np.random.shuffle(row_idx)
    train_x, train_y = train_x[row_idx], train_y[row_idx]
    
    batch_size = 1024
    nb_batch = (train_x.size(0) + batch_size - 1) // batch_size

    running_loss = 0.0

    for i in range(nb_batch):
        batch_x, batch_y = train_x[i * batch_size: (i + 1) * batch_size], train_y[i * batch_size: (i + 1) * batch_size]
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / nb_batch


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    y_pred = model(x)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(y_pred, y_true.float())
    print('test loss: ', loss.item())

    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    x = scatter(data.edge_attr, data.edge_index[0], dim=0,
                dim_size=data.num_nodes, reduce='mean').to('cpu')

    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)

    x = x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = MLP(x.size(-1), args.hidden_channels, 112, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, x, y_true, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()