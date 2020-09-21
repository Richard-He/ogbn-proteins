# import os.path as osp
import argparse
# import torch
import torch.nn.functional as F
# from torch.nn import Linear as Lin
# from tqdm import tqdm
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from torch_geometric.data import NeighborSampler
# from torch_geometric.nn import GATConv
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import subgraph

from gen_conv import GENConv
from torch_geometric.nn import DeepGCNLayer
from torch_geometric.data import RandomNodeSampler, NeighborSampler, GraphSAINTRandomWalkSampler
from meta_net import Record, MetaNet
from utils import _filter
from logger import Logger
# dataset = PygNodePropPredDataset('ogbn-products', 'mnt\data\ogbn_product')
# split_idx = dataset.get_idx_split()
# evaluator = Evaluator(name='ogbn-products')
# data = dataset[0]

# train_idx = split_idx['train']
# train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
#                                sizes=[10, 10, 10], batch_size=512,
#                                shuffle=True, num_workers=12)
# subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
#                                   batch_size=1024, shuffle=False,
#                                   num_workers=12)
parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--inductive', action='store_true')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--walk_length', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_steps', type=int, default=30)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_steps', type=int, default=2)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--prune_ratio', type=float, default=0.5)
args = parser.parse_args()

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        #self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='stat',
                           t=1.0, learn_t=True, num_layers=2, norm='layer', msg_norm=True)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, int(torch.max(data.y)+1))

    def forward(self, x, edge_index):
        x = self.node_encoder(x)
        #edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers[1:]:
            x = layer(x, edge_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = (dataset.num_features, 128, dataset.num_classes, num_layers=3,
#             heads=4)
# model = model.to(device)

# x = data.x.to(device)
# y = data.y.squeeze().to(device)


# def train(epoch):
#     model.train()

#     pbar = tqdm(total=train_idx.size(0))
#     pbar.set_description(f'Epoch {epoch:02d}')

#     total_loss = total_correct = 0
#     for batch_size, n_id, adjs in train_loader:
#         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
#         adjs = [adj.to(device) for adj in adjs]

#         optimizer.zero_grad()
#         out = model(x[n_id], adjs)
#         loss = F.nll_loss(out, y[n_id[:batch_size]])
#         loss.backward()
#         optimizer.step()

#         total_loss += float(loss)
#         total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
#         pbar.update(batch_size)

#     pbar.close()

#     loss = total_loss / len(train_loader)
#     approx_acc = total_correct / train_idx.size(0)

#     return loss, approx_acc


# @torch.no_grad()
# def test():
#     model.eval()

#     out = model.inference(x)

#     y_true = y.cpu().unsqueeze(-1)
#     y_pred = out.argmax(dim=-1, keepdim=True)

#     train_acc = evaluator.eval({
#         'y_true': y_true[split_idx['train']],
#         'y_pred': y_pred[split_idx['train']],
#     })['acc']
#     val_acc = evaluator.eval({
#         'y_true': y_true[split_idx['valid']],
#         'y_pred': y_pred[split_idx['valid']],
#     })['acc']
#     test_acc = evaluator.eval({
#         'y_true': y_true[split_idx['test']],
#         'y_pred': y_pred[split_idx['test']],
#     })['acc']

#     return train_acc, val_acc, test_acc


# test_accs = []
# for run in range(1, 11):
#     print('')
#     print(f'Run {run:02d}:')
#     print('')

#     model.reset_parameters()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     best_val_acc = final_test_acc = 0
#     for epoch in range(1, 101):
#         loss, acc = train(epoch)
#         print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

#         if epoch > 50 and epoch % 10 == 0:
#             train_acc, val_acc, test_acc = test()
#             print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
#                   f'Test: {test_acc:.4f}')

#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 final_test_acc = test_acc
#     test_accs.append(final_test_acc)

# test_acc = torch.tensor(test_accs)
# print('============================')
# print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')

def train(epoch, model, optimizer):
    model.train()
    meta_net.eval()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)

        record = recorder.get_record(data.n_id).to(device)
        weights = meta_net(record, data.edge_index)
        weights = weights / weights.mean()

        out = model(data.x * weights, data.edge_index)

        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask].long().squeeze())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # update records
        recorder.update_output(data.n_id[data.train_mask].to(
            'cpu'), out[data.train_mask].detach().to('cpu'))
        recorder.update_train_loss(
            data.n_id[data.train_mask].to('cpu'), loss.detach().to('cpu'))

        total_loss += float(loss.mean()) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples

def train_wprune(epoch, k, model,optimizer):
    model.train()
    meta_net.eval()

    pbar = tqdm(total=len(p_train_loader))
    pbar.set_description(f'Training with pruning epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in p_train_loader:
        data = data.to(device)

        record = recorder.get_record(data.n_id).to(device)
        with torch.no_grad():
            weights = meta_net(record, data.edge_index).squeeze()
        n_weights, ind = torch.topk(weights, k)
        n_weights = n_weights / n_weights.mean()
        data = _filter(data.to('cpu'), ind.to('cpu')).to(device)

        #out = model(data.x * n_weights, data.edge_index, data.edge_attr)
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask].long().squeeze())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # update records
        recorder.update_output(data.n_id[data.train_mask].to(
            'cpu'), out[data.train_mask].detach().to('cpu'))
        recorder.update_train_loss(
            data.n_id[data.train_mask].to('cpu'), loss.detach().to('cpu'))

        total_loss += float(loss.mean()) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test2(model, data, evaluator, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc

def test(model):
    model.eval()
    meta_net.train()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)

        record = recorder.get_record(data.n_id).to(device)
        weights = meta_net(record, data.edge_index)
        weights = weights / weights.mean()

        out = model(data.x * weights, data.edge_index)

        mask = data.train_mask + data.valid_mask

        loss = criterion(out[mask], data.y[mask].long().squeeze())

        meta_optimizer.zero_grad()
        loss.mean().backward()
        meta_optimizer.step()

        recorder.update_output(data.n_id[mask], out[mask].detach().to('cpu'))
        recorder.update_val_loss(data.n_id[mask], loss.detach().to('cpu'))

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].argmax(dim=-1, keepdim=True).cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return train_rocauc, valid_rocauc, test_rocauc

@torch.no_grad()
def test_wprune(model):
    model.eval()
    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)

        record = recorder.get_record(data.n_id).to(device)

        out = model(data.x, data.edge_index)

        mask = data.train_mask + data.valid_mask

        loss = criterion(out[mask], data.y[mask].long().squeeze())

        # meta_optimizer.zero_grad()
        # loss.mean().backward()
        # meta_optimizer.step()

        recorder.update_output(data.n_id[mask], out[mask].detach().to('cpu'))
        recorder.update_val_loss(data.n_id[mask], loss.detach().to('cpu'))

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].argmax(dim=-1, keepdim=True).cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['acc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['acc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['acc']

    return train_rocauc, valid_rocauc, test_rocauc

num_parts=40
prune_ratio=0.8
dataset = PygNodePropPredDataset('ogbn-products', root='/mnt/data/ogbdata')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.n_id = torch.arange(data.num_nodes)
data.node_species = None
data.y = data.y.float()
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
#Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = GraphSAINTRandomWalkSampler(data,
                                         batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir)
test_loader = GraphSAINTRandomWalkSampler(data,
                                         batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir)

p_train_loader = GraphSAINTRandomWalkSampler(data,
                                         batch_size=args.batch_size*2,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir)

subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12)

k = int(args.batch_size * args.prune_ratio)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)
model2 = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout).to(device)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
evaluator = Evaluator('ogbn-products')

recorder = Record(num_nodes=data.num_nodes, num_classes=47)

meta_net = MetaNet(input_dim=49, hidden_dim=32).to(device)
meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=1e-4)

# best = 0
# for epoch in range(1, 1001):
#     loss = train(epoch, model1, optimizer1)
#     train_rocauc, valid_rocauc, test_rocauc = test(model1)
#     if test_rocauc > best:
#         best = test_rocauc
#     print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
#         f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
# print('best: {}'.format(best))
#logger = Logger(args.runs, args)

N1 = 40
N2 = 150
best_test_1 = 0
best_test_2 = 0
best_acc_1 =0
best_acc_2 =0
for epoch in range(1,N1+1):
    loss = train(epoch, model1, optimizer1)
    train_rocauc, valid_rocauc, test_rocauc = test(model1)
    if test_rocauc > best_test_1:
        best_test_1 = test_rocauc
    print(f'Phase 1 Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
        f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    if epoch > 9 and epoch % args.eval_steps == 0:
            result = test2(model1, data, evaluator, subgraph_loader, device)
            #logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            if best_acc_1 < test_acc:
                best_acc_1 = test_acc
            print(  f'Epoch: {epoch:02d}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Valid: {100 * valid_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
print(f'best_test_1: {best_test_1}, best_acc_1:{best_acc_1}')
for epoch in range(1, N2+1):
    loss = train_wprune(epoch, k, model1, optimizer1)
    loss = train_wprune(epoch, k, model1, optimizer1)
    # train_rocauc, valid_rocauc, test_rocauc = test_wprune(model1)
    # if test_rocauc > best_test_2:
    #     best_test_2 = test_rocauc
    # print(f'Phase 2 Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
    #     f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    if epoch % args.eval_steps == 0:
        result = test2(model1, data, evaluator, subgraph_loader, device)
        #logger.add_result(run, result)

        train_acc, valid_acc, test_acc = result
        if best_acc_2 < test_acc:
            best_acc_2 = test_acc
        print(  f'Epoch: {epoch:02d}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%')
print(f'best_test_2: {best_test_2}, best_acc_2:{best_acc_2}')
print('best_test_1 : {} , best _test_2 : {}'.format(best_test_1, best_test_2))
print('num parts {}'.format(num_parts))
print('prune ratio {}'.format(prune_ratio))