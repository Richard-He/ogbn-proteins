import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from gen_conv import GENConv
from torch_geometric.nn import DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from sampler import GraphSAINTRandomWalkSampler
from meta_net import Record, MetaNet
from utils import _filter
import logging




class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='stat',
                           t=1.0, learn_t=True, num_layers=2, norm='layer', msg_norm=True)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)

class DeeperGCN2(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN2, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='stat',
                           t=1.0, learn_t=True, num_layers=2, norm='layer', msg_norm=True)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)



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

        out = model(data.x * weights, data.edge_index, data.edge_attr)

        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask]).mean(dim=-1)

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
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask],
                         data.y[data.train_mask]).mean(dim=-1)

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

        out = model(data.x * weights, data.edge_index, data.edge_attr)

        mask = data.train_mask + data.valid_mask

        loss = criterion(out[mask], data.y[mask]).mean(dim=-1)

        meta_optimizer.zero_grad()
        loss.mean().backward()
        meta_optimizer.step()

        recorder.update_output(data.n_id[mask], out[mask].detach().to('cpu'))
        recorder.update_val_loss(data.n_id[mask], loss.detach().to('cpu'))

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

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

        out = model(data.x, data.edge_index, data.edge_attr)

        mask = data.train_mask + data.valid_mask

        loss = criterion(out[mask], data.y[mask]).mean(dim=-1)

        # meta_optimizer.zero_grad()
        # loss.mean().backward()
        # meta_optimizer.step()

        recorder.update_output(data.n_id[mask], out[mask].detach().to('cpu'))
        recorder.update_val_loss(data.n_id[mask], loss.detach().to('cpu'))

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

num_parts=40
prune_ratio=0.
dataset = PygNodePropPredDataset('ogbn-proteins', root='mnt/data/ogbdata')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.n_id = torch.arange(data.num_nodes)
data.node_species = None
data.y = data.y.to(torch.float)
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='sum')
#Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

# train_loader = GraphSAINTRandomWalkSampler(data, batch_size=int(data.num_nodes / 400), num_steps=10,
#                                 walk_length=10)
train_loader = RandomNodeSampler(data, num_parts=num_parts, num_workers=5, shuffle=True)
test_loader = RandomNodeSampler(data, num_parts=10, num_workers=5)

# p_train_loader = GraphSAINTRandomWalkSampler(data, batch_size=int(data.num_nodes / 200), num_steps=10,
#                                 walk_length=10)
p_train_loader = RandomNodeSampler(data, num_parts=int(num_parts/2), num_workers=5, shuffle=True)
k = int(data.num_nodes / num_parts *2*prune_ratio)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = DeeperGCN2(hidden_channels=64, num_layers=28).to(device)
model2 = DeeperGCN2(hidden_channels=64, num_layers=2).to(device)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
evaluator = Evaluator('ogbn-proteins')

recorder = Record(num_nodes=data.num_nodes, num_classes=data.y.size(-1))

meta_net = MetaNet(input_dim=data.y.size(-1) + 2, hidden_dim=32).to(device)
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

N1 = 600
N2 = 700
best_test_1 = 0
best_test_2 = 0
for epoch in range(1,N1+1):
    loss = train(epoch, model1, optimizer1)
    train_rocauc, valid_rocauc, test_rocauc = test(model1)
    if test_rocauc > best_test_1:
        best_test_1 = test_rocauc
    print(f'Phase 1 Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
        f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
print('best_test_1: {}'.format(best_test_1))
for epoch in range(1, N2+1):
    loss = train_wprune(epoch, k, model1, optimizer1)
    loss = train_wprune(epoch, k, model1, optimizer1)
    train_rocauc, valid_rocauc, test_rocauc = test_wprune(model1)
    if test_rocauc > best_test_2:
        best_test_2 = test_rocauc
    print(f'Phase 2 Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
        f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
print('best_test_1 : {} , best _test_2 : {}'.format(best_test_1, best_test_2))
print('num parts {}'.format(num_parts))
print('prune ratio {}'.format(prune_ratio))
# N=[300]*10
# for i in range(len(N)):
#     if i%2 ==0:
#         for epoch in range(1,N[i]+1):
#             loss = train(epoch, model1, optimizer1)
#             train_rocauc, valid_rocauc, test_rocauc = test(model2)
#             if test_rocauc > best_test_1:
#                 best_test_1 = test_rocauc
#             print(f'Phase 1 Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
#                 f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
#     else:
#         for epoch in range(1, N[i]+1):
#             loss = train_wprune(epoch, k, model1, optimizer1)
#             train_rocauc, valid_rocauc, test_rocauc = test_wprune(model1)
#             if test_rocauc > best_test_2:
#                 best_test_2 = test_rocauc
#             print(f'Phase 2 Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
#                 f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
# print('best_test_1 : {} , best _test_2 : {}'.format(best_test_1, best_test_2))
        
