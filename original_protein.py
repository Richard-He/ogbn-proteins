import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.nn import GENConv, DeepGCNLayer
from sampler import RandomNodeSampler
import logging

o_ratio = 0.9
times = 10
threshold = 0
num_parts = 20
best = 0
start_epochs = 100
prune_epochs = 100
#logging.basicConfig(filename= f'./log/test_{ratio}_{times}_{num_parts}.log', encoding = 'utf-8',
#                    level=logging.DEBUG)

dataset = PygNodePropPredDataset('ogbn-proteins', root='mnt/data/ogbdata')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask
train_loader = RandomNodeSampler(data, num_parts=num_parts, shuffle=True,
                                  split_idx=splitted_idx, prune=True, num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=num_parts, num_workers=5)

recordloss = torch.zeros(data.num_nodes)

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
evaluator = Evaluator('ogbn-proteins')



def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask]).mean()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(epoch, prune=False):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        if prune == True:
            recordloss[data.n_id] = criterion(out, data.y).mean(dim=-1).cpu()
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()
    if prune == False:
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
    else:
        return recordloss

for epoch in range(start_epochs):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test(epoch=epoch)
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    #logging.info(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
    #      f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    if best < test_rocauc:
        best = test_rocauc
        print(f'********************best roc_auc: {best:.4f}***********')
        #logging.info(f'best roc_auc: {best}')
ttepochs=0
ratio = o_ratio
best_times =0
for i in range(times):
    recloss = test(prune=True, epoch=0)
    print(f'ratio: {ratio}')
    #logging.info(f'ratio: {ratio}')
    del(test_loader)
    train_loader.prune(recloss, ratio)
    test_loader = RandomNodeSampler(train_loader.data, num_edges=train_loader.data.edge_index.size(1), num_parts=num_parts, num_workers=5)
    for epoch in range(prune_epochs):
        print(f'*******************epochs : {ttepochs}*******************')
        ttepochs += 1
        loss = train(epoch)
        train_rocauc, valid_rocauc, test_rocauc = test(epoch=epoch)
        print(f'ratio:{ratio:.4f} Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
        #logging.info(f'ratio:{ratio:.4f}, Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
        #    f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
        if best < test_rocauc:
            best = test_rocauc
            best_times = times
            print(f'********************best roc_auc: {best:.4f}***********')
            #logging.info(f'best roc_auc: {best:.4f}')
print(f'best times :{best_times}')