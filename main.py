import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from gen_conv import GENConv
from torch_geometric.nn import DeepGCNLayer
from torch_geometric.data import RandomNodeSampler

from meta_net import Record, MetaNet


dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]

data.n_id = torch.arange(data.num_nodes)
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='sum')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = RandomNodeSampler(data, num_parts=40, shuffle=True,
                                 num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=10, num_workers=5)


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
evaluator = Evaluator('ogbn-proteins')

recorder = Record(num_nodes=data.num_nodes, num_classes=data.y.size(-1))

meta_net = MetaNet(input_dim=data.y.size(-1) + 2, hidden_dim=32).to(device)
meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr=0.01)


def train(epoch):
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

        # out = model(data.x * weights, data.edge_index, data.edge_attr)
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


def test():
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

        # out = model(data.x * weights, data.edge_index, data.edge_attr)
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


for epoch in range(1, 1001):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test()
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
