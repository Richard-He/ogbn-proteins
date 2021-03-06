import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import numpy as np
from torch_geometric.nn import GENConv, DeepGCNLayer, SAGEConv, GCNConv
from sampler import RandomNodeSampler
from loguru import logger
from utils import StyleAdapter
from torch_sparse import SparseTensor
import argparse



parser = argparse.ArgumentParser(description='OGBN-Products (GAT)')
parser.add_argument('--lr', type=float, default=0.01)

parser.add_argument('--model_n', type=str, default='deepgcn')
parser.add_argument('--method', type=str, default='ada')
parser.add_argument('--reset',type=lambda x: (str(x).lower() == 'true'), default=False)

parser.add_argument('--num_test_parts',type=int, default=5)
parser.add_argument('--num_parts',type=int, default=30)
parser.add_argument('--times',type=int, default=20)

parser.add_argument('--prune_epochs', type=int, default=250)
parser.add_argument('--start_epochs', type=int, default=200)

parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--ratio', type=float, default=0.95)
parser.add_argument('--prune_set',type=str, default='train')
parser.add_argument('--dropout',type=float, default=0.5)
parser.add_argument('--data_dir',type=str,default='./data/')
parser.add_argument('--savept',type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--globe',type=lambda x: (str(x).lower() == 'true'), default=False)
args = parser.parse_args()

ratio = args.ratio
times = args.times
num_parts = args.num_parts
num_test_parts=args.num_test_parts
best = 0
start_epochs = args.start_epochs
prune_epochs = args.prune_epochs
prune_set = args.prune_set
reset = args.reset

num_workers = args.num_workers
model_n = args.model_n
#logging.basicConfig(filename= f'./log/test_{ratio}_{times}_{num_parts}.log', encoding = 'utf-8',
#                    level=logging.DEBUG)

log_name = 'log/protein_{}_numworker{}_model_{}_method_{}_test_full_reset_{}_{}_{}_{}_{}_{}_{}.log'.format(args.globe,num_workers,model_n,args.method,num_parts,num_test_parts,ratio,start_epochs,prune_epochs,prune_set,reset)
logger.add(log_name)
logger.info('logname: {}'.format(log_name))
logger.info('params: ratio {ratio}, times {times}, numparts {num_parts}, start epochs {start_epochs}, prune epochs {prune_epochs} ',
                                                                        ratio = ratio,
                                                                        times = times,
                                                                        num_parts = num_parts,
                                                                        start_epochs = start_epochs,
                                                                        prune_epochs = prune_epochs)
dataset = PygNodePropPredDataset('ogbn-proteins', root=args.data_dir)
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
                                  split_idx=splitted_idx, prune=True,prune_set=prune_set, num_workers=num_workers)
test_loader = RandomNodeSampler(data, num_parts=8, num_workers=num_workers)

recordloss = torch.zeros(data.num_nodes)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        adj_t = edge_index
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


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

    def reset(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

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

    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model_n == 'deepgcn':
    model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
elif model_n == 'gcn':
    model = GCN(data.num_features,256,112,num_layers=3, dropout=args.dropout).to(device)
elif model_n == 'sage':
    model = SAGE(data.num_features,256,112,num_layers=3, dropout=args.dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
evaluator = Evaluator('ogbn-proteins')



def train(epoch):
    model.train()

    # pbar = tqdm(total=len(train_loader))
    # pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        #print(data.edge_index.size())
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask]).mean()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        #pbar.update(1)

    # pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(epoch, prune=False):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    # pbar = tqdm(total=len(test_loader))
    # pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        if prune == True:
            recordloss[data.n_id] = criterion(out, data.y).mean(dim=-1).cpu()
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

    #     pbar.update(1)

    # pbar.close()
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
    # print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
    #       f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    logger.info('Epochs: {}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch+1, loss, train_rocauc,valid_rocauc, test_rocauc))

    if best < test_rocauc:
        best = test_rocauc
        # print(f'********************best roc_auc: {best:.4f}***********')
        logger.info('********************best roc_auc: {:.4f}'.format(best))
ttepochs=0
best_times =0
best_auc_roc = []
best_train_auc = []
best_val_auc = []
tr_best = 0
val_best = 0
ttratio=[]
for i in range(times):
    time_best = 0
    # print(f'ratio is {o_ratio ** (i+1)}')
    logger.info(f'--------ratio is {ratio ** (i+1)}')
    ttratio.append(ratio**(i+1))
    recloss = test(prune=True, epoch=0)
    #logger.info(f'ratio: {ratio}')
    del(test_loader)
    train_loader.prune(recloss, ratio, method=args.method, savept=args.savept,globe=args.globe)
    test_loader = RandomNodeSampler(train_loader.data, num_edges=train_loader.data.edge_index.size(1), num_parts=args.num_test_parts, num_workers=num_workers)
    if reset:
        model.reset()
        print('reset_done')
    tr_best=0
    val_best=0
    time_best=0
        
    for epoch in range(prune_epochs):
        # print(f'*******************epochs : {ttepochs}*******************')
        # logger.info('*******************epochs : {}*******************'.format(ttepochs))
        loss = train(epoch)
        train_rocauc, valid_rocauc, test_rocauc = test(epoch=epoch)
        # print(f'ratio:{ratio:.4f} Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
        #     f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
        logger.info('epochs : {}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(ttepochs,loss, train_rocauc,valid_rocauc, test_rocauc))
        ttepochs += 1

        if time_best < test_rocauc:
            time_best = test_rocauc
            # print('+++++++++++++++best roc_auc: {:.4f} at time {}'.format(time_best,i))
            logger.info('+++++++++++++++best test roc_auc: {:.4f} at time {}'.format(time_best,i))
        if tr_best < train_rocauc:
            tr_best = train_rocauc
            # print('+++++++++++++++best roc_auc: {:.4f} at time {}'.format(time_best,i))
            #logger.info('+++++++++++++++best train roc_auc: {:.4f} at time {}'.format(tr_best,i))
        if val_best < valid_rocauc:
            val_best = valid_rocauc
            # print('+++++++++++++++best roc_auc: {:.4f} at time {}'.format(time_best,i))
            logger.info('+++++++++++++++best valid roc_auc: {:.4f} at time {}'.format(val_best,i))
    best_auc_roc.append(time_best)
    best_train_auc.append(tr_best)
    best_val_auc.append(val_best)
global_best_id = np.argmax(best_auc_roc)
global_best_id_tr = np.argmax(best_train_auc)
global_best_id_val = np.argmax(best_val_auc)
# print(f'best auc_roc:{best_auc_roc[global_best_id]} at {global_best_id} time')
logger.info('best train auc_roc:{} at {} time'.format(best_train_auc[global_best_id_tr],global_best_id_tr))
logger.info('best valid auc_roc:{} at {} time'.format(best_val_auc[global_best_id_val],global_best_id_val))
logger.info('best auc_roc:{} at {} time'.format(best_auc_roc[global_best_id],global_best_id))
logger.info('best score: train: {},valid : {}, test : {}, ratios : {}'.format(str(best_train_auc) ,str(best_val_auc), str(best_auc_roc),str(ttratio)))
