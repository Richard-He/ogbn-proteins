import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import numpy as np
from torch_geometric.nn import GENConv, DeepGCNLayer
from sampler import RandomNodeSampler
from loguru import logger
from utils import StyleAdapter

ratio = 0.9
times = 20
num_parts = 15
num_test_parts=10
best = 0
start_epochs = 400
prune_epochs = 600
prune_set = 'train'
reset = True
naive = True
num_workers = 0
#logging.basicConfig(filename= f'./log/test_{ratio}_{times}_{num_parts}.log', encoding = 'utf-8',
#                    level=logging.DEBUG)

log_name = 'log/protein_numworker{}_naive_{}_test_full_reset_{}_{}_{}_{}_{}_{}_{}.log'.format(num_workers,naive,num_parts,num_test_parts,ratio,start_epochs,prune_epochs,prune_set,reset)
logger.add(log_name)
logger.info('logname: {}'.format(log_name))
logger.info('params: ratio {ratio}, times {times}, numparts {num_parts}, start epochs {start_epochs}, prune epochs {prune_epochs} ',
                                                                        ratio = ratio,
                                                                        times = times,
                                                                        num_parts = num_parts,
                                                                        start_epochs = start_epochs,
                                                                        prune_epochs = prune_epochs)
dataset = PygNodePropPredDataset('ogbn-proteins', root='/mnt/ogbdata/')
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
test_loader = RandomNodeSampler(data, num_parts=5, num_workers=num_workers)

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

    def reset(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
evaluator = Evaluator('ogbn-proteins')



def train(epoch):
    model.train()

    # pbar = tqdm(total=len(train_loader))
    # pbar.set_description(f'Training epoch: {epoch:04d}')

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
    train_loader.prune(recloss, ratio, naive=naive)
    new_train_loader = RandomNodeSampler(train_loader.data, num_parts=num_parts, shuffle=True,
                                  split_idx=splitted_idx, prune=True,prune_set=prune_set, num_workers=num_workers)
    del(train_loader)
    train_loader = new_train_loader
    test_loader = RandomNodeSampler(train_loader.data, num_edges=train_loader.data.edge_index.size(1), num_parts=num_parts, num_workers=num_workers)
    if reset:
        model.reset()
        tr_best=0
        val_best=0
        time_best=0
        print('reset_done')
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
