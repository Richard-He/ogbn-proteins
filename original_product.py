import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sampler import NeighborSampler
from torch_geometric.nn import GATConv

sizes=[10]*3
batch_size = 512
test_size = 1024
ratio = 0.90
times = 10
best = 0
start_epochs = 500
prune_epochs = 200

prune_set = 'train'
dataset = PygNodePropPredDataset('ogbn-products', root='/mnt/data/ogbdata')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0]

train_idx = split_idx['train']
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                split_idx=split_idx,
                               sizes=sizes, batch_size=batch_size,
                               shuffle=True, prune=True, prune_set=prune_set num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)
recordloss = torch.zeros(data.num_nodes)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super(GAT, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(dataset.num_features, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        #pbar = tqdm(total=x_all.size(0) * self.num_layers)
        #pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                #pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        #pbar.close()

        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(dataset.num_features, 128, dataset.num_classes, num_layers=3,
            heads=4)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()
    meta_net.eval()
    # pbar = tqdm(total=train_idx.size(0))
    # pbar.set_description(f'Training Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        #pbar.update(batch_size)

    # pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test(epoch, prune=False):
    model.eval()

    out = model.inference(x)

    if prune == True:
        recordloss = F.nll_loss(out, data.y,reduction='false').squeeze()
    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

    for epoch in range(1, 101):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 50:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)


for epoch in range(start_epochs):
    loss, acc = train(epoch)
    # train_rocauc, valid_rocauc, test_rocauc = test(epoch=epoch)
    # print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
    #       f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    logger.info('Epochs: {}, Loss: {:.4f}, Approx. Train:{}'.format(epoch+1, loss, acc))
    if epoch > 50:
        train_acc, val_acc, test_acc = test()
        print(f'Epochs:{epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')
        if best < test_acc:
            best = test_acc
            # print(f'********************best roc_auc: {best:.4f}***********')
            logger.info('********************best acc: {:.4f}'.format(best))
ttepochs=0
best_times =0
best_auc_roc = []
best_train_auc = []
best_val_auc = []
tr_best = 0
val_best = 0
for i in range(times):
    time_best = 0
    # print(f'ratio is {o_ratio ** (i+1)}')
    logger.info(f'--------ratio is {ratio ** (i+1)}')
    recloss = test(prune=True, epoch=0)
    #logger.info(f'ratio: {ratio}')
    del(test_loader)
    train_loader.prune(recloss, ratio)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)
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
            best_times = times
            # print('+++++++++++++++best roc_auc: {:.4f} at time {}'.format(time_best,i))
            logger.info('+++++++++++++++best test roc_auc: {:.4f} at time {}'.format(time_best,i))
        if tr_best < train_rocauc:
            tr_best = train_rocauc
            best_times = times
            # print('+++++++++++++++best roc_auc: {:.4f} at time {}'.format(time_best,i))
            #logger.info('+++++++++++++++best train roc_auc: {:.4f} at time {}'.format(tr_best,i))
        if val_best < valid_rocauc:
            val_best = valid_rocauc
            best_times = times
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

