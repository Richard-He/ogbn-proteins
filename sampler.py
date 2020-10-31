import copy
import os.path as osp

import torch
from tqdm import tqdm
from torch_sparse import SparseTensor, spmm
import logging
from typing import List, Optional, Tuple, NamedTuple


class GraphSAINTSampler(torch.utils.data.DataLoader):
    r"""The GraphSAINT sampler base class from the `"GraphSAINT: Graph
    Sampling Based Inductive Learning Method"
    <https://arxiv.org/abs/1907.04931>`_ paper.
    Given a graph in a :obj:`data` object, this class samples nodes and
    constructs subgraphs that can be processed in a mini-batch fashion.
    Normalization coefficients for each mini-batch are given via
    :obj:`node_norm` and :obj:`edge_norm` data attributes.
    .. note::
        See :class:`torch_geometric.data.GraphSAINTNodeSampler`,
        :class:`torch_geometric.data.GraphSAINTEdgeSampler` and
        :class:`torch_geometric.data.GraphSAINTRandomWalkSampler` for
        currently supported samplers.
        For an example of using GraphSAINT sampling, see
        `examples/graph_saint.py <https://github.com/rusty1s/pytorch_geometric/
        blob/master/examples/graph_saint.py>`_.
    Args:
        data (torch_geometric.data.Data): The graph data object.
        batch_size (int): The approximate number of samples per batch.
        num_steps (int, optional): The number of iterations per epoch.
            (default: :obj:`1`)
        sample_coverage (int): How many samples per node should be used to
            compute normalization statistics. (default: :obj:`0`)
        save_dir (string, optional): If set, will save normalization
            statistics to the :obj:`save_dir` directory for faster re-use.
            (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log any
            pre-processing progress. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
            :obj:`num_workers`.
    """
    def __init__(self, data, batch_size: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None,
                 log: bool = True, prune = False, prune_set='train', prune_type='adaptive',
                 **kwargs):

        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

        self.num_steps = num_steps
        self.__batch_size__ = batch_size
        self.sample_coverage = sample_coverage
        self.log = log

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        self.data = copy.copy(data)

        super(GraphSAINTSampler,
              self).__init__(self, batch_size=1, collate_fn=self.__collate__,
                             **kwargs)
        if prune == True:
            if prune_set == 'train':
                self.train_idx = self.data.train_mask.nonzero(as_tuple=False).squeeze()
            else:
                self.train_idx = (self.data.train_mask + self.data.valid_mask).nonzero(as_tuple=False).squeeze()                
            subadj, _ = self.adj.saint_subgraph(self.train_idx)
            # subadj = self.adj.to_dense()[self.train_idx][:,self.train_idx].view(-1)
            _,_,e_idx = subadj.coo()
            self.train_e_idx = e_idx.squeeze().long()
            self.train_edge_index = self.data.edge_index[:, self.train_e_idx] 
            self.rest_e_idx = torch.LongTensor(list(set(range(self.E))  - set(self.train_e_idx.tolist())))

        # if self.sample_coverage > 0:
        #     path = osp.join(save_dir or '', self.__filename__)
        #     if save_dir is not None and osp.exists(path):  # pragma: no cover
        #         self.node_norm, self.edge_norm = torch.load(path)
        #     else:
        #         self.node_norm, self.edge_norm = self.__compute_norm__()
        #         if save_dir is not None:  # pragma: no cover
        #             torch.save((self.node_norm, self.edge_norm), path)

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.sample_coverage}.pt'

    def __len__(self):
        return self.num_steps

    def __sample_nodes__(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def prune(self, loss, ratio=0, naive=False):
        if naive == False:
            diff_loss = torch.abs(loss[self.train_edge_index[0]] - loss[self.train_edge_index[1]])
            print(diff_loss.nonzero().size())
            # print(int(len(diff_loss)*ratio))
            # print(diff_loss)
            _, mask = torch.topk(diff_loss, int(diff_loss.size(0)*ratio), largest=False)
        else:
            newE =self.train_edge_index.size(1)
            mask = torch.randperm(newE)[:int(newE*ratio)]
        # print(mask.size())
        # mask = (diff_loss <= threshold)
        # self.train_edge_index = self.train_edge_index[:,mask]
        # edge_index = torch.cat([self.train_edge_index,self.rest_edge_index], dim=1)
        # self.data.edge_index = edge_index
        self.train_e_idx = self.train_e_idx[mask]
        self.train_edge_index = self.train_edge_index[:, mask]
        # print('train', self.train_edge_index.size())
        self.data.edge_attr = self.data.edge_attr[torch.cat([self.train_e_idx, self.rest_e_idx])]
        self.data.edge_index = self.data.edge_index[:,torch.cat([self.train_e_idx, self.rest_e_idx])]
        # print(self.data.edge_attr.size(), self.data.edge_index.size())
        self.train_e_idx = torch.arange(self.train_e_idx.size(0))
        self.rest_e_idx = torch.arange(self.train_e_idx.size(0),self.train_e_idx.size(0) + self.rest_e_idx.size(0))
        # print(len(self.train_e_idx),len(self.rest_e_idx), self.train_edge_index.size(),self.data.edge_index.size())
        self.E = self.data.num_edges
        self.adj = SparseTensor(
            row=self.data.edge_index[0], col=self.data.edge_index[1],
            value=torch.arange(self.E, device=self.data.edge_index.device),
            sparse_sizes=(self.N, self.N)) 


    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        data.node_idx = node_idx
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if item.size(0) == self.N:
                data[key] = item[node_idx]
            elif item.size(0) == self.E and key != 'edge_index':
                data[key] = item[edge_idx]
            elif key!= 'edge_index':
                data[key] = item

        # if self.sample_coverage > 0:
        #     data.node_norm = self.node_norm[node_idx]
        #     data.edge_norm = self.edge_norm[edge_idx]

        return data

    # def __compute_norm__(self):
    #     node_count = torch.zeros(self.N, dtype=torch.float)
    #     edge_count = torch.zeros(self.E, dtype=torch.float)

    #     loader = torch.utils.data.DataLoader(self, batch_size=200,
    #                                          collate_fn=lambda x: x,
    #                                          num_workers=self.num_workers)

    #     if self.log:  # pragma: no cover
    #         pbar = tqdm(total=self.N * self.sample_coverage)
    #         pbar.set_description('Compute GraphSAINT normalization')

    #     num_samples = total_sampled_nodes = 0
    #     while total_sampled_nodes < self.N * self.sample_coverage:
    #         for data in loader:
    #             for node_idx, adj in data:
    #                 edge_idx = adj.storage.value()
    #                 node_count[node_idx] += 1
    #                 edge_count[edge_idx] += 1
    #                 total_sampled_nodes += node_idx.size(0)

    #                 if self.log:  # pragma: no cover
    #                     pbar.update(node_idx.size(0))
    #         num_samples += 200

    #     if self.log:  # pragma: no cover
    #         pbar.close()

    #     row, _, edge_idx = self.adj.coo()
    #     edge_norm = (node_count[row[edge_idx]] / edge_count).clamp_(0, 1e4)
    #     edge_norm[torch.isnan(edge_norm)] = 0.1

    #     node_count[node_count == 0] = 0.1
    #     node_norm = num_samples / node_count / self.N

    #     return node_norm, edge_norm


class GraphSAINTNodeSampler(GraphSAINTSampler):
    r"""The GraphSAINT node sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).
    """
    def __sample_nodes__(self, batch_size):
        edge_sample = torch.randint(0, self.E, (batch_size, self.batch_size),
                                    dtype=torch.long)

        return self.adj.storage.row()[edge_sample]


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    r"""The GraphSAINT edge sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).
    """
    def __sample_nodes__(self, batch_size):
        row, col, _ = self.adj.coo()

        deg_in = 1. / self.adj.storage.colcount()
        deg_out = 1. / self.adj.storage.rowcount()
        prob = (1. / deg_in[row]) + (1. / deg_out[col])

        # Parallel multinomial sampling (without replacement)
        # https://github.com/pytorch/pytorch/issues/11931#issuecomment-625882503
        rand = torch.rand(batch_size, self.E).log() / (prob + 1e-10)
        edge_sample = rand.topk(self.batch_size, dim=-1).indices

        source_node_sample = col[edge_sample]
        target_node_sample = row[edge_sample]

        return torch.cat([source_node_sample, target_node_sample], -1)


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class (see
    :class:`torch_geometric.data.GraphSAINTSampler`).
    Args:
        walk_length (int): The length of each random walk.
    """
    def __init__(self, data, batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super(GraphSAINTRandomWalkSampler,
              self).__init__(data, batch_size, num_steps, sample_coverage,
                             save_dir, log, **kwargs)

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, batch_size):
        start = torch.randint(0, self.N, (batch_size, ), dtype=torch.long)
        node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
        return node_idx.view(-1)




class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)


class NeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.
    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.
    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.
    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.
    .. note::
        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.
    Args:
        edge_index (LongTensor): The edge indices of the full-graph.
        size ([int]): The number of neighbors to
            sample for each node in each layer. If set to :obj:`sizes[i] = -1`,
            all neighbors are included in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: torch.Tensor, sizes: List[int],
                 split_idx=None,
                 node_idx: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 flow: str = "source_to_target", 
                 prune=False, prune_set='train',
                 prune_type='adaptive',
                 **kwargs):

        self.N = N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
        edge_attr = torch.arange(edge_index.size(1))
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           value=edge_attr, sparse_sizes=(N, N),
                           is_sorted=False)
        adj = adj.t() if flow == 'source_to_target' else adj
        self.adj = adj.to('cpu')

        if node_idx is None:
            node_idx = torch.arange(N)
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        self.sizes = sizes
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        super(NeighborSampler, self).__init__(node_idx.tolist(),
                                              collate_fn=self.sample, **kwargs)
        
        if prune == True:
            self.split_idx = split_idx
            self.E = edge_index.size(1)
            self.edge_index = edge_index
            if prune_set == 'train':
                self.train_idx = self.split_idx['train']
            else:
                self.train_idx = torch.cat([self.split_idx['train'], self.split_idx['valid']])
            subadj, _ = self.adj.saint_subgraph(self.train_idx)
            # subadj = self.adj.to_dense()[self.train_idx][:,self.train_idx].view(-1)
            _,_,e_idx = subadj.coo()
            self.train_e_idx = e_idx.squeeze().long()
            self.train_edge_index = self.edge_index[:, self.train_e_idx] 
            self.rest_idx = torch.cat([self.split_idx['valid'], self.split_idx['test']])
            subadj2, _ = self.adj.saint_subgraph(self.rest_idx)
            _,_,rest_e_idx = subadj2.coo()
            self.rest_e_idx = rest_e_idx.squeeze().long()

    # def prune_naive(self, ratio=0):
    #     mask = torch.randperm(self.E)

    #     edge_index, values = self.adj.coo()
    #     newvalues = torch.zeros(edge_index.size(1))
    #     newvalues[values>0] = 1
    #     num_edges = spmm(edge_index, values, 1, values.size)
        

    def prune(self, loss, ratio=0, naive=False):
        if naive == False:
            diff_loss = torch.abs(loss[self.train_edge_index[0]] - loss[self.train_edge_index[1]])
            #print(diff_loss.nonzero().size())
            # print(int(len(diff_loss)*ratio))
            _, mask = torch.topk(diff_loss, int(len(diff_loss)*ratio), largest=False)
        else:
            newE =self.train_edge_index.size(1)
            mask = torch.randperm(newE)[:int(newE*ratio)]
        #print('len diff_loss', len(diff_loss))
        #print('len mask', len(mask))
        # mask = (diff_loss < threshold)
        # self.train_edge_index = self.train_edge_index[:,mask]
        # edge_index = torch.cat([self.train_edge_index,self.rest_edge_index], dim=1)
        # self.data.edge_index = edge_index
        self.train_e_idx = self.train_e_idx[mask]
        self.train_edge_index = self.train_edge_index[:, mask]
        # self.edge_attr = self.data.edge_attr[torch.cat([self.train_e_idx, self.rest_e_idx])]
        self.edge_index = self.edge_index[:,torch.cat([self.train_e_idx, self.rest_e_idx])]

        # print(self.data.edge_attr.size(), self.data.edge_index.size())
        self.train_e_idx = torch.arange(self.train_e_idx.size(0))
        self.rest_e_idx = torch.arange(self.train_e_idx.size(0),self.train_e_idx.size(0) + self.rest_e_idx.size(0))
        #print(len(self.train_e_idx),len(self.rest_e_idx), self.train_edge_index.size(),self.edge_index.size())
        self.E = self.edge_index.size(1)
        self.adj = SparseTensor(
            row=self.edge_index[0], col=self.edge_index[1],
            value=torch.arange(self.E, device=self.edge_index.device),
            sparse_sizes=(self.N, self.N)) 
    
    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)
        adjs: List[Adj] = []

        n_id = batch
        for size in self.sizes:
            adj, n_id = self.adj.sample_adj(n_id, size, replace=False)
            if self.flow == 'source_to_target':
                adj = adj.t()
            row, col, e_id = adj.coo()
            size = adj.sparse_sizes()
            edge_index = torch.stack([row, col], dim=0)

            adjs.append(Adj(edge_index, e_id, size))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class RandomIndexSampler(torch.utils.data.Sampler):
    def __init__(self, num_nodes: int, num_parts: int, shuffle: bool = False):
        self.N = num_nodes
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):
        n_id = torch.randint(self.num_parts, (self.N, ), dtype=torch.long)
        n_ids = [(n_id == i).nonzero(as_tuple=False).view(-1)
                 for i in range(self.num_parts)]
        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts


class RandomNodeSampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.
    .. note::
        For an example of using :obj:`RandomNodeSampler`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.
    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        shuffle (bool, optional): If set to :obj:`True`, the data is reshuffled
            at every epoch (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(self, data, num_parts: int, shuffle: bool = False, split_idx=None, num_edges=None, prune=False, prune_set='train', **kwargs):
        assert data.edge_index is not None

        self.N = N = data.num_nodes
        if num_edges != None:
            self.E = num_edges
        else:
            self.E = data.num_edges
        self.split_idx = split_idx
        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))
        self.data = copy.copy(data)
        # self.data.edge_index = edge_index
        super(RandomNodeSampler, self).__init__(
            self, batch_size=1,
            sampler=RandomIndexSampler(self.N, num_parts, shuffle),
            collate_fn=self.__collate__, **kwargs)
        if prune == True:
            self.edge_index = data.edge_index
            if prune_set == 'train':
                self.train_idx = self.split_idx['train']
            else:
                self.train_idx = torch.cat([self.split_idx['train'], self.split_idx['valid']])
            subadj, _ = self.adj.saint_subgraph(self.train_idx)
            # subadj = self.adj.to_dense()[self.train_idx][:,self.train_idx].view(-1)
            _,_,e_idx = subadj.coo()
            self.train_e_idx = e_idx.squeeze().long()
            self.train_edge_index = self.edge_index[:, self.train_e_idx] 
            self.rest_e_idx = torch.LongTensor(list(set(range(self.E))  - set(self.train_e_idx.tolist())))

        self.times = 0

    def __getitem__(self, idx):
        return idx

    def prune(self, loss, ratio, naive=False, savept=False):
        #p_loss = loss[self.train_idx]
        i = self.times
        if naive== False:        
            diff_loss = torch.abs(loss[self.train_edge_index[0]] - loss[self.train_edge_index[1]])
            # print(diff_loss.nonzero().size())
            # print(int(len(diff_loss)*ratio))
            _, mask = torch.topk(diff_loss, int(len(diff_loss) * ratio), largest=False)
        else:
            newE =self.train_edge_index.size(1)
            mask = torch.randperm(newE)[:int(newE * ratio)]
        if savept==True:
            torch.save(self.train_edge_index, f'./savept/edge_index_protein.pt')
            # torch.save(mask1, f'./savept/p_smart_mask_prune_edges_{ratio ** i:.4f}.pt')
            # torch.save(mask2, f'./savept/p_naive_mask_prune_edges_{ratio ** i:.4f}.pt')
            torch.save(loss, f'./savept/p_loss_protein.pt')
        # self.train_edge_index = self.train_edge_index[:,mask]
        # edge_index = torch.cat([self.train_edge_index,self.rest_edge_index], dim=1)
        # self.data.edge_index = edge_index
        self.train_e_idx = self.train_e_idx[mask]
        self.train_edge_index = self.train_edge_index[:, mask]
        self.data.edge_attr = self.data.edge_attr[torch.cat([self.train_e_idx, self.rest_e_idx])]
        self.data.edge_index = self.data.edge_index[:,torch.cat([self.train_e_idx, self.rest_e_idx])]

        # print(self.data.edge_attr.size(), self.data.edge_index.size())
        self.train_e_idx = torch.arange(self.train_e_idx.size(0))
        self.rest_e_idx = torch.arange(self.train_e_idx.size(0),self.train_e_idx.size(0) + self.rest_e_idx.size(0))
        # print(len(self.train_e_idx),len(self.rest_e_idx), self.train_edge_index.size(),self.data.edge_index.size())
        self.E = self.data.num_edges
        self.adj = SparseTensor(
            row=self.data.edge_index[0], col=self.data.edge_index[1],
            value=torch.arange(self.E, device=self.data.edge_index.device),
            sparse_sizes=(self.N, self.N)) 
    
    def __collate__(self, node_idx):
        node_idx = node_idx[0]
        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        data.n_id = node_idx

        adj, _ = self.adj.saint_subgraph(node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if item.size(0) == self.N:
                data[key] = item[node_idx]
            elif item.size(0) == self.E and key != 'edge_index':
                data[key] = item[edge_idx]
            elif key!= 'edge_index':
                data[key] = item

        return data