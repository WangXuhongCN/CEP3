import numpy as np
import torch
import dgl


import dgl.function as fn




class TemporalEdgeDataLoader(dgl.dataloading.EdgeDataLoader):
    """ TemporalEdgeDataLoader is an iteratable object to generate blocks for temporal embedding
    as well as pos and neg pair graph for memory update.

    The batch generated will follow temporal order

    Parameters
    ----------
    g : dgl.Heterograph
        graph for batching the temporal edge id as well as generate negative subgraph

    eids : torch.tensor() or numpy array
        eids range which to be batched, it is useful to split training validation test dataset

    block_sampler : dgl.dataloading.BlockSampler
        temporal neighbor sampler which sample temporal and computationally depend blocks for computation

    device : str
        'cpu' means load dataset on cpu
        'cuda' means load dataset on gpu

    collator : dgl.dataloading.EdgeCollator
        Merge input eid from pytorch dataloader to graph

    Example
    ----------
    Please refers to examples/pytorch/tgn/train.py

    """

    def __init__(self, g, eids, block_sampler, comm_nodegraphs, comm_khopgraphs, earliest_ts = None, device='cpu', collator=None, **kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        self.dataloader_kwargs = dataloader_kwargs
        self.collator = collator(g, eids, block_sampler, comm_nodegraphs , comm_khopgraphs, earliest_ts, **collator_kwargs)

        assert not isinstance(g, dgl.distributed.DistGraph), \
            'EdgeDataLoader does not support DistGraph for now. ' \
            + 'Please use DistDataLoader directly.'
        self.dataloader = torch.utils.data.DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **dataloader_kwargs)
        self.device = device

        # Precompute the CSR and CSC representations so each subprocess does not
        # duplicate.
        if dataloader_kwargs.get('num_workers', 0) > 0:
            g.create_formats_()
        self.eids_backup = eids

    def _reset_dataloader(self,eids):
        self.collator._dataset = eids
        self.dataloader = torch.utils.data.DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **self.dataloader_kwargs) 
# ====== Simple Mode ======

# Part of code comes from paper
# "APAN: Asynchronous Propagation Attention Network for Real-time Temporal Graph Embedding"
# that will be appeared in SIGMOD 21, code repo https://github.com/WangXuhongCN/APAN

class SimpleTemporalSampler(dgl.dataloading.BlockSampler):
    '''
    Simple Temporal Sampler just choose the edges that happen before the current timestamp, to build the subgraph of the corresponding nodes. 
    And then the sampler uses the simplest static graph neighborhood sampling methods.

    Parameters
    ----------

    fanouts : [int, ..., int] int list
        The neighbors sampling strategy 

    '''

    def __init__(self, g, fanouts, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts  
        self.ts = 0
        self.frontiers = [None for _ in range(len(fanouts))]
        self.community_nodes = None

    def sample_frontier(self, block_id, g, seed_nodes):
        '''
        Deleting the the edges that happen after the current timestamp, then use a simple topk edge sampling by timestamp.
        '''
        fanout = self.fanouts[block_id]
        # List of neighbors to sample per edge type for each GNN layer, starting from the first layer.
        g = dgl.in_subgraph(g, seed_nodes)  
        g = dgl.remove_edges(g, torch.where(g.edata['timestamp'] > self.ts)[0])  # Deleting the the edges that happen after the current timestamp

        if fanout is None:  # full neighborhood sampling
            frontier = g
        else:
            frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)  # most recent timestamp edge sampling
        self.frontiers[block_id] = frontier  # save frontier
        return frontier

class SimpleTemporalEdgeCollator(dgl.dataloading.EdgeCollator):
    '''
    Temporal Edge collator merge the edges specified by eid: items

    Parameters
    ----------

    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.

    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.

    block_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.

    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.

    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None, which excludes nothing.

        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.

        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.

        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.

    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.

    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.

    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:

        * The original (heterogeneous) graph.

        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.

        It should return

        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.

        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    '''
    def __init__(self, g, eids, block_sampler, g_sampling=None, exclude=None,
                reverse_eids=None, reverse_etypes=None, negative_sampler=None):
        super(SimpleTemporalEdgeCollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        self.n_layer = len(self.block_sampler.fanouts)

    def collate(self,items): 
        '''
        items: edge id in graph g.
        We sample iteratively k-times and batch them into one single subgraph.
        '''
        current_ts = self.g.edata['timestamp'][items[0]]     #only sample edges before current timestamp
        self.block_sampler.ts = current_ts    # restore the current timestamp to the graph sampler.


        # if link prefiction, we use a negative_sampler to generate neg-graph for loss computing.
        if self.negative_sampler is None:
            neg_pair_graph = None
            input_nodes, pair_graph, blocks = self._collate(items) # Here items are edge id. 
        else:
            input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

        # we sampling k-hop subgraph and batch them into one graph
        for i in range(self.n_layer-1):
            self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[i+1].edges())
        frontier = self.block_sampler.frontiers[0]
        # computing node last-update timestamp
        frontier.update_all(fn.copy_e('timestamp','ts'), fn.max('ts','timestamp'))
    
        return input_nodes, pair_graph, neg_pair_graph, [frontier]#,src_node,dst_node

class SimpleTemporalEdgeMPICollator(dgl.dataloading.EdgeCollator):
    '''
    Temporal Edge collator merge the edges specified by eid: items
    The edge items are splited temporally to different processes
    according to its the rank of the process

    Parameters
    ----------

    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.

    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.

    block_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.

    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.

    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * None, which excludes nothing.

        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.

        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.

        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.

    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.

    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.

    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:

        * The original (heterogeneous) graph.

        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.

        It should return

        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.

        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    '''
    def __init__(self, g, eids, block_sampler, comm_nodegraphs , comm_khopgraphs, earliest_ts, g_sampling=None, exclude=None,
                reverse_eids=None, reverse_etypes=None, negative_sampler=None,rank=None,n_proc=None):
        super(SimpleTemporalEdgeMPICollator,self).__init__(g,eids,block_sampler,
                                                 g_sampling,exclude,reverse_eids,reverse_etypes,negative_sampler)
        assert(rank!=None)
        assert(n_proc!=None)
        self.rank = rank
        self.n_proc = n_proc
        self.n_layer = len(self.block_sampler.fanouts)
        self.fanouts = self.block_sampler.fanouts
        self.num_comm = len(comm_nodegraphs)
        self.comm_nodegraphs = comm_nodegraphs
        self.comm_khopgraphs = comm_khopgraphs
        self.queried_comm_id = 0
        self.earliest_ts = earliest_ts

    def collate(self,items): 
        '''
        items: timestamps in graph g.
        We sample iteratively k-times and batch them into one single subgraph.
        only sample edges before current timestamp
        
        '''
        sub_batch_size = len(items)//self.n_proc
        items = items[sub_batch_size*self.rank:sub_batch_size*(self.rank+1)]
        latest_ts = items[-1]
        current_ts = latest_ts
        nodegraph = self.comm_nodegraphs[self.queried_comm_id]
        pair_graph = dgl.remove_edges(nodegraph, torch.where(nodegraph.edata['timestamp'] >= current_ts)[0])
        if pair_graph.num_edges() > sub_batch_size:
            pair_graph = dgl.remove_edges(pair_graph, torch.arange(0,pair_graph.num_edges()-sub_batch_size))

        frontier = self.comm_khopgraphs[self.queried_comm_id]
        frontier = dgl.remove_edges(frontier, torch.where(frontier.edata['timestamp'] >= current_ts)[0])

        selected_edges = []
        selected_nodes = [nodegraph.ndata['OriginalID']]
        for i in range(self.n_layer):
            sg = dgl.compact_graphs(dgl.sampling.select_topk(frontier, self.fanouts[i],'timestamp', selected_nodes[i], 'in'))
            selected_nodes.append(sg.ndata[dgl.NID])
            selected_edges.append(sg.edata[dgl.EID])
        edges = selected_edges[-1]

        frontier = frontier.edge_subgraph(edges,preserve_nodes=True)
        frontier.edata[dgl.EID] = frontier.edata['OriginalID']

        frontier.update_all(fn.copy_e('timestamp','ts'), fn.max('ts','timestamp'))

        return torch.tensor(self.queried_comm_id),pair_graph,[frontier]#,src_node,dst_node

