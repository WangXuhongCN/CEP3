import pickle
import os
import time
import dgl
import networkx as nx
import networkx.algorithms.community as nx_comm
import torch
import random

class Community_Hander():
    def __init__(self, g, g_sampling, args, split_ratio, random_seed):
        self.g = g
        self.g_sampling = g_sampling
        self.split_ratio = split_ratio
        self.dataset = args.dataset
        self.k_hop = args.k_hop
        self.random_seed = random_seed
        self.comm_list = self.comm_detecting()
        self.comm_tb_detelted = {}
        self.comm_tb_detelted['wikipedia'] = [128, 1, 2, 129, 3, 7, 135, 136, 10, 11, 9, 141, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 27, 29, 30, 33, 35, 40, 43, 44, 48, 50, 51, 137, 54, 55, 56, 60, 62, 63, 69, 72, 78, 79, 80, 82, 83, 85, 88, 90, 91, 94, 96, 97, 98, 101, 103, 105, 106, 108, 110, 114, 115, 116, 119, 121, 122, 123, 124, 127]
        self.comm_tb_detelted['mooc'] = [0, 2, 3, 4, 5, 9, 13, 14, 17, 20, 22, 24]
        self.comm_tb_detelted['github'] = [1, 2, 3, 4, 6, 7, 9, 12, 13, 14, 15, 16]
        self.comm_tb_detelted['social'] = [9]
        # delete some communites that have events in training phase but have no event in test phase
        # or has no event in training.
        # user may change these list by a auto-detect algorithrm.
        self.detele_comm() 
        self.num_communities = len(self.comm_list)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.k_hop)
        self.nodegraphs = None
        self.khopgraphs = None
        self.extract_communities()
        

    def comm_detecting(self):
        trainval_div = int(self.split_ratio*self.g.num_edges())
        split_ts = self.g.edata['timestamp'][trainval_div]
        g = dgl.remove_edges(self.g, torch.where(self.g.edata['timestamp'] > split_ts)[0])
        
        if not os.path.exists('communities/{}_communities.pickle'.format(self.dataset)):
            print("{} Dataset Community Detecting...".format(self.dataset))
            start = time.time()
            # g = dgl.to_simple(g, return_counts='weight', copy_ndata=False)
            nxg = g.to_networkx()
            nxg = nx.to_undirected(nxg)
            comms = nx_comm.louvain_communities(nxg,resolution=2,seed=self.random_seed) # 
            comm_sets = list(filter(lambda x: (len(x) >= 5),comms))
            comm_list = list(map(lambda x: list(x),comm_sets))
            period = round((time.time() - start),3)
            with open('communities/{}_communities.pickle'.format(self.dataset), 'wb+') as file:
                pickle.dump(comm_list, file)
            print("{} Dataset Community List File Saved, Cost {} Seconds.".format(self.dataset,period))
        else:
            print("{} Communities file is exist, directly loaded.".format(self.dataset))
            with open('communities/{}_communities.pickle'.format(self.dataset), 'rb+') as file:
                comm_list =pickle.load(file)
        # random.shuffle(comm_list)
        
        return comm_list

    def detele_comm(self):
        for idx in self.comm_tb_detelted[self.dataset]:
            self.comm_list[idx] = 0
        self.comm_list = [nodes for nodes in self.comm_list if nodes!=0]
        # print(self.comm_list)
        # self.comm_list = list(set(self.comm_list).difference(set(self.comm_tb_detelted[self.dataset])))

    def extract_communities(self):
        print('{} dataset has {} communities.'.format(self.dataset, self.num_communities))
        self.nodegraphs = list(map(lambda x: self.sorted_nodegraph(x),self.comm_list))  
        self.khopgraphs = list(map(lambda x: self.sorted_khopgraph(x),self.comm_list))
        # self.earliest_ts = list(map(lambda x: x.edata['timestamp'].min(),self.nodegraphs))  
        
    def sorted_nodegraph(self,nodes):
        sg = dgl.node_subgraph(self.g,nodes)
        sg.edata['timestamp'], indices = torch.sort(sg.edata['timestamp'])
        ntype = torch.zeros(sg.num_nodes()).bool()
        ntype[sg.out_degrees().bool()] = True
        sg.ndata['is_source'] = ntype
        sg.edata[dgl.EID] = sg.edata[dgl.EID][indices]
        sg.ndata['OriginalID'] = sg.ndata[dgl.NID]
        sg.edata['OriginalID'] = sg.edata[dgl.EID]
        return sg

    def sorted_khopgraph(self,nodes):
        # blocks = self.sampler.sample_blocks(self.g,nodes)
        selected_edges = []
        selected_nodes = [nodes]
        for i in range(self.k_hop):
            sg = dgl.compact_graphs(dgl.sampling.sample_neighbors(self.g_sampling, selected_nodes[i], -1))
            selected_nodes.append(sg.ndata[dgl.NID])
            selected_edges.append(sg.edata[dgl.EID])
        edges = selected_edges[-1]
        khopgraph = self.g_sampling.edge_subgraph(edges,preserve_nodes=True)
        # khopgraph.edata['timestamp'], indices = torch.sort(khopgraph.edata['timestamp'])
        # khopgraph.edata[dgl.EID] = khopgraph.edata[dgl.EID][indices]
        del khopgraph.ndata[dgl.NID]
        khopgraph.edata['OriginalID'] = khopgraph.edata[dgl.EID]
        return khopgraph
