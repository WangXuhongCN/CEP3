import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from modules import TimeEncode

class StackedGCN(nn.Module):
    def __init__(self,hidden_dim,k_hop):
        super(StackedGCN,self).__init__()
        self.hidden_dim = hidden_dim
        self.k_hop = k_hop
        self.layers = nn.ModuleList()
        for l in range(self.k_hop-1):
            self.layers.append(dglnn.GraphConv(self.hidden_dim,self.hidden_dim,activation=F.relu))
        self.layers.append(dglnn.GraphConv(self.hidden_dim,self.hidden_dim,allow_zero_in_degree=True))

    def forward(self,g,x):
        with g.local_scope():
            for layer in self.layers:
                x = layer(g,x)
        return x

class ImpactFunction(nn.Module):
    def __init__(self,hidden_dim,time_dim,k_hop):
        super(ImpactFunction,self).__init__()
        self.hidden_dim = hidden_dim
        # Time encode may be problematic...
        self.time_encoder = TimeEncode(time_dim)
        self.self_prop = nn.GRUCell(hidden_dim+time_dim,hidden_dim) #GRUCell
        self.k_hop = k_hop
        self.act = nn.ReLU()
        self.neighbor_prop = StackedGCN(hidden_dim,k_hop)

    def forward(self,g,hidden,t):
        t = t.unsqueeze(dim=1) if isinstance(t,torch.Tensor) else torch.tensor([t]).unsqueeze(dim=1)
        time_encode = self.time_encoder(t).squeeze().repeat(hidden.size(0),1)
        hidden = self.neighbor_prop(g,hidden)
        hidden = self.act(hidden)
        hidden_prime = torch.cat([hidden, time_encode],dim=1)
        next_hid = self.self_prop(hidden_prime,hidden) #
        return next_hid

class HiddenInitFn(nn.Module):
    def __init__(self,
                 emb_dim,
                 hid_dim):
        super(HiddenInitFn,self).__init__()
        self.embed_dim = emb_dim
        self.hidden_dim = hid_dim
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(self.embed_dim,self.embed_dim//2)
        self.bn = nn.BatchNorm1d(self.embed_dim//2)
        self.fc2 = nn.Linear(self.embed_dim//2,self.hidden_dim)


    def forward(self,node_emb):
        x = self.fc1(node_emb)
        x = self.bn(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class EdgeProb(nn.Module):
    # p(dst | src,t,Hist)
    def __init__(self,hid_dim,time_dim):
        super(EdgeProb,self).__init__()
        time_dim = time_dim
        self.hidden_dim = hid_dim
        self.time_encode = TimeEncode(time_dim)
        self.fc1 = nn.Linear(2*self.hidden_dim+time_dim,self.hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim,1)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self,hidden_states,t):
        t = t.unsqueeze(dim=1) if isinstance(t,torch.Tensor) else torch.tensor([t]).unsqueeze(dim=1)
        time_encode = self.time_encode(t).float().squeeze().repeat(hidden_states.size(0),1)
        hidden_states = torch.cat([hidden_states,time_encode],dim=1)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        s_t = self.fc2(hidden_states).squeeze()
        return s_t

class NodeProb(nn.Module):
    def __init__(self,hidden_dim,time_dim):
        super(NodeProb,self).__init__()
        time_dim = time_dim
        self.time_encode = TimeEncode(time_dim)
        self.fc1 = nn.Linear(hidden_dim+time_dim,hidden_dim)
        # self.bn = nn.BatchNorm1d(hidden_dim+time_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,1)
    
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self,hidden_states,t):
        t = t.unsqueeze(dim=1) if isinstance(t,torch.Tensor) else torch.tensor([t]).unsqueeze(dim=1)
        # t = t.double().repeat(hidden_states.size(0)).view(1,-1)
        time_encode = self.time_encode(t).squeeze().repeat(hidden_states.size(0),1)
        # time_encode.squeeze()
        if hidden_states.size(0)==1:
            time_encode = time_encode.view(1,-1)
        
        hidden_states = torch.cat([hidden_states,time_encode],dim=1)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        prob = self.fc2(hidden_states).squeeze()
        # print(prob.grad)
        return prob

class NodeIntensityFunction(nn.Module):
    # Compute node reciprocal intensity for time sampling
    # using reciprocal number is because the output can not stable in a small number due to sum function.
    def __init__(self,hidden_dim, num_communities):
        super(NodeIntensityFunction,self).__init__()
        self.hidden_dim = hidden_dim
        self.lambda_fn1 = nn.Linear(self.hidden_dim,self.hidden_dim//2)
        self.lambda_fn2 = nn.Linear(self.hidden_dim//2,1)
        self.act = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(self.hidden_dim//2)
        self.beta = nn.Parameter(torch.rand(num_communities))
        

    def forward(self,hidden_states,comm_id):
        hidden_states = self.lambda_fn1(hidden_states)
        hidden_states = self.bn1(hidden_states)
        lambda_t = self.lambda_fn2(hidden_states).squeeze()

        beta = torch.clamp(self.beta[comm_id],0)
        lambda_t = 1/beta * torch.log(1 + torch.exp(lambda_t*beta))

        lambda_t = lambda_t.sum()
        return lambda_t