import time
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
import dgl
from utils import estimate_nll,grad_printer
from scipy.special import kl_div
from copy import deepcopy


class OneLevelMLELoss(nn.Module):
    def __init__(self,distribution='pointprocess'):
        super(OneLevelMLELoss,self).__init__()
        self.eps = float(np.finfo(np.float32).eps)
        self.dist = distribution
        # Get the machine epsilon for representation

    def forward(self,lambda_t,Lambda_t=None):
        if self.dist == 'pointprocess':
            return -(lambda_t+self.eps).log().sum() + Lambda_t.sum()
        elif self.dist == 'multinoulli':
            if isinstance(lambda_t,float):
                return -torch.tensor(lambda_t+self.eps).log()
            return -(lambda_t+self.eps).log()

class TwoLevelMLELoss(nn.Module):
    def __init__(self):
        super(TwoLevelMLELoss,self).__init__()
        self.l1_loss = OneLevelMLELoss(distribution='multinoulli')
        self.l2_loss = OneLevelMLELoss(distribution='multinoulli')

    # Here the input lambda should be output of softmax
    # If it is a softmax output we need to take the log of result.
    def forward(self,l1_lambda_t,l2_lambda_t):
        # Here lambda should be the lambda of particular 
        l1_loss = self.l1_loss(l1_lambda_t)
        l2_loss = self.l2_loss(l2_lambda_t)
        return l2_loss + l1_loss

# Loss For negative sampling
class TwoLevelBCELoss(nn.Module):
    def __init__(self):
        super(TwoLevelBCELoss,self).__init__()
        self.loss_fn = nn.BCELoss().to('cpu')

    def forward(self,l1_pos,l2_pos,l1_neg,l2_neg):
        loss = self.loss_fn(l1_pos.sigmoid(),torch.ones_like(l1_pos))
        loss = loss + self.loss_fn(l2_pos.sigmoid(),torch.ones_like(l2_pos))
        loss = loss + self.loss_fn(l1_neg.sigmoid(),torch.zeros_like(l1_neg))
        loss = loss + self.loss_fn(l2_neg.sigmoid(),torch.zeros_like(l2_neg))
        return loss

def mae_metric(gen_time_seq,true_time_seq,scale=1.0):
    n = gen_time_seq.size(0)
    m = true_time_seq.size(0)
    T = true_time_seq[-1]
    distance = (gen_time_seq-true_time_seq[:n]).norm(p=1)
    distance = distance + (m-n)*T - true_time_seq[n:].norm(p=1)
    distance /= m
    mae = distance/(scale)
    return mae
    
class SimulationLayer(nn.Module):
    def __init__(self,
                 update_period,
                 impact_function,
                 node_intensity_function,
                 hidden_init_function,
                 edge_function,
                 node_function,
                 bipartite=True,
                 viz=False):
        super(SimulationLayer,self).__init__()
        self.update_period = update_period
        self.impact_function = impact_function
        self.node_intensity_function = node_intensity_function
        self.hidden_init = hidden_init_function
        self.edge_function = edge_function
        self.node_function = node_function
        self.bipartite = bipartite
        self.viz = viz
        self.eps = float(np.finfo(np.float32).eps)
        self.loss_fn = TwoLevelMLELoss()
        self.eval_loss_fn = TwoLevelMLELoss()
        self.recommend = False
        self.time_loss_fn = OneLevelMLELoss()
        self.sim_batch_cnt = 0

    def toggle_recommend(self):
        self.recommend = ~self.recommend

    def cont_nosim(self,step,max_step):
        if not self.training and not self.recommend:
            return False
        else:
            if step < max_step:
                return True
            else:
                return False
                
    def cont_sim(self,step,max_step,time,max_time):
        if self.training or self.recommend:
            return False
        else:
            # if step < max_step and time < max_time:
            if step < max_step :
                return True
            else:
                return False

    def save_graphs(self,epoch, dataset):

        self.viz_g.edata['timestamp'] = torch.cat(self.viz_g_ts)
        self.viz_g.edata['prob'] = torch.cat(self.viz_g_prob)
        dgl.save_graphs('viz_graph/{}_epoch{}.bin'.format(dataset,epoch),self.viz_g)
            
    def _reset_viz_graph(self,num_nodes):
        self.viz_g = dgl.graph(([],[]))
        self.viz_g.add_nodes(num_nodes)
        self.viz_g_ts = []
        self.viz_g_prob = []

    def compute_delta_t(self, step, true_events_graph):
        end_t = true_events_graph.edata['timestamp'][step]
        if step == 0:
            start_t = true_events_graph.edata['timestamp'][0]
        else:
            start_t = true_events_graph.edata['timestamp'][step-1]
        return (end_t - start_t).view(1)

    def forward(self, 
                node_embed,
                start_t, 
                stop_t, 
                true_events_graph, 
                intrinsic_edges = None,
                comm_id = None):
        self.g = dgl.graph(([],[]))
        self.g.add_nodes(node_embed.size(0))
        self.g = dgl.add_self_loop(self.g)
        if intrinsic_edges != None:
            self.g.add_edges(intrinsic_edges[0],intrinsic_edges[1])

        if self.bipartite:        
            src_nodes = true_events_graph.nodes()[true_events_graph.ndata['is_source']]
            dst_nodes = true_events_graph.nodes()[~true_events_graph.ndata['is_source']]
        else:
            src_nodes = dst_nodes = true_events_graph.nodes()

        # Should automatically set dst node intensity to zero 
        if self.bipartite:
            src_node_mask = torch.zeros(self.g.num_nodes()).bool()
            src_node_mask[src_nodes] = True
            dst_node_mask = torch.zeros(self.g.num_nodes()).bool()
            dst_node_mask[dst_nodes] = True
            new_srcnode_mask = ((node_embed == 0.0).all(dim=1) * src_node_mask).bool()
            new_dstnode_mask = ((node_embed == 0.0).all(dim=1) * dst_node_mask).bool()
            num_new_srcnode = new_srcnode_mask.sum()
            num_new_dstnode = new_dstnode_mask.sum()
            if num_new_srcnode != src_nodes.size(0):
                node_embed[new_srcnode_mask] = node_embed[~new_srcnode_mask+src_node_mask].mean(dim=0).repeat(num_new_srcnode,1)
            if num_new_dstnode != dst_nodes.size(0):
                node_embed[new_dstnode_mask] = node_embed[~new_dstnode_mask+dst_node_mask].mean(dim=0).repeat(num_new_dstnode,1)
        else:
            src_node_mask = torch.zeros(self.g.num_nodes()).bool()
            src_node_mask[src_nodes] = True
            dst_node_mask = torch.zeros(self.g.num_nodes()).bool()
            dst_node_mask[dst_nodes] = True
            
            new_node_mask = (node_embed==0.0).all(dim=1)
            num_new_node = new_node_mask.sum()
            if num_new_node != self.g.num_nodes():
                node_embed[new_node_mask] = node_embed[~new_node_mask].mean(dim=0).repeat(num_new_node,1)
        # The hidden state should be nothing since it should not contains any information
        hidden_states = self.hidden_init(node_embed)
        step = 0
        last_step = step
        total_delta_t = 0
        update_t = 0
        source_node = dest_node = None
        time_loss = 0
        total_time_loss = 0
        total_type_loss = 0
        total_nll = 0
        total_nll_sim = []
        gen_time_seq = []
        max_delta_t = stop_t - start_t
        num_edges = true_events_graph.num_edges()
        while self.cont_nosim(step,num_edges) or self.cont_sim(step,num_edges,total_delta_t,max_delta_t):
            # Hierarchical Sampling process
            # First let's sample time using information of local all nodes involved
            hidden_src = hidden_states[src_nodes]
            if self.bipartite:
                lambda_t = 1/self.node_intensity_function(hidden_src, comm_id)
            else:
                lambda_t = 1/self.node_intensity_function(hidden_states, comm_id)

            # The update node feature
            if step != 0 and step != last_step:
                update_delta_t = total_delta_t - update_t
                hidden_states = self.impact_function(self.g,hidden_states,update_delta_t)      
                update_t = deepcopy(total_delta_t)
            
            # Simulation branch should never use negative sample !
            if not self.training and not self.recommend:
                loop_cnt = self.viz+1
                while loop_cnt:
                    loop_cnt -= 1
                    if self.viz:
                        s = np.random.exponential(1/float(lambda_t+1e-10))
                    else:
                        s = 1/float(lambda_t+1e-10)
                    if self.bipartite:
                        prob_srcnodes = self.node_function(hidden_src,s).softmax(dim=0)
                    else:
                        prob_srcnodes = self.node_function(hidden_states,s).softmax(dim=0)

                    if self.viz and len(src_nodes) > 1:
                        if self.bipartite:
                            source_node = np.random.choice(src_nodes,p = prob_srcnodes.detach().numpy())
                        else:
                            source_node = np.random.choice(self.g.nodes().numpy(),p = prob_srcnodes.detach().numpy())
                    else:
                        if self.bipartite:
                            source_node = src_nodes[np.argmax(prob_srcnodes.detach().numpy())]
                        else:
                            source_node = self.g.nodes()[np.argmax(prob_srcnodes.detach().numpy())]

                    source_feat = hidden_states[source_node]
                    if self.bipartite:
                        source_feat = source_feat.repeat(dst_nodes.size(0)).view(dst_nodes.size(0),-1)
                        cat_feat = torch.cat([source_feat,hidden_states[dst_node_mask]],dim=1)
                    else:
                        source_feat = source_feat.repeat(self.g.num_nodes()).view(self.g.num_nodes(),-1)
                        cat_feat = torch.cat([source_feat,hidden_states],dim=1)
                        cat_feat = torch.cat([cat_feat[:source_node],cat_feat[source_node+1:]],dim=0)
                    if self.bipartite:
                        src_id = self.g.nodes()[src_node_mask].tolist()
                        dst_id = self.g.nodes()[dst_node_mask].tolist()
                    else:
                        src_id = self.g.nodes().tolist()
                        dst_id = self.g.nodes().tolist()
                        dst_id.remove(source_node.item())
                    prob_dstnodes = self.edge_function(cat_feat,s).squeeze().softmax(dim=0)
                    
                    if self.viz and len(dst_nodes) > 1:
                        if self.bipartite:
                            dest_node = np.random.choice(dst_nodes, 
                                                         p=prob_dstnodes.detach().numpy())
                        else:
                            dest_node = np.random.choice(dst_id,p=prob_dstnodes.detach().numpy())
                    else:
                        if self.bipartite:
                            dest_node = dst_nodes[np.argmax(prob_dstnodes.detach().numpy())]
                        else:
                            dest_node = dst_id[np.argmax(prob_dstnodes.detach().numpy())]
                    
                    total_delta_t += s
                    gen_time_seq.append(total_delta_t)
                    p_src = prob_srcnodes[src_id.index(source_node)] if prob_srcnodes.shape != torch.Size([]) else prob_srcnodes.item()
                    p_dst = prob_dstnodes[dst_id.index(dest_node)] if prob_dstnodes.shape != torch.Size([]) else prob_dstnodes.item()
                    total_nll_sim.append(self.loss_fn(p_src,p_dst))
                    self.g.add_edges([source_node,dest_node],[dest_node,source_node])
                    if type(p_src) == float:
                        p_src = torch.tensor(p_src)
                    if type(p_dst) == float:
                        p_dst = torch.tensor(p_dst)
                    if self.viz:
                        self.viz_g.add_edges(true_events_graph.ndata['OriginalID'][source_node],true_events_graph.ndata['OriginalID'][dest_node])
                        self.viz_g_ts.append(torch.tensor(start_t+total_delta_t).unsqueeze(-1).float())
                        self.viz_g_prob.append(p_src.data*p_dst.data.unsqueeze(-1).float())
            # Teacher Forcing branch should use negative sample during training and original evaluation during recommend
            else:
                true_src_node = int(true_events_graph.edges()[0][step])
                true_dst_node = int(true_events_graph.edges()[1][step])
                source_node = true_src_node
                dest_node = true_dst_node
                ts = self.compute_delta_t(step, true_events_graph)
                Lambda_t = lambda_t*ts
                total_delta_t += ts
                time_loss = self.time_loss_fn(lambda_t.sum(),Lambda_t)
                if self.recommend or self.training:
                    if self.bipartite:
                        src_id = self.g.nodes()[src_node_mask].tolist()
                        prob_srcnodes = self.node_function(hidden_src,ts).softmax(dim=0)
                    else:
                        src_id = self.g.nodes().tolist()
                        prob_srcnodes = self.node_function(hidden_states,ts).softmax(dim=0)
                    source_feat = hidden_states[true_src_node]
                    if self.bipartite:
                        source_feat = source_feat.repeat(dst_nodes.size(0)).view(dst_nodes.size(0),-1)
                        cat_feat = torch.cat([source_feat,hidden_states[dst_node_mask]],dim=1)
                        dst_id = self.g.nodes()[dst_node_mask].tolist()
                    else:
                        source_feat = source_feat.repeat(self.g.num_nodes()).view(self.g.num_nodes(),-1)
                        cat_feat = torch.cat([source_feat,hidden_states],dim=1)
                        cat_feat = torch.cat([cat_feat[:true_src_node],cat_feat[true_src_node+1:]],dim=0)
                        dst_id = self.g.nodes().tolist()
                        dst_id.remove(true_src_node)
                        
                    prob_dstnodes = self.edge_function(cat_feat,ts).squeeze()
                    prob_dstnodes = prob_dstnodes.softmax(dim=0)
                    p_src = prob_srcnodes[src_id.index(true_src_node)] if prob_srcnodes.shape != torch.Size([]) else prob_srcnodes.item()
                    p_dst = prob_dstnodes[dst_id.index(true_dst_node)] if prob_dstnodes.shape != torch.Size([]) else prob_dstnodes.item()
                    
                    loss_type = self.eval_loss_fn(p_src,p_dst)
                    nll = loss_type.detach()
                    total_nll += nll

                # Update rest of parameters
                total_time_loss = total_time_loss + time_loss
                total_type_loss = total_type_loss + loss_type
                self.g.add_edges([true_src_node,true_dst_node],[true_dst_node,true_src_node])
               
            last_step = step
            step += 1
        # Here we temporarily use entire process as metrics
        if not self.training and not self.recommend:
            self.sim_batch_cnt += 1
            if gen_time_seq == []:
                gen_time_seq.append(max_delta_t)
            return np.mean(total_nll_sim), gen_time_seq
        elif self.recommend:
            return total_time_loss, total_type_loss, total_nll
        else:
            return total_time_loss, total_type_loss

