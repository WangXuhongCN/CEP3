import torch
import dgl
import numpy as np
from networkx.algorithms import bipartite
from scipy.linalg import toeplitz
import pyemd
import networkx as nx
import random

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=1e-6, path='./saved_models/', name='test',trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.name = name+'.pth'

    def __call__(self, val_loss, encoder,simulator):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, simulator)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, simulator)
            self.counter = 0

    def save_checkpoint(self, val_loss, encoder, simulator):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(encoder.state_dict(), self.path+'encoder_'+self.name)
        torch.save(simulator.state_dict(), self.path+'simulator_'+self.name)
        self.val_loss_min = val_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
    #  torch.backends.cudnn.deterministic = True

class TBRecorder:
    def __init__(self):
        self.tb_cnt = 0
        self.tb_loss = torch.tensor([0]).float()

    def record(self,loss,writer):
        self.tb_loss += float(loss)
        if self.tb_cnt % 10 == 0:
            writer.add_scalar("Loss",self.tb_loss,self.tb_cnt//10)
            #print("Checkpoint:",self.tb_cnt//10)
            self.tb_loss = torch.tensor([0]).float()
        self.tb_cnt += 1

def grad_printer(grad):
    print(grad.norm())

# Emprically useful. For Xuhong but i want to delete see what happens
def compute_intrinsic_edges(block,true_g,k_hop):
    if k_hop == 1:
        sg = dgl.node_subgraph(block,true_g.nodes())
        (srcs, dsts) = sg.edges()
        # two_hopg = dgl.transform.khop_graph(block,k_hop)
    else:
        two_hopg = dgl.transform.khop_graph(block,k_hop)
        two_hopg = dgl.remove_self_loop(two_hopg)
        two_hopsrcs = two_hopg.edges()[0]
        two_hopdsts = two_hopg.edges()[1]
        one_hopsrcs = block.edges()[0]
        one_hopdsts = block.edges()[1]
        # only keep src and dst are all in true_event_graph NID
        srcs = []
        dsts = []
        true_id = true_g.ndata[dgl.NID]
        true_id_map = dict(zip(true_g.ndata[dgl.NID].tolist(),true_g.nodes().tolist()))
        for i in range(len(two_hopdsts)):
            if (two_hopsrcs[i] == true_id).any():
                if (two_hopdsts[i] == true_id).any():
                    src = true_id_map[int(two_hopsrcs[i])]
                    dst = true_id_map[int(two_hopdsts[i])]
                    if not (src in srcs and dst in dsts):
                        srcs.append(src)
                        dsts.append(dst)

        for i in range(len(one_hopsrcs)):
            if (one_hopsrcs[i]==true_id).any():
                if (one_hopdsts[i]==true_id).any():
                    src = true_id_map[int(one_hopsrcs[i])]
                    dst = true_id_map[int(one_hopdsts[i])]
                    if not (src in srcs and dst in dsts):
                        srcs.append(src)
                        dsts.append(dst)
    return srcs,dsts

# def emd(x, y, distance_scaling=1.0):
#     support_size = max(len(x), len(y))
#     d_mat = toeplitz(range(support_size)).astype(np.float64)
#     distance_mat = d_mat / distance_scaling

#     # convert histogram values x and y to float, and make them equal len
#     x = x.astype(np.float64)
#     y = y.astype(np.float64)
#     if len(x) < len(y):
#         x = np.hstack((x, [0.0] * (support_size - len(x))))
#     elif len(y) < len(x):
#         y = np.hstack((y, [0.0] * (support_size - len(y))))

#     emd = pyemd.emd(x, y, distance_mat)
#     return emd

# def l2(x, y):
#     dist = np.linalg.norm(x - y, 2)
#     return dist

# def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
#     ''' Gaussian kernel with squared distance in exponential term replaced by EMD
#     Args:
#       x, y: 1D pmf of two distributions with the same support
#       sigma: standard deviation
#     '''
#     support_size = max(len(x), len(y))
#     d_mat = toeplitz(range(support_size)).astype(np.float64)
#     distance_mat = d_mat / distance_scaling

#     # convert histogram values x and y to float, and make them equal len
#     x = x.astype(np.float64)
#     y = y.astype(np.float64)
#     if len(x) < len(y):
#         x = np.hstack((x, [0.0] * (support_size - len(x))))
#     elif len(y) < len(x):
#         y = np.hstack((y, [0.0] * (support_size - len(y))))

#     emd = pyemd.emd(x, y, distance_mat)
#     return np.exp(-emd * emd / (2 * sigma * sigma))

# def gaussian(x, y, sigma=1.0):
#     dist = np.linalg.norm(x - y, 2)
#     return np.exp(-dist * dist / (2 * sigma * sigma))

# def graphsim_metric(x_g,y_g,kernel=l2):
#     n = y_g.num_nodes()
#     # Degree similarity
#     x_g_indegree,_ = np.histogram(x_g.in_degrees().numpy(),
#                                 range=(0,200),
#                                 bins = 40)
#     x_g_inmean = x_g_indegree.mean()

#     x_g_outdegree,_ = np.histogram(x_g.out_degrees().numpy(),
#                                    range = (0,200),
#                                    bins = 40)
#     x_g_outmean = x_g_outdegree.mean()
#     # Normalize histogram and concatenate them
#     x_g_degree = np.concatenate([x_g_indegree/x_g_inmean,x_g_outdegree/x_g_outmean])

#     y_g_indegree,_ = np.histogram(y_g.in_degrees().numpy(),
#                                 range=(0,200),
#                                 bins = 40)
#     y_g_insum = y_g_indegree.sum()
#     y_g_outdegree,_ = np.histogram(y_g.out_degrees().numpy(),
#                                    range = (0,200),
#                                    bins = 40)
#     y_g_outsum = y_g_outdegree.sum()

#     # Normalize histogram and concatenate them
#     y_g_degree = np.concatenate([y_g_indegree/y_g_insum,y_g_outdegree/y_g_outsum])
#     degree_similarity = kernel(x_g_degree,y_g_degree)

#     # How many portion of edges are similar
#     x_g = dgl.add_reverse_edges(x_g)
#     y_g = dgl.add_reverse_edges(y_g)
#     nx_x_g = nx.Graph(x_g.to_networkx())
#     nx_y_g = nx.Graph(y_g.to_networkx())

#     x_clustering = np.array(list(bipartite.clustering(nx_x_g).values()))
#     y_clustering = np.array(list(bipartite.clustering(nx_y_g).values()))
#     structural_similarity = kernel(x_clustering,y_clustering)
#     return degree_similarity, structural_similarity

def detect_nan(value):
    if torch.isnan(value).any():
        print("NaN Detected")
    return value

# Here the negative Log likelihood is assume to be always positive
# For negative sampling
def estimate_nll(nlogp_pos,nlogp_neg,num):
    assert(nlogp_pos>=0)
    assert(nlogp_neg>=0)
    p_pos = torch.exp(-nlogp_pos)
    p_neg = torch.exp(-nlogp_neg)
    ratio = p_pos/p_neg
    p_pos_hat = ratio/(ratio + num-1)
    nll_pos = -torch.log(p_pos_hat)
    return nll_pos

def compute_perplexity(nll_loss):
    # assert(nll_loss>=0)
    perplexity = np.exp(nll_loss)
    return perplexity

