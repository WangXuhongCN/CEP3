import os
import argparse
import traceback
import time
import psutil
import csv
from functools import partial
from dgl.convert import bipartite

from tqdm import tqdm
import random
import numpy as np
import dgl
import torch
import torch.nn as nn
import math
from dgl.base import DGLError

from tgn import TGN
from data_preprocess import TemporalMoocDataset, TemporalWikipediaDataset, TemporalGithubDataset, TemporalSocialEvolveDataset
from dataloading import (SimpleTemporalEdgeMPICollator, SimpleTemporalSampler,
                         TemporalEdgeDataLoader)
from community import Community_Hander
from pointprocess import SimulationLayer,mae_metric
from graphtpp import ImpactFunction, HiddenInitFn, NodeIntensityFunction, NodeProb, EdgeProb
from utils import TBRecorder, compute_intrinsic_edges, compute_perplexity, setup_seed, EarlyStopping
from mpi.mpi_pytorch import *
from mpi.mpi_tools import *

START_SPLIT = 0.3
TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85


tb_recorder = TBRecorder()

def train_generation(args,model,dataloader,simulator,memory):
    model.train()
    simulator.train()
    last_timestamp = time.time()
    cum_time = 0
    optimizer = torch.optim.AdamW([{'params':model.parameters(),'lr':args.lr},
                                  {'params':simulator.parameters(),'lr':args.lr}])
    num_communities = dataloader.collator.num_comm
    ts_min = dataloader.eids_backup[0]
    ts_max = dataloader.eids_backup[-1]
    comm_list = list(range(num_communities))
    random.shuffle(comm_list)
    comm_time_loss,comm_type_loss = [],[]
    for comm_id in comm_list:
        batch_cnt = 1
        dataloader.collator.queried_comm_id = comm_id
        community = dataloader.collator.comm_nodegraphs[comm_id]
        comm_ts = community.edata['timestamp'][(community.edata['timestamp']>=ts_min) & (community.edata['timestamp']<=ts_max)]
        # if (not args.shuffle) and len(comm_ts) <= 5 and len(comm_ts) <= args.cpus:
        #     # print(f'give comm {comm_id} up')
        #     comm_time_loss.append(0)
        #     comm_type_loss.append(0)
        #     continue
        simulator.scale = (comm_ts.max() - comm_ts.min())/(len(comm_ts))
        dataloader._reset_dataloader(comm_ts)
        total_type_loss = 0
        total_time_loss = 0
        for i, (_, positive_pair_g, blocks) in enumerate(dataloader):
            # if positive_pair_g.num_edges() < 5 and positive_pair_g.num_edges() <= args.cpus:
            #     # print(f'give comm {comm_id} up')
            #     break
            optimizer.zero_grad()
            sync_params(simulator)
            sync_params(model)
            
            true_events_graph, feat,_,_= model.embed_forecast(positive_pair_g, blocks=blocks)
            start_t = float(true_events_graph.edata['timestamp'][0])
            stop_t = float(true_events_graph.edata['timestamp'][-1])
            edges = compute_intrinsic_edges(blocks[0],true_events_graph,1)
            loss_time,loss_type = simulator(feat,start_t,stop_t,true_events_graph,edges, comm_id=comm_id)
            total_type_loss += float(loss_type)
            total_time_loss += float(loss_time)
            retain_graph = True if batch_cnt == 1 else False
            loss = loss_type + loss_time
            loss.backward(retain_graph=retain_graph)
            batch_cnt += 1
            mpi_avg_grads(simulator)
            mpi_avg_grads(model)
            optimizer.step()
            model.detach_memory()
            if memory:
                model.update_memory(positive_pair_g)
            last_timestamp,interval_time = time_tick(last_timestamp)
            cum_time += interval_time
        comm_time_loss.append(total_time_loss/len(comm_ts))
        comm_type_loss.append(total_type_loss/len(comm_ts))
    return mpi_sum(np.mean(comm_time_loss)), mpi_sum(np.mean(comm_type_loss)), cum_time
    

def eval_recommendation(args,model,dataloader,simulator,memory):
    model.eval()
    simulator.eval()
    simulator.toggle_recommend()
    num_communities = dataloader.collator.num_comm
    ts_min = dataloader.eids_backup[0]
    ts_max = dataloader.eids_backup[-1]
    comm_list = list(range(num_communities))
    comm_time_loss,comm_type_loss,comm_ppl = [],[],[]
    for comm_id in comm_list:
        dataloader.collator.queried_comm_id = comm_id
        community = dataloader.collator.comm_nodegraphs[comm_id]
        comm_ts = community.edata['timestamp'][(community.edata['timestamp']>=ts_min) & (community.edata['timestamp']<=ts_max)]
        # if len(comm_ts)  <= 5 and len(comm_ts)  <= args.cpus:
        #     comm_time_loss.append(0)
        #     comm_type_loss.append(0)
        #     comm_ppl.append(0)
        #     continue
        simulator.scale = (comm_ts.max() - comm_ts.min())/(len(comm_ts))
        dataloader._reset_dataloader(comm_ts)        
        total_type_loss = 0
        total_time_loss = 0
        total_nll = 0
        batch_cnt = 0
        for idx, (_, positive_pair_g, blocks) in enumerate(dataloader):
            # if positive_pair_g.num_edges() < 5 and positive_pair_g.num_edges() <= args.cpus:
            #     break
            true_events_graph, feat, _, _ = model.embed_forecast(positive_pair_g, blocks=blocks)
            start_t = float(true_events_graph.edata['timestamp'][0])
            stop_t = float(true_events_graph.edata['timestamp'][-1])
            edges = compute_intrinsic_edges(blocks[0],true_events_graph,1)
            loss_time,loss_type, nll = simulator(feat,start_t,stop_t,true_events_graph,edges, comm_id=comm_id)
            total_type_loss += float(loss_type)
            total_time_loss += float(loss_time)
            total_nll += float(nll)
            model.detach_memory()
            if memory:
                model.update_memory(positive_pair_g)
            batch_cnt += 1
        
        comm_time_loss.append(total_time_loss/len(comm_ts))
        comm_type_loss.append(total_type_loss/len(comm_ts))
        comm_ppl.append(total_nll/len(comm_ts))
    simulator.toggle_recommend()
    return mpi_sum(np.mean(comm_time_loss)), mpi_sum(np.mean(comm_type_loss)), compute_perplexity(mpi_sum(np.mean(comm_ppl)))

def eval_simulation(args,model,dataloader,simulator,memory):
    model.eval()
    simulator.eval()
    comm_seqsim, comm_nll = [], []
    num_communities = dataloader.collator.num_comm
    ts_min = dataloader.eids_backup[0]
    ts_max = dataloader.eids_backup[-1]
    comm_list = list(range(num_communities))
    sim_t_l = np.zeros((num_communities,2))
    for comm_id in comm_list:
        batch_cnt = 0
        dataloader.collator.queried_comm_id = comm_id
        community = dataloader.collator.comm_nodegraphs[comm_id]
        comm_ts = community.edata['timestamp'][(community.edata['timestamp']>=ts_min) & (community.edata['timestamp']<=ts_max)]
        # if len(comm_ts) <= 5 and len(comm_ts)  <= args.cpus:
        #     comm_seqsim.append(0)
        #     comm_nll.append(0)
        #     continue
        pred_time_seq_list = []
        true_time_seq = comm_ts - comm_ts[0]
        simulator.scale = (comm_ts.max() - comm_ts.min())/(len(comm_ts))
        dataloader._reset_dataloader(comm_ts)
        total_nll = 0
        sim_t = 0
        for idx,(_, positive_pair_g, blocks) in enumerate(dataloader):
            # if positive_pair_g.num_edges() < 5 and positive_pair_g.num_edges() <= args.cpus:
            #     break
            true_events_graph, feat, _, _ = model.embed_forecast(positive_pair_g,blocks=blocks)
            start_t = float(true_events_graph.edata['timestamp'][0])
            stop_t = float(true_events_graph.edata['timestamp'][-1])
            start = time.time()
            edges = compute_intrinsic_edges(blocks[0],true_events_graph,1)
            nll, time_seq_ = simulator(feat,start_t,stop_t,true_events_graph,edges, comm_id=comm_id)
            sim_t += time.time() - start
            pred_time_seq_list.extend(time_seq_)
            total_nll+= float(nll)
            model.detach_memory()
            if memory:
                model.update_memory(positive_pair_g)
            batch_cnt += 1
        sim_t_l[comm_id][0] = len(comm_ts)
        sim_t_l[comm_id][1] = sim_t
        comm_nll.append(total_nll)
        pred_time_seq = torch.tensor(pred_time_seq_list)  
        if len(pred_time_seq[:-1]) <= len(true_time_seq[1:]):
            comm_seqsim.append(mae_metric(pred_time_seq[:-1],true_time_seq[1:],scale=simulator.scale))
        else:
            comm_seqsim.append(mae_metric(true_time_seq[:-1],pred_time_seq[1:],scale=simulator.scale))
    return np.mean(comm_nll), np.mean(comm_seqsim)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test',
                        help='name for a experiment')
    parser.add_argument("--dataset", type=str, default="github",
                        help="dataset selection wikipedia/social/mooc/github")
    parser.add_argument("--cpus",type=int,default=1)
    parser.add_argument("--k_hop", type=int, default=2,
                        help="sampling k-hop neighborhood")
    parser.add_argument("--batch_size", type=int,
                        default=200, help="Size of each batch")
    parser.add_argument("--lr", type=float,
                        default=1e-3, help="Size of each batch")                        
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--epochs", type=int, default=100,
                        help='epochs for training on entire dataset')                       
    parser.add_argument("--encoder_dim", type=int, default=100,
                        help="Embedding dim for link prediction")
    parser.add_argument("--forecaster_dim", type=int, default=50,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--aggregator", type=str, default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--seed", type=int, default=2022,
                        help="random seed")
    parser.add_argument("--viz",type=int, default=0,help="sample k edges when viz, 0 means no viz")
    parser.add_argument("--model",type=str,default='add',help='attention model [add/dot/original]')
    parser.add_argument("--shuffle", action='store_true', default=False,
                        help="If use shuffle training")
    parser.add_argument("--use-memory",action='store_true',default=False)
    parser.add_argument("--use-savedmodel",type=str,default='')
    parser.add_argument("--drop_last",action='store_true',default=False)
    args = parser.parse_args()
    args.batch_size = (args.batch_size//args.cpus)*args.cpus
    mpi_fork(args.cpus)
    setup_pytorch_for_mpi()
    if args.dataset == 'wikipedia':
        graph = TemporalWikipediaDataset()
        bipartite = True
    elif args.dataset == 'github':
        graph = TemporalGithubDataset()
        bipartite = False
    elif args.dataset == 'social':
        graph = TemporalSocialEvolveDataset()
        bipartite = False
    elif args.dataset == 'mooc':
        graph = TemporalMoocDataset()
        bipartite = True
    else:
        raise DGLError('Unknow dataset')
    if args.viz:
        if not os.path.exists('viz_graph'):
            os.mkdir('viz_graph')

    # Change the unit of the timestamp from seconds to hours. It is very important for easy traning.
    graph.edata['timestamp'] = graph.edata['timestamp']/3600

    g_sampling =  dgl.add_reverse_edges(
        graph, copy_edata=True)
    g_sampling.ndata[dgl.NID] = g_sampling.nodes()

    # args.encoder_dim = args.forecaster_dim = args.encoder_dim = args.encoder_dim
    setup_seed(args.seed)
    # Pre-process data, mask new node in test set from original graph
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    community_finder = Community_Hander(graph, g_sampling, args, TRAIN_SPLIT, args.seed)
    num_communities = community_finder.num_communities
    unseen_split = max(1,int(0.1 * num_communities))
    comm_nodegraphs = community_finder.nodegraphs
    comm_khopgraphs = community_finder.khopgraphs
    
    start_from = int(START_SPLIT*num_edges)
    num_train = int(TRAIN_SPLIT*num_edges)
    num_val = int((1-VALID_SPLIT)*num_edges)
    train_ts = graph.edata['timestamp'][start_from:num_train].long()
    earliest_train_ts = train_ts[0]
    earliest_val_ts = earliest_new_test_ts = train_ts[-1]
    earliest_test_ts = graph.edata['timestamp'][num_train+num_val-1]
    val_ts = graph.edata['timestamp'][num_train:num_train+num_val].long()
    test_ts = graph.edata['timestamp'][num_train+num_val:].long()
    new_test_ts = graph.edata['timestamp'][num_train:].long()

    # Sampler Initialization
    fan_out = [args.n_neighbors for _ in range(args.k_hop)]
    sampler = SimpleTemporalSampler(graph, fan_out)
    edge_collator = partial(SimpleTemporalEdgeMPICollator,rank=proc_id(),n_proc=num_procs())

    '''
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(
        k=args.num_negative_samples)
    '''
    neg_sampler = None # TPPGGNNegativeSampler(k=args.num_negative_samples)

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct. :-unseen_split
    train_dataloader = TemporalEdgeDataLoader(graph,
                                              train_ts,
                                              sampler,
                                              comm_nodegraphs,
                                              comm_khopgraphs,
                                              earliest_ts = earliest_train_ts,
                                              batch_size=args.batch_size, # TODO
                                              negative_sampler=neg_sampler,
                                              shuffle=args.shuffle,
                                              drop_last=args.drop_last,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling)

    valid_dataloader = TemporalEdgeDataLoader(graph,
                                              val_ts,
                                              sampler,
                                              comm_nodegraphs,
                                              comm_khopgraphs,
                                              earliest_ts = earliest_val_ts,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=args.drop_last,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling
                                              )

    test_dataloader = TemporalEdgeDataLoader(graph,
                                             test_ts,
                                             sampler,
                                             comm_nodegraphs,
                                             comm_khopgraphs,
                                             earliest_ts=earliest_test_ts,
                                             batch_size=args.batch_size,
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=args.drop_last,
                                             num_workers=0,
                                             collator=edge_collator,
                                             g_sampling=g_sampling)

    edge_dim = graph.edata['feats'].shape[1]
    num_node = graph.num_nodes()

    model = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.encoder_dim,
                temporal_dim=args.encoder_dim,
                embedding_dim=args.encoder_dim,
                num_heads=args.num_heads,
                num_nodes=num_node,
                n_neighbors=args.n_neighbors,
                memory_updater_type='gru',
                model=args.model)

    # Prepare for simulator
    # print(scale)
    hidden_dim = args.forecaster_dim
    time_dim = 10
    hidden_init_fn = HiddenInitFn(args.encoder_dim,hidden_dim)

    impact_function = ImpactFunction(hidden_dim,time_dim,args.k_hop)
    edge_prob_fn = EdgeProb(hidden_dim,time_dim)
    node_prob_fn = NodeProb(hidden_dim,time_dim) 
    intensity_fn = NodeIntensityFunction(hidden_dim,num_communities)
   
    simulator = SimulationLayer(update_period=1,
                                    impact_function=impact_function,
                                    node_function=node_prob_fn,
                                    edge_function=edge_prob_fn,
                                    hidden_init_function=hidden_init_fn,
                                    node_intensity_function = intensity_fn,
                                    bipartite=bipartite,
                                    viz = args.viz)

    if args.use_savedmodel!='':
        simulator.load_state_dict(torch.load('./saved_models/'+args.use_savedmodel+'_simulator.pth'))
        model.load_state_dict(torch.load('./saved_models/'+args.use_savedmodel+'_model.pth'))

    # Implement Logging mechanism
    train_log = open("logs/{}_{}_{}.csv".format(args.dataset,args.name,args.seed), 'w')
    train_writer = csv.writer(train_log)
    train_writer.writerow([k+':'+str(v) for k,v in vars(args).items()])
    train_writer.writerow(["epoch","community","stage","MAE","loss","PPL"])
    converge_loss = np.zeros(args.epochs)
    time_stamps = np.zeros(args.epochs)
    try:
        for i in range(args.epochs):
            simulator._reset_viz_graph(graph.num_nodes())
            start_time = time.time()
            model.reset_memory()
            avg_comm_log_vector = np.zeros((num_communities,4,3))
            if args.use_savedmodel == '':
                train_time_loss,train_nll, cum_time = train_generation(args,model,train_dataloader,simulator,args.use_memory)
            with torch.no_grad():
                valid_time_loss, valid_nll, valid_ppl = eval_recommendation(args,model,valid_dataloader,simulator,args.use_memory)
                memory_checkpoint = model.store_memory()
                test_time_loss, test_nll, test_ppl = eval_recommendation(args,model,test_dataloader,simulator,args.use_memory)

                # Multistep Forecasting only after 10 epoch will the simulator model stable enough for running simulation
                train_mae = valid_mae = test_mae = test_nn_mae = 0
                model.restore_memory(memory_checkpoint)
                valid_ppl_sim, valid_mae = eval_simulation(args,model,valid_dataloader,simulator,args.use_memory)

                test_ppl_sim, test_mae = eval_simulation(args,model,test_dataloader,simulator,args.use_memory)

            log_content = []       
            if args.use_savedmodel != '':
                log_vector = np.array([[(valid_ppl+valid_mae),valid_ppl,valid_mae],
                                    [(test_ppl+test_mae),test_ppl,test_mae],
                                    ])
                
            else:
                log_vector = np.array([[cum_time,train_time_loss,train_nll],
                                    [(valid_ppl+valid_mae),valid_ppl,valid_mae],
                                    [(test_ppl+test_mae),test_ppl,test_mae],
                                    ])
            
            log_vector = mpi_avg(log_vector)
            
            end_time = time.time() - start_time
            if proc_id() == 0:
                offset = 0
                if args.use_savedmodel == '':
                    train_writer.writerow([i, 'Train', log_vector[0,0],log_vector[0,1],log_vector[0,2]])
                    print("Epoch: {} Train time loss {:.6f} type Loss {:.6f}, Cost {:.2f} seconds".format(
                        i, train_time_loss,train_nll, end_time))
                else:
                    offset = 1
                train_writer.writerow([i, 'Valid', log_vector[1-offset,0],log_vector[1-offset,1],log_vector[1-offset,2]])
                print("Epoch: {} Valid Total Loss {:.6f} Perplexity {:.6f} MAE {:.6f}".format(
                        i, log_vector[1-offset,0],log_vector[1-offset,1],log_vector[1-offset,2]))
                train_writer.writerow([i, 'Test', log_vector[2-offset,0],log_vector[2-offset,1],log_vector[2-offset,2]])
                print("Epoch: {} Test Loss {:.6f} Perplexity {:.6f} MAE {:.6f}".format(
                        i, log_vector[2-offset,0],log_vector[2-offset,1],log_vector[2-offset,2]))
                

    except:
        traceback.print_exc()
        print("========Training Interreputed!========")
    finally:
        train_log.close()
        print("========Training is Done========")
    if proc_id() == 0:
        torch.save(model.state_dict(),'./saved_models/{}_model.pth'.format(args.dataset))
        torch.save(simulator.state_dict(),'./saved_models/{}_simulator.pth'.format(args.dataset))
        if args.viz:
            simulator.save_graphs(i,args.dataset)