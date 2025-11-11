# import pytorch libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.deprecation import deprecated

import os
import csv
import copy
import math
import time
import random
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse
from scipy import stats
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

from atg_91_model import GO_model_91
from atg_91_model import lm_general_allen_to_gtex
from atg_91_model import lm_general_gtex_to_gtex


# model parameters
def arg_parse():
    parser = argparse.ArgumentParser(description = 'Model arguments')
    parser.add_argument("--lr", type=float, help = "learning rate for model training")
    parser.add_argument("--flr", type=float, help = "learning rate for model finetuning")
    parser.add_argument("--epoch", type=int, help = "number of epoch")
    parser.add_argument("--fepoch", type=int, help = "number of epoch in finetuning")
    parser.add_argument("--GNN_hl1_size", type=int, help = "the size of the first hidden layer in GNN")
    parser.add_argument("--r_emb_size", type=int, help = "the size of embeddings for regions")
    parser.add_argument("--g_emb_size", type=int, help = "the size of embeddings for genes")
    parser.add_argument("--MLP_hl1_size", type=int, help = "the size of 1st hidden layer in MLP")
    parser.add_argument("--MLP_hl2_size", type=int, help = "the size of 2nd hidden layer in MLP")
    parser.add_argument("--dropout", type=float, help = "dropout rate")
    parser.add_argument("--batchsize", type=int, help = "batch size for model training")
    parser.add_argument("--predicted_region", type=str, help = "the gtex region to predict")

    parser.set_defaults(lr = 0.00001,
                        flr = 0.000001,
                        epoch = 300,
                        fepoch = 300,
                        GNN_hl1_size = 2 ** 10,
                        r_emb_size = 2 ** 4,
                        g_emb_size = 2 ** 4,
                        MLP_hl1_size = 2 ** 10,
                        MLP_hl2_size = 2 ** 6,
                        dropout = 0.5,
                        batchsize = 2048,
                        predicted_region = 'Amygdala')
    
    return parser.parse_args()


def train(args):
    
    # settings
    all_ids = ['10021', '12876', '14380', '15496', '15697', '9861']
    train_ids = ['10021', '12876', '14380', '15496', '15697']
    test_ids = ['9861']
    lr = args.lr
    f_lr = args.flr
    epoch = args.epoch
    f_epoch = args.fepoch
    GNN_hl1_size = args.GNN_hl1_size
    r_emb_size = args.r_emb_size
    g_emb_size = args.g_emb_size
    MLP_hl1_size = args.MLP_hl1_size
    MLP_hl2_size = args.MLP_hl2_size
    dropout = args.dropout
    batchsize = args.batchsize
    pick_gtex_region = args.predicted_region
    
    # initial settings
    # train the model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pick_gtex_region = 'Amygdala'
    # pick_gtex_region = 'Anterior_cingulate_cortex_BA24'
    # pick_gtex_region = 'Caudate_basal_ganglia'
    # pick_gtex_region = 'Cerebellar_Hemisphere'
    # pick_gtex_region = 'Frontal_Cortex_BA9'
    # pick_gtex_region = 'Hippocampus'
    # pick_gtex_region = 'Hypothalamus'
    # pick_gtex_region = 'Nucleus_accumbens_basal_ganglia'
    # pick_gtex_region = 'Putamen_basal_ganglia'
    # pick_gtex_region = 'Substantia_nigra'

    # name difference
    if pick_gtex_region=='Cerebellar_Hemisphere':
        allen_name='Cerebellum'
    elif pick_gtex_region=='Frontal_Cortex_BA9':
        allen_name='Cortex'
    else:
        allen_name=pick_gtex_region

    # settings
    all_ids = ['10021', '12876', '14380', '15496', '15697', '9861']

    # path
    allen_data_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/data/allen_data/allen/'
    save_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/data/allen_data/quantile_normalized_allen/'

    GeneExpression_allen_dict = {}
    # iterate over all 6 subjects
    for i in range(len(all_ids)):
        donor = all_ids[i]
        file_name = save_path + "normalized_expr_" + donor + ".csv"
        normalized_mat = pd.read_csv(file_name, header = 0)
        normalized_mat = normalized_mat.set_index('gene_symbol')
        GeneExpression_allen_dict[donor] = normalized_mat
        
    
    #####-----find the allen regions used to generate summarized gtex info-----#####
    ontology_path = allen_data_path + 'normalized_microarray_donor' + '9861' + '/Ontology.csv'
    ontology = pd.read_csv(ontology_path, header = 0)
    # From the ontology file, find the sub-regions in allen under gtex region
    gtex_map_path = allen_data_path + "map_gtex_structure.txt"
    gTex_map_dict = {}
    print("Total number of regions in allen ontology:", ontology.shape[0])
    for i in open(gtex_map_path):
        i = i.strip().split("\t")
        gtex_region = i[0].strip()
        allen_region = i[1].strip()
        if((allen_region == "none?") | (allen_region == 'pituitary body')):
            continue
        covered_allen_region = ontology.loc[(ontology['name']==allen_region) | ontology['structure_id_path'].str.startswith(ontology.loc[ontology['name']==allen_region, 'structure_id_path'].values[0]), 'id']
        gTex_map_dict[gtex_region] = covered_allen_region.tolist()
        print(gtex_region, "-->", allen_region, ";  number of regions in allen:", len(covered_allen_region))
    print("\n")
        
    intersected_region = GeneExpression_allen_dict['9861'].columns.tolist()
    used_intersected_region_dict = {}
    # unseen_intersected_region_dict = {}
    for gtex_region, covered_allen_region in gTex_map_dict.items():
        used_region_list = [x for x in intersected_region if int(x) in covered_allen_region]
        used_intersected_region_dict[gtex_region] = used_region_list
        print(gtex_region, " # regions expired:", len(used_region_list))
    num_used_region = sum(len(value) for value in used_intersected_region_dict.values())
    print("Total number of intersected region between allen and gtex:", len(intersected_region))
    print("Total number of used allen region for generating regions for gtex:", num_used_region)
    print("Total number of unseen allen regions when generating regions for gtex:", len(intersected_region)-num_used_region)
    print("\n")
    
    #####-----summarized gtex info-----#####
    # read the summarized allen data (in gtex format) into a dictionary
    save_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/data/allen_data/quantile_normalized_allen/'
    # find the file and read it into a dictionary
    summarized_gtex_dict = {}
    for file_name in os.listdir(save_path):
        if file_name.endswith('-gtex.txt'):
            key = file_name.split('-gtex.txt')[0]
            file_path = os.path.join(save_path, file_name)
            mat = pd.read_csv(file_path, sep='\t', index_col=0)
            #mat = mat.iloc[:-1]
            # Store the dataframe in the dictionary with the key
            summarized_gtex_dict[key] = mat
            
    #####-----Load gtex data-----#####
    data_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/12052023/'
    gt = pd.read_csv(data_dir+"new_normed_gtex_gtex_allen_gene.txt", low_memory=False, index_col=0, sep="\t")

    region_pick = ['Amygdala', 'Anterior_cingulate_cortex_BA24', 'Caudate_basal_ganglia', 
                   'Cerebellar_Hemisphere', 'Frontal_Cortex_BA9', 'Hippocampus', 'Hypothalamus', 
                   'Nucleus_accumbens_basal_ganglia', 'Putamen_basal_ganglia', 'Substantia_nigra']

    # build a dictionary to count the freq of each subject 
    sample_subject_list = gt.loc['subject'].tolist()
    subject_count_dict = {}
    for s in sample_subject_list:
        if s in subject_count_dict:
            subject_count_dict[s] = subject_count_dict[s] + 1
        else:
            subject_count_dict[s] = 1

    # build a dictionary to count the freq of each region
    sample_region_list = gt.loc['region'].tolist()
    region_count_dict = {}
    for s in sample_region_list:
        if s in region_count_dict:
            region_count_dict[s] = region_count_dict[s] + 1
        else:
            region_count_dict[s] = 1

    # find the subjects that have all 10 regions
    pick_subject = [s for s, c in subject_count_dict.items() if c==10]

    # build a dictionary for exp data for each subject in gtex who has all 10 brain regions
    exp_gtex_dict = {}
    for subject in pick_subject:
        submat = gt[gt.columns[gt.iloc[1]==subject]]
        submat.columns = submat.loc['region',:]
        submat = submat.iloc[2:,]
        submat.index.names = ['gene_id']
        submat = submat.sort_values(by=['gene_id'])
        submat = submat[region_pick]
        # And also, transform the dataframe in gtex from strings to numbers
        submat = submat.apply(pd.to_numeric, errors='ignore')
        # Take the average if more than 1 sample have the same gene names
        submat = submat.groupby(submat.index).mean()

        exp_gtex_dict[subject] = submat

    # find all the ids, training ids, testing ids, in the df
    sub_all_ids = list(exp_gtex_dict.keys())

    
    #####-----find genes in both allen and gtex-----#####
    # gene_module = pd.read_csv(allen_data_path+'41593_2015_BFnn4171_MOESM97_ESM.csv')
    allen_gene_list = GeneExpression_allen_dict['9861'].index
    gtex_gene_list = exp_gtex_dict['GTEX-N7MT'].index
    overlapped_gene_list = [x for x in gtex_gene_list if x in allen_gene_list]  # 15044 genes here

    # allen subject gene expression profile on the overlapped genes
    exp_allen_dict = {}
    for key, mat in GeneExpression_allen_dict.items():
        exp_allen_dict[key] = mat.loc[overlapped_gene_list]

    # summarized gtex info for allen subjects on the overlapped genes
    summ_gtex_info = {}
    for key, mat in summarized_gtex_dict.items():
        summ_gtex_info[key] = mat.loc[overlapped_gene_list]
    # rename the Cerebellum to Cerebellar_Hemisphere and Cortex to Frontal_Cortex_BA for allen people
    for subject, mat in summ_gtex_info.items():
        mat.columns = exp_gtex_dict['GTEX-N7MT'].columns
        
    # As we are trying to use 9 regions to predict the 10th, we remove the column for the predicted region
    pick_gtex_region = pick_gtex_region
    summ_9_gtex_info = {}
    for sub, mat in summ_gtex_info.items():
        summ_9_gtex_info[sub] = mat.drop(pick_gtex_region, axis=1)
        
        
    #####-----find used regions and unseen regions-----#####
    nodes_for_gtex_region = used_intersected_region_dict[allen_name]
    # Create two list, one for used regions and the other for unseen regions,
    used_region_list = []
    for region_list in used_intersected_region_dict.values():
        used_region_list = used_region_list + region_list
    unseen_region_list = [x for x in intersected_region if x not in used_region_list]
    # then, we move the picked gtex region from used regions to unseen regions and not use its info in the future
    unseen_region_list = unseen_region_list + nodes_for_gtex_region
    used_region_list = [x for x in intersected_region if x not in unseen_region_list]

    #used_regions and unseen regions indices
    used_region_idx = np.sort([intersected_region.index(x) for x in used_region_list])
    unseen_region_idx = np.sort([intersected_region.index(x) for x in unseen_region_list])
    
        
    #####-----load the gene embeddings-----#####
    g_emb_error = 0.035
    g_emb_size = 2 ** 4
    g_emb_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_91_103/'
    g_emb_name = f'allen_gtex_gene_emb_all6subjects_size_{g_emb_size}_pearson_err_{g_emb_error}_intersected103.csv'
    # np.savetxt(g_emb_path+g_emb_name, pretrain_g_emb, delimiter=',')
    # read the pretrained gene embedding
    pretrain_g_emb = np.genfromtxt(g_emb_path+g_emb_name, delimiter=',', dtype=np.float32)

    import pickle
    # Load the gene names from the file
    with open(g_emb_path+g_emb_name+'_genenames.pkl', 'rb') as file:
        gene_emb_names_list = pickle.load(file)
    
    #####-----build the edge list-----#####
    # From the Ontology find the node relationship list
    onto_file_path = 'normalized_microarray_donor10021/Ontology.csv'
    onto_file_path = os.path.join(allen_data_path, onto_file_path)
    ontology = pd.read_csv(onto_file_path)
    ontology_id = ontology.loc[:, ['id', 'parent_structure_id']]
    # set the parent node of 4005 to -1
    ontology_id.iloc[0,1] = -1
    ontology_id['parent_structure_id'] = ontology_id['parent_structure_id'].astype(int)

    # View the nodes in a hierarchical way
    node_child = [int(x) for x in intersected_region]
    all_node = []
    for i in range(1,20):
        if i==1:
            print(f"level {i}: {len(node_child)}")
            print(node_child)
            all_node.append(node_child)
        if len(node_child)==1:
            break
        if i!=1:
            node_parent = []
            for node in node_child:
                pos = ontology_id['id'].index[ontology_id['id']==node]
                # skip if it's already the ancestor
                if len(pos)==0: continue
                parent = ontology_id['parent_structure_id'][pos].values[0]
                node_parent.append(parent)
            node_parent = set(node_parent)
            node_child = [x for x in node_parent]
            print(f"level {i}: {len(node_child)}")
            print(node_child)
            all_node.append(node_child)

    repeated_nodes = [x for y in all_node for x in y]
    pick_nodes = set(repeated_nodes)
    print(f"There are {len(pick_nodes)} nodes in total")
    print("\n")
    
    pick_nodes = [x for x in pick_nodes]
    pick_nodes.sort()
    # exclude the ancestor node (4005) and the '-1' node
    intersected_nodes_child = pick_nodes[2:]
    child_nodes_chr = list(exp_allen_dict['9861'].columns)
    child_nodes = [int(x) for x in child_nodes_chr]
    # append other hyper-level nodes to the pick_nodes
    for x in intersected_nodes_child:
        if x not in child_nodes:
            child_nodes.append(x)
    # find the parent nodes for the pick_nodes
    parent_nodes = []
    for x in child_nodes:
        pos = ontology_id['id'].index[ontology_id['id']==x][0]
        parent = ontology_id['parent_structure_id'][pos]
        parent_nodes.append(parent)
        
    #trim the graph
    for _ in range(len(parent_nodes)):
        length = len(parent_nodes)
        for i in range(length):
            cid = child_nodes[i]
            pid = parent_nodes[i]
            if pid!=4005:
                # find how many children this parent node has
                count1 = parent_nodes.count(pid)
                # if this count is more than one, we don't remove this node
                if count1 > 1:
                    continue
                # if this parent node only has one child, we remove it
                else:
                    # find the position of this parent node in the children node list
                    pidx = child_nodes.index(pid)
                    # find the grandparent
                    ppid = parent_nodes[pidx]
                    # remove this parent and directly connect the child to its grandparent
                    child_nodes[pidx] = cid
                    child_nodes.pop(i)
                    parent_nodes.pop(i)
                    break
        if len(parent_nodes)==length:
            break
            
    # put the leaves at the beginning
    initial_nodes_chr = list(exp_allen_dict['9861'].columns)
    new_child_nodes = [int(x) for x in initial_nodes_chr]
    new_parent_nodes = []
    for x in child_nodes:
        if x not in new_child_nodes:
            new_child_nodes.append(x)
    for x in new_child_nodes:
        new_parent_nodes.append(parent_nodes[child_nodes.index(x)])

    # put all nodes together in order so we can re-assign node id
    all_nodes = new_child_nodes.copy()
    for x in new_parent_nodes:
        if x not in all_nodes:
            all_nodes.append(x)

    # re-index all the nodes and all the dataframe
    child_nodes_idx = []
    parent_nodes_idx = []
    for node in new_child_nodes:
        child_nodes_idx.append(all_nodes.index(node))
    for node in new_parent_nodes:
        parent_nodes_idx.append(all_nodes.index(node))

    # Also get the indices for the nodes for generating the predicted gtex region
    nodes_for_gtex_region = used_intersected_region_dict[allen_name]
    nodes_for_gtex_region_idx = [new_child_nodes.index(int(x)) for x in nodes_for_gtex_region]
        
        
    ##########----------Model pre-setting----------##########
    # train the model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # other settings
    N_gene = len(exp_allen_dict['9861'])
    N_node = len(child_nodes_idx)+1
    n_node = len(intersected_region)

    # define the edge list
    # add edges between region nodes
    edge_index_1 = [[child_nodes_idx[i], parent_nodes_idx[i]] for i in range(len(child_nodes_idx))]
    edge_index_2 = [[parent_nodes_idx[i], child_nodes_idx[i]] for i in range(len(child_nodes_idx))]
    edge_index = edge_index_1 + edge_index_2
    for i in range(N_node):
        edge_index.append([i, i])
        
        
#     ##########----------training section----------##########
#     # record the time
#     start_time = time.time()

#     # define the model, optimizer and loss function
#     model = GO_model_91(GNN_hl1_size, r_emb_size, g_emb_size, MLP_hl1_size, MLP_hl2_size, 
#                         edge_index, dropout, N_gene, N_node, n_node).to(device)
#     # save_path = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/09032023/'
#     # pretrain_GNN_name = 'allen_GNN_pretrained_model.pth'
#     # model.child_r.load_state_dict(torch.load(save_path+pretrain_GNN_name))

#     # set the embedding weights to pretrained gene embeddings
#     gene_embed = torch.tensor(pretrain_g_emb, dtype=torch.float32).to(device)
#     model.gen_emb.weight = torch.nn.Parameter(gene_embed)
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # create a dictionary to store lists of training loss and testing loss by key = region
#     keys = list(exp_allen_dict[all_ids[0]].columns)
#     train_loss = {}
#     unseen_train_avg_loss, used_train_avg_loss = [], []
#     # specific training average loss on the allen regions that we used to generate the predicted gtex region
#     specific_train_avg_loss = []
#     specific_30_test_avg_loss = []

#     for r in keys:
#         train_loss[r] = []

#     # train the model and test the model
#     for e in range(epoch):

#         #####-----Train the model on 6 allen subjects on all 103 regions-----#####
#         # set model to training mode
#         model.train()
#         # Freeze the weights of the gene embedding layer
#         model.gen_emb.weight.requires_grad = False
#         # randomized shuffle training
#         n_subject = len(all_ids)
#         n_region = len(keys)
#         n_total = n_subject * n_region
#         training_shuffle = random.sample(range(n_total), n_total)
#         for i in training_shuffle:
#             # find the subject and the region
#             subject_id = i // n_region
#             subject = all_ids[subject_id]
#             reg_id = i % n_region
#             r = keys[reg_id]
#             gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
#             targets = torch.tensor(exp_allen_dict[subject].loc[:,r]).to(device)
#             # also shuffle the genes
#             gene_shuffle_order = np.random.choice(N_gene, N_gene, replace=False)
#             gen_ids = gene_shuffle_order
#             targets = targets[gene_shuffle_order]
#             gtex_exp = gtex_exp[gene_shuffle_order]
#             # batch training
#             sample_size = len(gen_ids)
#             num_batches = int((sample_size - sample_size % batchsize) / batchsize)
#             for b in range(num_batches):
#                 # reset optimizer gradients
#                 optimizer.zero_grad()
#                 # forward propagation
#                 gen_tuple = torch.tensor(gen_ids[b*batchsize:(b+1)*batchsize], dtype=torch.long).to(device)
#                 x_reg_exp = gtex_exp[b*batchsize:(b+1)*batchsize].clone().float().to(device)
#                 pred = model([reg_id], gen_tuple, x_reg_exp, edge_index, batchsize).reshape(-1).to(device)
#                 real = targets[b*batchsize:(b+1)*batchsize].clone().float().to(device)
#                 # calculate loss
#                 loss = criterion(pred, real)
#                 # backward propagation
#                 loss.backward()
#                 # update model parameters
#                 optimizer.step()


#         #####-----Compute the average error across all unseen allen regions on training subjects-----#####
#         # set model to training mode
#         model.eval()
#         # record all pred and real to compute correlation
#         train_e = {}
#         for r in unseen_region_list:
#             train_e[r] = 0
#         for subject in all_ids:
#             for r in unseen_region_list:
#                 reg_id = keys.index(r)
#                 gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
#                 targets = torch.tensor(exp_allen_dict[subject].loc[:,r]).to(device)
#                 with torch.no_grad():
#                     gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
#                     x_reg_exp = gtex_exp.clone().float().to(device)
#                     pred = model([reg_id], gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
#                     real = targets.clone().float().to(device)
#                     loss = criterion(pred, real)
#                     train_e[r] = train_e[r] + loss.item()
#         # compute the MSE loss
#         train_e_total = 0
#         for r in unseen_region_list:
#             train_e_avg = train_e[r] / len(all_ids)
#             train_loss[r].append(train_e_avg)
#             train_e_total = train_e_total + train_e_avg
#         train_e_total_avg = train_e_total / len(unseen_region_list)
#         unseen_train_avg_loss.append(train_e_total_avg)

#         print(f"Epoch {e + 1}, training mse on {len(unseen_region_list)} unseen regions: {train_e_total_avg};")


#         #####-----Compute the average error across all used allen regions on training subjects-----#####
#         # set model to training mode
#         model.eval()
#         # record all pred and real to compute correlation
#         train_e = {}
#         for r in used_region_list:
#             train_e[r] = 0
#         for subject in all_ids:
#             for r in used_region_list:
#                 reg_id = keys.index(r)
#                 gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
#                 targets = torch.tensor(exp_allen_dict[subject].loc[:,r]).to(device)
#                 with torch.no_grad():
#                     gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
#                     x_reg_exp = gtex_exp.clone().float().to(device)
#                     pred = model([reg_id], gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
#                     real = targets.clone().float().to(device)
#                     loss = criterion(pred, real)
#                     train_e[r] = train_e[r] + loss.item()
#         # compute the MSE loss
#         train_e_total = 0
#         for r in used_region_list:
#             train_e_avg = train_e[r] / len(all_ids)
#             train_loss[r].append(train_e_avg)
#             train_e_total = train_e_total + train_e_avg
#         train_e_total_avg = train_e_total / len(used_region_list)
#         used_train_avg_loss.append(train_e_total_avg)

#         print(f"Epoch {e + 1}, training mse on {len(used_region_list)} used regions: {train_e_total_avg};")


#         #####-----Compute the specific error on the regions used to generate the predicted gtex region on training subjects-----#####
#         # set model to training mode
#         model.eval()
#         # record all pred and real to compute correlation
#         specific_train_e = 0
#         for subject in all_ids:
#             gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
#             targets = torch.tensor(summ_gtex_info[subject].loc[:,pick_gtex_region]).to(device)
#             gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
#             x_reg_exp = gtex_exp.clone().float().to(device)
#             with torch.no_grad():
#                 pred = model(nodes_for_gtex_region_idx, gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
#             real = targets.clone().float().to(device)
#             loss = criterion(pred, real)
#             specific_train_e = specific_train_e + loss.item()        
#         # compute the MSE loss
#         specific_train_avg = specific_train_e / len(all_ids)
#         specific_train_avg_loss.append(specific_train_avg)

#         print(f"Epoch {e + 1}, mse on the predicted gtex region on 6 allen subjects: {specific_train_avg};")


#         #####-----Compute the error on 30 gtex testing subjects-----#####
#         # set model to testing mode
#         model.eval()
#         # record all pred and real to compute correlation
#         specific_test_e = 0
#         for subject in sub_all_ids:
#             gtex_exp_10 = exp_gtex_dict[subject]
#             gtex_exp_9 = gtex_exp_10.drop(pick_gtex_region, axis=1)
#             gtex_exp_target = gtex_exp_10.loc[:,pick_gtex_region]
#             targets = torch.tensor(gtex_exp_target.values).to(device)
#             gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
#             x_reg_exp = torch.tensor(gtex_exp_9.values).float().to(device)
#             with torch.no_grad():
#                 pred = model(nodes_for_gtex_region_idx, gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
#             real = targets.clone().float().to(device)
#             loss = criterion(pred, real)
#             specific_test_e = specific_test_e + loss.item()
#         # compute the MSE loss
#         specific_test_avg = specific_test_e / len(sub_all_ids)
#         specific_30_test_avg_loss.append(specific_test_avg)

#         print(f"Epoch {e + 1}, mse on the predicted gtex region on 30 gtex subjects: {specific_test_avg};")


#         # print the running time
#         now_time = time.time()
#         total_time = now_time - start_time
#         print('Running time: {:.2f} seconds'.format(total_time), "\n")


#     # Save the model
#     data_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_91_103/Result/'
#     model_name = f'ATG_91_103_{pick_gtex_region}_epoch_{epoch}_bf_architecture.pth'
#     torch.save(model, data_dir+model_name)
#     weights_name = f'ATG_91_103_{pick_gtex_region}_epoch_{epoch}_bf_weights.pth'
#     torch.save(model.state_dict(), data_dir+weights_name)
        
        
#     #####-----Plot-----#####
#     # plot the average loss for every epoch
#     fig, (ax) = plt.subplots(1, 1, figsize=(10, 10))

#     # find those important dots and their positions
#     min_nn_unseen_train = min(unseen_train_avg_loss)
#     min_nn_unseen_train_pos = unseen_train_avg_loss.index(min_nn_unseen_train)+1
#     min_nn_used_train = min(used_train_avg_loss)
#     min_nn_used_train_pos = used_train_avg_loss.index(min_nn_used_train)+1
#     min_nn_specific_train = min(specific_train_avg_loss)
#     min_nn_specific_train_pos = specific_train_avg_loss.index(min_nn_specific_train)+1
#     min_nn_specific_30_test = min(specific_30_test_avg_loss)
#     min_nn_specific_30_test_pos = specific_30_test_avg_loss.index(min_nn_specific_30_test)+1

#     ax.set_title(f"Predicted Gtex region: {pick_gtex_region}; batch size: {batchsize};\n" + 
#                  f"testing subjects: 30 gtex subjects;\n", size=10)
#     ax.set_xlabel("MSE loss")
#     ax.plot(range(1,1+len(unseen_train_avg_loss)), unseen_train_avg_loss, color='black', label=f"NN: {len(unseen_region_list)} unseen regions on 6 allen")
#     ax.plot(range(1,1+len(used_train_avg_loss)), used_train_avg_loss, color='purple', label=f"NN: {len(used_region_list)} used regions on 6 allen")
#     ax.plot(range(1,1+len(specific_train_avg_loss)), specific_train_avg_loss, color='blue', label="NN: predicted region on 6 allen")
#     ax.plot(range(1,1+len(specific_30_test_avg_loss)), specific_30_test_avg_loss, color='orange', label="NN: predicted region on 30 testing gtex")

#     # add annotated dots
#     ax.scatter(min_nn_unseen_train_pos, min_nn_unseen_train, color='red')
#     ax.text(min_nn_unseen_train_pos, min_nn_unseen_train-0.02, f"{min_nn_unseen_train:.4f}", ha="left", va="center", fontsize=10)
#     ax.scatter(min_nn_used_train_pos, min_nn_used_train, color='red')
#     ax.text(min_nn_used_train_pos, min_nn_used_train-0.02, f"{min_nn_used_train:.4f}", ha="left", va="center", fontsize=10)
#     ax.scatter(min_nn_specific_train_pos, min_nn_specific_train, color='red')
#     ax.text(min_nn_specific_train_pos, min_nn_specific_train-0.02, f"{min_nn_specific_train:.4f}", ha="left", va="center", fontsize=10)
#     ax.scatter(min_nn_specific_30_test_pos, min_nn_specific_30_test, color='red')
#     ax.text(min_nn_specific_30_test_pos, min_nn_specific_30_test-0.02, f"{min_nn_specific_30_test:.4f}", ha="left", va="center", fontsize=10)

#     ax.set_ylim(0, 1)
#     ax.legend(loc='upper left', fontsize='small')

#     result_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_91_103/Result/'
#     fig_name = f'ATG_91_103_MSEcurve_{pick_gtex_region}_epoch_{epoch}'
#     plt.savefig(result_dir+fig_name+'.png', dpi=300, bbox_inches='tight')
    

    #####-----Model fine-tuning-----#####
    data_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_91_103/Result/'

    epoch = epoch
    f_epoch = f_epoch
    lr = f_lr
    model_name = f'ATG_91_103_{pick_gtex_region}_epoch_{epoch}_bf_architecture.pth'
    weights_name = f'ATG_91_103_{pick_gtex_region}_epoch_{epoch}_bf_weights.pth'
    # Load the model arthitecture and weights
    model = torch.load(data_dir+model_name)
    model.load_state_dict(torch.load(data_dir+weights_name))
    
    print("\n")
    print("##########--------------------##########")
    print("Here we start our model fine-tuning!")
    print("##########--------------------##########")
    print("\n")
    
    # record the time
    start_time = time.time()

    # define the model, optimizer and loss function
    finetuned_model = GO_model_91(GNN_hl1_size, r_emb_size, g_emb_size, MLP_hl1_size, MLP_hl2_size, 
                                  edge_index, dropout, N_gene, N_node, n_node).to(device)
    finetuned_model.load_state_dict(model.state_dict())

    # Freezing the parameters of child_r
    for param in finetuned_model.child_r.parameters():
        param.requires_grad = False

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(finetuned_model.parameters(), lr=lr)

    # create a dictionary to store lists of training loss and testing loss by key = region
    keys = list(exp_allen_dict[all_ids[0]].columns)
    explanatory_region_list = summ_9_gtex_info[all_ids[0]].columns.tolist()

    allen6_allregions_loss_af = {}
    allen6_unseen_loss_af, allen6_used_loss_af = [], []
    # specific training average loss on the allen regions that we used to generate the predicted gtex region
    allen6_specific_predicted_loss_af = []
    gtex30_specific_predicted_loss_af = []

    for r in keys:
        allen6_allregions_loss_af[r] = []

    # train the model and test the model
    for e in range(f_epoch):

        #####-----Fine tune the model on the 9 gtex regions on 30 gtex subjects-----#####
        # set model to training mode
        finetuned_model.train()
        # Freeze the weights of the gene embedding layer
        finetuned_model.gen_emb.weight.requires_grad = False
        # randomized shuffle training
        n_subject = len(sub_all_ids)
        n_region = 9
        n_samples = n_subject * n_region
        sample_shuffle = random.sample(range(n_samples), n_samples)
        for i in sample_shuffle:
            # find the subject and the region
            subject_idx = i // n_region
            subject = sub_all_ids[subject_idx]
            reg_idx = i % n_region
            reg_name = explanatory_region_list[reg_idx]
            if reg_name=='Cerebellar_Hemisphere':
                reg_name='Cerebellum'
            elif reg_name=='Frontal_Cortex_BA9':
                reg_name='Cortex'
            reg_id_list = used_intersected_region_dict[reg_name]
            reg_id_list_idx = [keys.index(x) for x in reg_id_list]
            # x exp mat
            gtex_exp_10 = exp_gtex_dict[subject]
            gtex_exp_9 = torch.tensor(gtex_exp_10.drop(pick_gtex_region, axis=1).values).to(device)
            gtex_exp_target = gtex_exp_10.loc[:,pick_gtex_region]
            targets = torch.tensor(gtex_exp_target.values).to(device)
            # shuffle the genes for training
            gene_shuffle_order = np.random.choice(N_gene, N_gene, replace=False)
            gen_ids = gene_shuffle_order
            targets = targets[gene_shuffle_order]
            gtex_exp = gtex_exp_9[gene_shuffle_order]        
            # batch training
            num_batches = int((N_gene - N_gene % batchsize) / batchsize)        
            for b in range(num_batches):
                # reset optimizer gradients
                optimizer.zero_grad()
                # forward propagation
                gen_tuple = torch.tensor(gen_ids[b*batchsize:(b+1)*batchsize], dtype=torch.long).to(device)
                x_reg_exp = gtex_exp[b*batchsize:(b+1)*batchsize].clone().float().to(device)
                pred = finetuned_model(reg_id_list_idx, gen_tuple, x_reg_exp, edge_index, batchsize).reshape(-1).to(device)
                real = targets[b*batchsize:(b+1)*batchsize].clone().float().to(device)
                # calculate loss
                loss = criterion(pred, real)
                # backward propagation
                loss.backward()
                # update model parameters
                optimizer.step()        


        #####-----Compute the testing error on the predicted region for 30 gtex subjects-----#####
        # set model to testing mode
        finetuned_model.eval()
        # record all pred and real to compute correlation
        gtex30_f_e = 0
        reg_name = pick_gtex_region
        if reg_name=='Cerebellar_Hemisphere':
            reg_name='Cerebellum'
        elif reg_name=='Frontal_Cortex_BA9':
            reg_name='Cortex'
        reg_id_list = used_intersected_region_dict[reg_name]
        reg_id_list_idx = [keys.index(x) for x in reg_id_list]
        for subject in sub_all_ids:
            gtex_exp_10 = exp_gtex_dict[subject]
            gtex_exp_9 = gtex_exp_10.drop(pick_gtex_region, axis=1)
            gtex_exp_target = gtex_exp_10.loc[:,pick_gtex_region]
            targets = torch.tensor(gtex_exp_target.values).to(device)
            gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
            x_reg_exp = torch.tensor(gtex_exp_9.values).float().to(device)
            with torch.no_grad():
                concat_pred = finetuned_model(reg_id_list_idx, gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
            real = targets.clone().float().to(device)
            loss = criterion(concat_pred, real)
            gtex30_f_e = gtex30_f_e + loss.item()
        # compute the MSE loss
        specific_gtex30_avg = gtex30_f_e / len(sub_all_ids)
        gtex30_specific_predicted_loss_af.append(specific_gtex30_avg)

        print(f"Model Fine-tuning Epoch {e + 1}, predicted region on 30 gtex subjects: {specific_gtex30_avg};")


        #####-----Compute the error on the unseen regions on allen subjects-----#####
        # set model to training mode
        finetuned_model.eval()
        # record all pred and real to compute correlation
        train_e = {}
        for r in unseen_region_list:
            train_e[r] = 0
        for subject in all_ids:
            for r in unseen_region_list:
                reg_id = keys.index(r)
                gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
                targets = torch.tensor(exp_allen_dict[subject].loc[:,r]).to(device)
                with torch.no_grad():
                    gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
                    x_reg_exp = gtex_exp.clone().float().to(device)
                    pred = finetuned_model([reg_id], gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
                    real = targets.clone().float().to(device)
                    loss = criterion(pred, real)
                    train_e[r] = train_e[r] + loss.item()
        # compute the MSE loss
        train_e_total = 0
        for r in unseen_region_list:
            train_e_avg = train_e[r] / len(all_ids)
            allen6_allregions_loss_af[r].append(train_e_avg)
            train_e_total = train_e_total + train_e_avg
        train_e_total_avg = train_e_total / len(unseen_region_list)
        allen6_unseen_loss_af.append(train_e_total_avg)

        print(f"Model Fine-tuning Epoch {e + 1}, {len(unseen_region_list)} unseen regions on 6 allen subjects: {train_e_total_avg};")


        #####-----Compute the error on the used regions on allen subjects-----#####
        # set model to training mode
        finetuned_model.eval()
        # record all pred and real to compute correlation
        train_e = {}
        for r in used_region_list: 
            train_e[r] = 0
        for subject in all_ids:
            for r in used_region_list:
                reg_id = keys.index(r)
                gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
                targets = torch.tensor(exp_allen_dict[subject].loc[:,r]).to(device)
                with torch.no_grad():
                    gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
                    x_reg_exp = gtex_exp.clone().float().to(device)
                    pred = finetuned_model([reg_id], gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
                    real = targets.clone().float().to(device)
                    loss = criterion(pred, real)
                    train_e[r] = train_e[r] + loss.item()
        # compute the MSE loss
        train_e_total = 0
        for r in used_region_list:
            train_e_avg = train_e[r] / len(all_ids)
            allen6_allregions_loss_af[r].append(train_e_avg)
            train_e_total = train_e_total + train_e_avg
        train_e_total_avg = train_e_total / len(used_region_list)
        allen6_used_loss_af.append(train_e_total_avg)

        print(f"Model Fine-tuning Epoch {e + 1}, {len(used_region_list)} used regions on 6 allen subjects: {train_e_total_avg};")


        #####-----Compute the predicted gtex region error on allen subjects-----#####
        # set model to training mode
        finetuned_model.eval()
        # record all pred and real to compute correlation
        allen_e = 0
        reg_name = pick_gtex_region
        if reg_name=='Cerebellar_Hemisphere':
            reg_name='Cerebellum'
        elif reg_name=='Frontal_Cortex_BA9':
            reg_name='Cortex'
        reg_id_list = used_intersected_region_dict[reg_name]
        reg_id_list_idx = [keys.index(x) for x in reg_id_list]
        for subject in all_ids:
            gtex_exp = torch.tensor(summ_9_gtex_info[subject].values).to(device)
            targets = torch.tensor(summ_gtex_info[subject].loc[:,pick_gtex_region]).to(device)
            gen_tuple = torch.tensor(np.arange(N_gene), dtype=torch.long).to(device)
            x_reg_exp = gtex_exp.clone().float().to(device)
            with torch.no_grad():
                concat_pred = finetuned_model(reg_id_list_idx, gen_tuple, x_reg_exp, edge_index, N_gene).reshape(-1).to(device)
            real = targets.clone().float().to(device)
            loss = criterion(concat_pred, real)
            allen_e = allen_e + loss.item()
        # compute the MSE loss
        specific_train_avg = allen_e / len(all_ids)
        allen6_specific_predicted_loss_af.append(specific_train_avg)

        print(f"Model Fine-tuning Epoch {e + 1}, predicted region on 6 allen subjects: {specific_train_avg};")


        # print the running time
        now_time = time.time()
        total_time = now_time - start_time
        print('Running time: {:.2f} seconds'.format(total_time), "\n")
        
        
    # save the model
    data_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_91_103/Result/'
    # Save the model arthitecture and weights
    #model_name = f'ATG_91_103_{pick_gtex_region}_trainable_GNN_af_epoch_{epoch}_finetuning_epoch_{f_epoch}_architecture.pth'
    #weights_name = f'ATG_91_103_{pick_gtex_region}_trainable_GNN_af_epoch_{epoch}_finetuning_epoch_{f_epoch}_weights.pth'
    model_name = f'ATG_91_103_{pick_gtex_region}_frozen_GNN_af_epoch_{epoch}_finetuning_epoch_{f_epoch}_architecture.pth'
    weights_name = f'ATG_91_103_{pick_gtex_region}_frozen_GNN_af_epoch_{epoch}_finetuning_epoch_{f_epoch}_weights.pth'
    torch.save(finetuned_model.state_dict(), data_dir+weights_name)
    torch.save(finetuned_model, data_dir+model_name)
        
        
    #####-----Plot-----#####
    # Loss curve before fine tuning
    fig, (ax) = plt.subplots(1, 1, figsize=(10, 10))
    e = len(allen6_unseen_loss_af)

    # find those important dots and their positions
    allen6_unseen_min = min(allen6_unseen_loss_af)
    allen6_unseen_min_pos = allen6_unseen_loss_af.index(allen6_unseen_min)+1
    allen6_used_min = min(allen6_used_loss_af)
    allen6_used_min_pos = allen6_used_loss_af.index(allen6_used_min)+1
    allen6_s_pd_min = min(allen6_specific_predicted_loss_af)
    allen6_s_pd_min_pos = allen6_specific_predicted_loss_af.index(allen6_s_pd_min)+1
    gtex30_s_min = min(gtex30_specific_predicted_loss_af)
    gtex30_s_min_pos = gtex30_specific_predicted_loss_af.index(gtex30_s_min)+1

    ax.set_title(f"FROZEN GNN; Predicted region: {pick_gtex_region};\n" + 
                 f"batch size: {batchsize}; training_epoch: {epoch}; finetuning_epoch: {f_epoch}\n" + 
                 f"gtex5 subjects: 30 gtex subjects; lr = {lr};\n", size=10)
    ax.set_xlabel("MSE loss")
    ax.plot(range(1,1+e), allen6_unseen_loss_af, color='black', label=f"NN: {len(unseen_region_list)} unseen regions on 6 allen")
    ax.plot(range(1,1+e), allen6_used_loss_af, color='purple', label=f"NN: {len(used_region_list)} used regions on 6 allen")
    ax.plot(range(1,1+e), allen6_specific_predicted_loss_af, color='blue', label="predicted region on 6 allen")
    ax.plot(range(1,1+e), gtex30_specific_predicted_loss_af, color='orange', label="predicted region on 30 testing gtex")

    # add annotated dots
    ax.scatter(allen6_unseen_min_pos, allen6_unseen_min, color='red')
    ax.text(allen6_unseen_min_pos, allen6_unseen_min-0.02, f"{allen6_unseen_min:.4f}", ha="left", va="center", fontsize=10)
    ax.scatter(allen6_used_min_pos, allen6_used_min, color='red')
    ax.text(allen6_used_min_pos, allen6_used_min-0.02, f"{allen6_used_min:.4f}", ha="left", va="center", fontsize=10)
    ax.scatter(allen6_s_pd_min_pos, allen6_s_pd_min, color='red')
    ax.text(allen6_s_pd_min_pos, allen6_s_pd_min-0.02, f"{allen6_s_pd_min:.4f}", ha="left", va="center", fontsize=10)
    ax.scatter(gtex30_s_min_pos, gtex30_s_min, color='red')
    ax.text(gtex30_s_min_pos, gtex30_s_min-0.02, f"{gtex30_s_min:.4f}", ha="left", va="center", fontsize=10)

    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize='small')

    result_dir = '/project/pi_rachel_melamed_uml_edu/Jianfeng/Allen/src/Pytorch/02162024/ATG_91_103/Result/'
    #fig_name = f'ATG_MSEcurve_91_finetuning_trainable_GNN_{pick_gtex_region}_epoch_{epoch}_f_epoch_{f_epoch}'
    fig_name = f'ATG_MSEcurve_91_finetuning_frozen_GNN_{pick_gtex_region}_epoch_{epoch}_f_epoch_{f_epoch}'
    plt.savefig(result_dir+fig_name+'.png', dpi=300, bbox_inches='tight')
    
    
#####-----run the model-----#####
def main():
    args = arg_parse()
    train(args) 

if __name__ == '__main__':
    main()