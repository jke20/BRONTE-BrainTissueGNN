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

# train the model on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create a function to generate lm model result as baseline, 
# use summarized gtex info in allen to predict gtex predicted region in gtex testing people
def lm_general_allen_to_gtex(allen_mat_dict, gtex_mat_dict, gtex_test_ids, x_region, y_region):
    # store the mse error for each module
    lm_general_train = {}
    lm_general_test = {}
    # build the x regions matrix
    allen_all_ids = ['10021', '12876', '14380', '15496', '15697', '9861']
    train_dfs = [allen_mat_dict[key].loc[:,x_region] for key in allen_all_ids]
    Xtrain = pd.concat(train_dfs, axis=0, ignore_index=True)
    Xtrain = sm.add_constant(Xtrain)
    test_dfs = [gtex_mat_dict[key].loc[:,x_region] for key in gtex_test_ids]
    Xtest = pd.concat(test_dfs, axis=0, ignore_index=True)
    Xtest = sm.add_constant(Xtest)
    # build the y region
    train_preds = [allen_mat_dict[key].loc[:,y_region] for key in allen_all_ids]
    ytrain = pd.concat(train_preds, axis=0, ignore_index=True)
    test_preds = [gtex_mat_dict[key].loc[:,y_region] for key in gtex_test_ids]
    ytest = pd.concat(test_preds, axis=0, ignore_index=True)
    # build the lm model
    fmod = sm.OLS(ytrain, Xtrain).fit()
    # Print the summary which includes p-values
    print(fmod.summary())
    # print the parameter
    intercept = fmod.params[0]
    coefficients = fmod.params[1:]
    # Print the intercept and coefficients
    print("Intercept (Bias):", intercept)
    print("\n")
    print("Coefficients (Beta values):")
    print(coefficients)
    train_pred = fmod.predict(Xtrain)
    test_pred = fmod.predict(Xtest)
    train_mse_all = mse(train_pred, ytrain)
    test_mse_all = mse(test_pred, ytest)
    # lm_result_dict[r] = [train_mse, test_mse]
    result = train_mse_all, test_mse_all
    
    return train_mse_all, test_mse_all


# create a function to generate lm model result as baseline
# use gtex training people to predict gtex testing people
def lm_general_gtex_to_gtex(gtex_mat_dict, gtex_train_ids, gtex_test_ids, x_region, y_region):
    # store the mse error for each module
    lm_result_dict = {}
    # build the x regions matrix
    train_dfs = [gtex_mat_dict[key].loc[:,x_region] for key in gtex_train_ids]
    Xtrain = pd.concat(train_dfs, axis=0, ignore_index=True)
    Xtrain = sm.add_constant(Xtrain)
    test_dfs = [gtex_mat_dict[key].loc[:,x_region] for key in gtex_test_ids]
    Xtest = pd.concat(test_dfs, axis=0, ignore_index=True)
    Xtest = sm.add_constant(Xtest)
    # build the y region
    train_preds = [gtex_mat_dict[key].loc[:,y_region] for key in gtex_train_ids]
    ytrain = pd.concat(train_preds, axis=0, ignore_index=True)
    test_preds = [gtex_mat_dict[key].loc[:,y_region] for key in gtex_test_ids]
    ytest = pd.concat(test_preds, axis=0, ignore_index=True)
    # build the lm model
    fmod = sm.OLS(ytrain, Xtrain).fit()
    # Print the summary which includes p-values
    print(fmod.summary())
    # print the parameter
    intercept = fmod.params[0]
    coefficients = fmod.params[1:]
    # Print the intercept and coefficients
    print("Intercept (Bias):", intercept)
    print("\n")
    print("Coefficients (Beta values):")
    print(coefficients)
    train_pred = fmod.predict(Xtrain)
    test_pred = fmod.predict(Xtest)
    lgg_train_mse_all = mse(train_pred, ytrain)
    lgg_test_mse_all = mse(test_pred, ytest)
    # lm_result_dict[r] = [train_mse, test_mse]
    result = lgg_train_mse_all, lgg_test_mse_all
    
    return result


# Graph Ontology model structure
# define the GCN model
class child_r(torch.nn.Module):
    def __init__(self, GNN_hl1_size, r_emb_size, dropout, N_gene, N_node):
        super(child_r, self).__init__()
        self.gcn_layers = torch.nn.ModuleList([
            GCNConv(N_node, GNN_hl1_size), 
            GCNConv(GNN_hl1_size, r_emb_size),
        ])
        self.dropout = torch.nn.Dropout(p=dropout)
        self.feature_matrix = torch.eye(N_node)
        
    def forward(self, reg_id, edge_index):
        edge_idx = edge_index
        edge_idx = torch.tensor(edge_idx, dtype=torch.long).t().contiguous().to(device)
        feature_matrix = self.feature_matrix
        x = feature_matrix.to(device)
        x = F.relu(self.gcn_layers[0](x, edge_idx))
        x = self.dropout(x)
        r_emb = self.gcn_layers[1](x, edge_idx)[reg_id]
        
        return r_emb


class GO_model_91(torch.nn.Module):
    def __init__(self, GNN_hl1_size, r_emb_size, g_emb_size, 
                 MLP_hl1_size, MLP_hl2_size, edge_index, dropout, N_gene, N_node, n_node):
        super(GO_model_91, self).__init__()
        # GNN
        self.child_r = child_r(GNN_hl1_size, r_emb_size, dropout, N_gene, N_node)
        # gene embedding
        self.gen_emb = torch.nn.Embedding(N_gene, g_emb_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(r_emb_size+g_emb_size+9, MLP_hl1_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl1_size, MLP_hl2_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(MLP_hl2_size, 1)
        )            

    def forward(self, reg_id_list, gen_id, reg_exp, edge_index, batchsize):
        predictions = []
        for reg_id in reg_id_list:
            # obtain the region embedding
            r_emb = self.child_r(reg_id, edge_index)
            # broadcast the region embedding
            r_emb_tile = r_emb.repeat(batchsize, 1)
            # gene embedding
            g_emb = self.gen_emb(gen_id)
            # concat the region embedding and gene embedding and gtex region expression and 
            # feed it into a MLP to predict the gene expression.
            concat_emb = torch.cat((r_emb_tile, g_emb, reg_exp), dim=1)
            pred = self.linear_relu_stack(concat_emb)
            predictions.append(pred)
        # compute the concatenated predictions
        concat_pred = torch.mean(torch.stack(predictions), dim=0)
        
        return concat_pred