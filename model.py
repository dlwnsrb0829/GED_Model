import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn.conv import GCNConv, GINConv
from math import exp
from layers import AttentionModule, TensorNetworkModule
# from GedMatrix import GedMatrixModule
# from layers import AttentionModule, TensorNetworkModule

class GedGNN(torch.nn.Module):
    def __init__(self, number_of_labels):
        super(GedGNN, self).__init__()
        self.number_labels = number_of_labels
        self.setup_layers()

    def setup_layers(self):
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(self.number_labels, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128, track_running_stats=False))
        
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.BatchNorm1d(64, track_running_stats=False))

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32, track_running_stats=False))
        
        self.convolution_1 = GINConv(nn1, train_eps=True)
        self.convolution_2 = GINConv(nn2, train_eps=True)
        self.convolution_3 = GINConv(nn3, train_eps=True)

        self.fully_cnnected_1 = nn.Linear(32, 64, bias=False)
        self.fully_cnnected_2 = nn.Linear(64, 32, bias=False)
        self.fully_cnnected_3 = nn.Linear(32, 1, bias=True)
        
        self.attention1 = AttentionModule()
        self.tensor_network1 = TensorNetworkModule()

        self.fully_connected_first1 = torch.nn.Linear(16, 16)
        self.fully_connected_second1 = torch.nn.Linear(16, 8)
        self.fully_connected_third1 = torch.nn.Linear(8, 4)
        self.scoring_layer1 = torch.nn.Linear(4, 1)

    def convolutional_pass(self, edge_index, features):
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=0.5,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=0.5,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        
        return features


    def forward(self, data):
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        features_1 = data["features_1"]
        features_2 = data["features_2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        max_len = max(len(abstract_features_1), len(abstract_features_2))
        new_tensor1 = torch.zeros(max_len, 32)
        new_tensor2 = torch.zeros(max_len, 32)
        new_tensor1[:len(abstract_features_1), :32] = abstract_features_1
        new_tensor2[:len(abstract_features_2), :32] = abstract_features_2
        abstract_features_1 = new_tensor1
        abstract_features_2 = new_tensor2

        att_mat = torch.empty(len(abstract_features_2), len(abstract_features_1))
        for i in range(len(abstract_features_2)):
            energy = torch.empty(len(abstract_features_1))
            for j in range(len(abstract_features_1)):

                concat_ij = abstract_features_1[i] + abstract_features_2[j]
                e_ij = concat_ij.view(-1, 32)
                e_ij = self.fully_cnnected_1(e_ij)
                e_ij = F.relu(e_ij)
                e_ij = self.fully_cnnected_2(e_ij)
                e_ij = F.relu(e_ij)
                e_ij = self.fully_cnnected_3(e_ij)

                energy[j] = e_ij
            for j in range(len(abstract_features_1)):
                a_ij = (torch.exp(energy[j]) / torch.sum(torch.exp(energy)))
                att_mat[i][j] = a_ij
        cost_i = torch.zeros(len(abstract_features_1))
        for i in range(len(abstract_features_1)):
            for j in range(len(abstract_features_2)):
                cost_i[i] = cost_i[i] + (att_mat[i][j] * (torch.dot(abstract_features_1[j], abstract_features_2[i])))

        bias_score = self.get_bias_score(abstract_features_1, abstract_features_2)
        cost_sum = cost_i.sum()
        
        score = torch.sigmoid(cost_sum + bias_score)

        pre_ged = (score * data["hb"])

        return pre_ged, score
    
    
    
    
    def get_bias_score(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention1(abstract_features_1)
        pooled_features_2 = self.attention1(abstract_features_2)
        scores = self.tensor_network1(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first1(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second1(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third1(scores))
        score = self.scoring_layer1(scores).view(-1)
        return score
    
    def get_bias_cost(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention2(abstract_features_1)
        pooled_features_2 = self.attention2(abstract_features_2)
        scores = self.tensor_network2(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first2(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second2(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third2(scores))
        score = self.scoring_layer2(scores).view(-1)
        return score