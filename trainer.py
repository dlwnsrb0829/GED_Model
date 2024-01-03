import sys
import time
import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import load_all_graphs, load_labels, load_ged
from model import GedGNN

class Trainer(object):
    def __init__(self):
        self.to_torch_time = 0.0
        self.results = []
        self.load_data()
        self.transfer_data_to_torch()
        self.init_graph_pairs()
        self.setup_model()
        
    def setup_model(self):
        self.model = GedGNN(self.number_of_labels)
        
    def load_data(self):
        dataset_name = "AIDS"
        self.graph_num, self.graphs = load_all_graphs(dataset_name)
        # print(self.graphs)
        print("Load {} graphs.".format(self.graph_num))
        self.number_of_labels, self.features = load_labels(dataset_name)
        ged_dict = dict()
        ged_dict = load_ged(dataset_name=dataset_name)
        self.ged_dict = ged_dict
        print("Load ged dict.")
        self.graphs = self.graphs[:10000]
        self.features = self.features[:10000]
        
    def transfer_data_to_torch(self):
        t1 = time.time()

        self.edge_index = []
        for g in self.graphs:
            edge = g['graphs']
            edge = edge + [[y, x] for x, y in edge]
            # edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)
        self.features = [torch.tensor(x['onehot']).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        ged = [[0 for i in range(n)] for j in range(n)]
        gid = [g['g_num'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                else:
                    gt_ged = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = gt_ged
        self.ged = ged
        not_none_count = sum(1 for row in ged for val in row if val is not None)
        print(not_none_count)
        t2 = time.time()
        self.to_torch_time = t2 - t1
        
    def check_pair(self, i, j):
        # if i == j:
        #     return (i, j)
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return (i, j)
        elif (id2, id1) in self.ged_dict:
            return (j, i)
        else:
            return None
        
    def init_graph_pairs(self):
        self.training_graphs = []
        self.testing_graphs = []
        # print(self.ged_dict)
        # exit()
        train_num = 8000
        test_num = len(self.graphs)

        for i in range(train_num):
            for j in range(i, train_num):
                tmp = self.check_pair(i, j)
                if tmp is not None:
                    self.training_graphs.append(tmp)

        for i in range(8000, 10000):
            for j in range(i, 10000):
                tmp = self.check_pair(i, j)
                if tmp is not None:
                    self.testing_graphs.append(tmp)

        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} testing graph pairs.".format(len(self.testing_graphs)))
        
    def create_batches(self):
        batches = []
        # print(self.training_graphs)
        # exit()
        for graph in range(0, len(self.training_graphs), 128):
            batches.append(self.training_graphs[graph:graph + 128])
        # print(batches)
        # exit()
        return batches
    
    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=0.001, weight_decay=5*10**-4)

        self.model.train()
        self.values = []
        with tqdm(total = 1 * len(self.training_graphs), unit="graph_pairs", leave=True, desc="Epoch",
                  file=sys.stdout) as pbar:
            for epoch in range(1):
                batches = self.create_batches()
                loss_sum = 0
                main_index = 0
                for index, batch in enumerate(batches):
                    batch_total_loss = self.process_batch(batch)
                    loss_sum += batch_total_loss
                    main_index += len(batch)
                    loss = loss_sum / main_index
                    pbar.update(len(batch))
                    pbar.set_description(
                        "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3),
                                                                       index,
                                                                       round(1000 * batch_total_loss / len(batch), 3)))
                tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
                training_loss = round(1000 * loss, 3)
        t2 = time.time()
        training_time = t2 - t1

        self.results.append(
            ('model_name', 'dataset', 'graph_set', "current_epoch", "training_time(s/epoch)", "training_loss(1000x)"))
        self.results.append(
            ("model", "AIDS", "train", self.cur_epoch + 1, training_time, training_loss))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        with open('result/' + 'results.txt', 'a') as f:
            print("## Training", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)
            
    def process_batch(self, batch):
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float()
        weight = 1.0
        for graph_pair in batch:
            data = self.pack_graph_pair(graph_pair)
            target = data["target"]
            gt_ged = data["gt_ged"]
            pcost, cost = self.model(data)
            losses = losses + 10 * F.mse_loss(target, cost)
        losses.requires_grad_(True)
        losses.backward()
        self.optimizer.step()
        return losses.item()
    
    def pack_graph_pair(self, graph_pair):
        new_data = dict()
        # print(graph_pair)
        # print((2, 417) in self.ged_dict)
        # print(self.gid)
        # print(self.ged_dict)
        (id_1, id_2) = graph_pair
        gid_pair = (self.gid[id_1], self.gid[id_2])
        if gid_pair not in self.ged_dict:
            id_1, id_2 = (id_2, id_1)

        gt_ged = self.ged[id_1][id_2]

        new_data["id_1"] = id_1
        new_data["id_2"] = id_2

        new_data["edge_index_1"] = self.edge_index[id_1]
        new_data["edge_index_2"] = self.edge_index[id_2]
        new_data["features_1"] = self.features[id_1]
        new_data["features_2"] = self.features[id_2]

        n1, m1 = (self.gn[id_1], self.gm[id_1])
        n2, m2 = (self.gn[id_2], self.gm[id_2])
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["ged"] = gt_ged

        higher_bound = max(n1, n2) + max(m1, m2)
        self.hb = max(n1, n2) + max(m1, m2)
        new_data["hb"] = higher_bound
        new_data["target"] = torch.tensor([gt_ged / higher_bound]).float()
        new_data["gt_ged"] = torch.tensor(gt_ged)

        return new_data

    def save(self, epoch):
        torch.save(self.model.state_dict(), "model_save/" + "ADIS" + '_' + str(epoch))
        
    def load(self, epoch):
        self.model.load_state_dict(
            torch.load("model_save/" + "ADIS" + '_' + str(epoch)))