import sys
import time
import torch
import torch.nn.functional as F
from utils import load_all_graphs, load_labels, load_ged
from model import GedGNN

class Trainer(object):
    def __init__(self):
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
        if i == j:
            return (i, j)
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
        for graph in range(0, len(self.training_graphs), 128):
            batches.append(self.training_graphs[graph:graph + 128])
        return batches

t = Trainer()
