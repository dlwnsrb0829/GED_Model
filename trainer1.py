import sys
import time
from typing import List

import dgl
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_all_graphs, load_labels, load_ged

from models import GedGNN
from GedMatrix import fixed_mapping_loss

class Trainer(object):
    def __init__(self):
        self.to_torch_time = 0.0
        self.results = []
        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        self.init_graph_pairs()
        self.setup_model()
        # print(self.mapping[0][1])

    def setup_model(self):
        weight = 10.0 # AIDS, LINUX weight
        self.model = GedGNN(weight, self.number_of_labels)
        # for param in self.model.parameters():
        #     print(param.requires_grad)

    def load_data(self):
        dataset_name = "AIDS"
        self.train_num, self.val_num, self.test_num, self.graphs = load_all_graphs(dataset_name)
        print("Load {} graphs. ({} for training)".format(len(self.graphs), self.train_num))
        
        self.number_of_labels = 0
        if dataset_name in ['AIDS']:
            self.global_labels, self.features = load_labels(dataset_name)
            self.number_of_labels = len(self.global_labels)
            # print(self.number_of_labels)
        
        ged_dict = dict()
        # We could load ged info from several files.
        # load_ged(ged_dict, self.args.abs_path, dataset_name, 'xxx.json')
        load_ged(ged_dict, dataset_name, 'TaGED.json')
        self.ged_dict = ged_dict
        print("Load ged dict.")

    def transfer_data_to_torch(self):
        t1 = time.time()
        self.edge_index = []
        for g in self.graphs:
            edge = g['graph']
            edge = edge + [[y, x] for x, y in edge]
#             edge = edge + [[x, x] for x in range(g['n'])]
            edge = torch.tensor(edge).t().long()
            self.edge_index.append(edge)

        self.features = [torch.tensor(x).float() for x in self.features]
        print("Feature shape of 1st graph:", self.features[0].shape)

        n = len(self.graphs)
        mapping = [[None for i in range(n)] for j in range(n)]
        ged = [[(0., 0., 0., 0.) for i in range(n)] for j in range(n)]
        gid = [g['gid'] for g in self.graphs]
        self.gid = gid
        self.gn = [g['n'] for g in self.graphs]
        self.gm = [g['m'] for g in self.graphs]
        for i in range(n):
            mapping[i][i] = torch.eye(self.gn[i], dtype=torch.float)
            for j in range(i + 1, n):
                id_pair = (gid[i], gid[j])
                n1, n2 = self.gn[i], self.gn[j]
                if id_pair not in self.ged_dict:
                    id_pair = (gid[j], gid[i])
                    n1, n2 = n2, n1
                if id_pair not in self.ged_dict:
                    ged[i][j] = ged[j][i] = None
                    mapping[i][j] = mapping[j][i] = None
                else:
                    ta_ged, gt_mappings = self.ged_dict[id_pair]
                    ged[i][j] = ged[j][i] = ta_ged
                    mapping_list = [[0 for y in range(n2)] for x in range(n1)]
                    for gt_mapping in gt_mappings:
                        for x, y in enumerate(gt_mapping):
                            mapping_list[x][y] = 1
                    mapping_matrix = torch.tensor(mapping_list).float()
                    mapping[i][j] = mapping[j][i] = mapping_matrix
        self.ged = ged
        self.mapping = mapping

        t2 = time.time()
        self.to_torch_time = t2 - t1


    def check_pair(self, i, j):
        if i == j:
            return (0, i, j)
        id1, id2 = self.gid[i], self.gid[j]
        if (id1, id2) in self.ged_dict:
            return (0, i, j)
        elif (id2, id1) in self.ged_dict:
            return (0, j, i)
        else:
            return None

    def init_graph_pairs(self):
#         random.seed(1)

        self.training_graphs = []
        self.val_graphs = []
        self.testing_graphs = []
        self.testing2_graphs = []

        train_num = self.train_num
        val_num = train_num + self.val_num
        test_num = len(self.graphs)

        dg = self.delta_graphs
        for i in range(train_num):
            if self.gn[i] <= 10:
                for j in range(i, train_num):
                    tmp = self.check_pair(i, j)
                    if tmp is not None:
                        self.training_graphs.append(tmp)
            elif dg[i] is not None:
                k = len(dg[i])
                for j in range(k):
                    self.training_graphs.append((1, i, j))

        li = []
        for i in range(train_num):
            if self.gn[i] <= 10:
                li.append(i)

        for i in range(train_num, val_num):
            if self.gn[i] <= 10:
#                 random.shuffle(li)
                self.val_graphs.append((0, i, li[:100])) # args num_testing_graphs
            elif dg[i] is not None:
                k = len(dg[i])
                self.val_graphs.append((1, i, list(range(k))))

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
#                 random.shuffle(li)
                self.testing_graphs.append((0, i, li[:100])) # args num_testing_graphs
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing_graphs.append((1, i, list(range(k))))

        li = []
        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
                li.append(i)

        for i in range(val_num, test_num):
            if self.gn[i] <= 10:
#                 random.shuffle(li)
                self.testing2_graphs.append((0, i, li[:100])) # args num_testing_graphs
            elif dg[i] is not None:
                k = len(dg[i])
                self.testing2_graphs.append((1, i, list(range(k))))

        # print(self.training_graphs)
        # print(train_num, val_num, test_num)
        print("Generate {} training graph pairs.".format(len(self.training_graphs)))
        print("Generate {} * {} val graph pairs.".format(len(self.val_graphs), 100))
        print("Generate {} * {} testing graph pairs.".format(len(self.testing_graphs), 100))
        print("Generate {} * {} testing2 graph pairs.".format(len(self.testing2_graphs), 100))

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")
        t1 = time.time()
#         self.optimizer = torch.optim.Adam(self.model.parameters(),
#                                           lr=0.003,
#                                           weight_decay=5*10**-3)
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
                    batch_total_loss = self.process_batch(batch)  # without average
                    # print(self.optimizer.param_groups[0]['lr'])
                    loss_sum += batch_total_loss
                    main_index += len(batch)
                    loss = loss_sum / main_index  # the average loss of current epoch
                    pbar.update(len(batch))
#                     pbar.set_description(
#                         "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.cur_epoch + 1, loss,
#                                                                        index,
#                                                                        batch_total_loss / len(batch)))
                    pbar.set_description(
                        "Epoch_{}: loss={} - Batch_{}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3),
                                                                       index,
                                                                       round(1000 * batch_total_loss / len(batch), 3)))
                tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3)))
                training_loss = round(1000 * loss, 3)
#                 tqdm.write("Epoch {}: loss={}".format(self.cur_epoch + 1, loss))
#                 training_loss = loss
        t2 = time.time()
        training_time = t2 - t1
        # if len(self.values) > 0:
        #     self.prediction_analysis(self.values, "training_score")

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

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), 128):
            batches.append(self.training_graphs[graph:graph + 128])
        return batches

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float()

        weight = 1.0
        for graph_pair in batch:
            data = self.pack_graph_pair(graph_pair)
            target, gt_mapping = data["target"], data["mapping"]
            gt_ged = data["gt_ged"]
            # print(gt_mapping)
            # prediction, _, mapping = self.model(data)
            pcost, cost = self.model(data)
#             print(gt_ged, pcost)
            # mapping = torch.tensor(mapping)
            # mapping.requires_grad_(True)
            # print(torch.tensor(mapping))
            
            # print(emb1, emb2)
#             losses = losses + fixed_mapping_loss(torch.transpose(mapping, 0, 1), gt_mapping)
#             print(target, cost)
#             cost = (cost/self.hb)*10
#             print(gt_ged, cost)

            losses = losses + 10 * F.mse_loss(target, cost)
#             losses = losses + 10 * F.mse_loss(target, cost) + fixed_mapping_loss(mapping, gt_mapping)
#             losses = losses + F.mse_loss(gt_ged.to(torch.float32), cost) # 이거 나중에 주석 해제
#             losses = losses + fixed_mapping_loss(mapping, gt_mapping)
        #     if self.args.finetune:
        #         if self.args.target_mode == "linear":
        #             losses = losses + F.relu(target - prediction)
        #         else: # "exp"
        #             losses = losses + F.relu(prediction - target)
        losses.requires_grad_(True)
        losses.backward()
        self.optimizer.step()
        return losses.item()
    def pack_graph_pair(self, graph_pair):
        """
        Prepare the graph pair data for GedGNN model.
        :param graph_pair: (pair_type, id_1, id_2)
        :return new_data: Dictionary of Torch Tensors.
        """
        new_data = dict()

        (pair_type, id_1, id_2) = graph_pair
        if pair_type == 0:  # normal case
            gid_pair = (self.gid[id_1], self.gid[id_2])
            if gid_pair not in self.ged_dict:
                id_1, id_2 = (id_2, id_1)

            real_ged = self.ged[id_1][id_2][0]
            ta_ged = self.ged[id_1][id_2][1:]

            new_data["id_1"] = id_1
            new_data["id_2"] = id_2

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = self.edge_index[id_2]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = self.features[id_2]

            new_data["mapping"] = self.mapping[id_1][id_2]
        elif pair_type == 1:  # delta graphs
            new_data["id"] = id_1
            dg: dict = self.delta_graphs[id_1][id_2]

            real_ged = dg["ta_ged"][0]
            ta_ged = dg["ta_ged"][1:]

            new_data["edge_index_1"] = self.edge_index[id_1]
            new_data["edge_index_2"] = dg["edge_index"]
            new_data["features_1"] = self.features[id_1]
            new_data["features_2"] = dg["features"]

            new_data["mapping"] = dg["mapping"]
        else:
            assert False

        n1, m1 = (self.gn[id_1], self.gm[id_1])
        n2, m2 = (self.gn[id_2], self.gm[id_2]) if pair_type == 0 else (dg["n"], dg["m"])
        new_data["n1"] = n1
        new_data["n2"] = n2
        new_data["ged"] = real_ged
        # new_data["ta_ged"] = ta_ged

        higher_bound = max(n1, n2) + max(m1, m2)
        self.hb = max(n1, n2) + max(m1, m2) # 이거 추가
        new_data["hb"] = higher_bound
        new_data["target"] = torch.tensor([real_ged / higher_bound]).float()
        new_data["ta_ged"] = (torch.tensor(ta_ged).float() / higher_bound)
        new_data["gt_ged"] = torch.tensor(real_ged)

        return new_data
    
    def score(self, testing_graph_set='test', test_k=0):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation on {} set.\n".format(testing_graph_set))
        if testing_graph_set == 'test':
            testing_graphs = self.testing_graphs
        elif testing_graph_set == 'test2':
            testing_graphs = self.testing2_graphs
        elif testing_graph_set == 'val':
            testing_graphs = self.val_graphs
        else:
            assert False

        self.model.eval()
        # self.model.train()

        num = 0  # total testing number
        time_usage = []
        mse = []  # score mse
        mae = []  # ged mae
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
#         rho = []
#         tau = []
#         pk10 = []
#         pk20 = []

        for pair_type, i, j_list in tqdm(testing_graphs, file=sys.stdout):
            pre = []
            gt = []
            t1 = time.time()
            for j in j_list:
                data = self.pack_graph_pair((pair_type, i, j))
                target, gt_ged = data["target"].item(), data["gt_ged"] # 이거 나중에 주석
#                 gt_ged = data["gt_ged"] # 이건 해제
#                 model_out = self.model(data) if test_k == 0 else self.test_matching(data, test_k)
#                 prediction, pre_ged = model_out[0], model_out[1]
                pcost, cost = self.model(data)
                round_pre_ged = round(pcost.item())

                num += 1

#                 mse.append(F.mse_loss(gt_ged.to(torch.float32), cost).item()) # 이거 나중에 주석 해제
                mse.append((cost.item() - target) ** 2) # 이거 나중에 주석처리
#                 pre.append(cost)
#                 gt.append(gt_ged)

                mae.append(abs(round_pre_ged - gt_ged))
#                 print(round_pre_ged, gt_ged)
                if round_pre_ged == gt_ged:
                    num_acc += 1
                    num_fea += 1
                elif round_pre_ged > gt_ged:
                    num_fea += 1
            t2 = time.time()
            time_usage.append(t2 - t1)
#             rho.append(spearmanr(pre, gt)[0])
#             tau.append(kendalltau(pre, gt)[0])
#             pk10.append(self.cal_pk(10, pre, gt))
#             pk20.append(self.cal_pk(20, pre, gt))

        time_usage = round(np.mean(time_usage), 3)
        mse = round(np.mean(mse)*1000, 3)
        mae = round(np.mean(mae), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
#         rho = round(np.mean(rho), 3)
#         tau = round(np.mean(tau), 3)
#         pk10 = round(np.mean(pk10), 3)
#         pk20 = round(np.mean(pk20), 3)
        self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mse', 'mae', 'acc',
                             'fea'))
        self.results.append(('model', 'ADIS', testing_graph_set, num, time_usage, mse, mae, acc,
                             fea))

#         self.results.append(('model_name', 'dataset', 'graph_set', '#testing_pairs', 'time_usage(s/100p)', 'mse', 'mae', 'acc',
#                              'fea', 'rho', 'tau', 'pk10', 'pk20'))
#         self.results.append((self.args.model_name, self.args.dataset, testing_graph_set, num, time_usage, mse, mae, acc,
#                              fea, rho, tau, pk10, pk20))

        print(*self.results[-2], sep='\t')
        print(*self.results[-1], sep='\t')
        
        with open('result/' + 'results.txt', 'a') as f:
            if test_k == 0:
                print("## Testing", file=f)
            else:
                print("## Post-processing", file=f)
            print("```", file=f)
            print(*self.results[-2], sep='\t', file=f)
            print(*self.results[-1], sep='\t', file=f)
            print("```\n", file=f)

    def save(self, epoch):
        torch.save(self.model.state_dict(), "model_save/" + "ADIS" + '_' + str(epoch))
        
    def load(self, epoch):
        self.model.load_state_dict(
            torch.load("model_save/" + "ADIS" + '_' + str(epoch)))
