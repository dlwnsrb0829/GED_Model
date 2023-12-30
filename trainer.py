

class Trainer(object):
    def __init__(self):
        self.load_data()
        self.transfer_data_to_torch()
        self.delta_graphs = [None] * len(self.graphs)
        self.init_graph_pairs()
        self.setup_model()
        
    def load_data(self):
        dataset_name = "AIDS"
        self.graph_num, self.graphs = load_all_graphs(dataset_name)
        print("Load {} graphs.".format(self.graph_num))
        
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