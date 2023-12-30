import json

def load_all_graphs(dataset_name):
    graphs = iterate_get_graphs("dataset/" + dataset_name + "/" + dataset_name, "json")
    graph_num = len(graphs)
    
    return graph_num, graphs

def iterate_get_graphs(dir, file_format):
    graphs = []
    data = json.load(open(dir + "." + file_format, 'r'))
    for g in range(len(data)):
        graphs.append(data[g])
    
    return graphs

def load_labels(dataset_name):
    features = iterate_get_graphs("dataset/" + dataset_name + "/" + "onehot", "json")
    labels_num = len(features[0]['onehot'][0])
    print('Load one-hot label features (dim = {}) of {}.'.format(labels_num, dataset_name))
    
    return labels_num, features

def load_ged(dataset_name='AIDS', file_name='pair_ged.json'):
    ged_dict = dict()
    path = "dataset/{}/{}".format(dataset_name, file_name)
    pair_ged = json.load(open(path, 'r'))
    for (id_1, id_2, ged) in pair_ged:
        ged_dict[(id_1, id_2)] = ged
    
    return ged_dict

a, b = load_all_graphs('AIDS')
print(b[0]['g_num'])

c, d = load_labels('AIDS')
print(c)
# print(d[0])

e = load_ged()
print(e)