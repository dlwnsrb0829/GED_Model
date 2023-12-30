from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx
import json
from texttable import Texttable

def load_all_graphs(dataset_name):
    graphs = iterate_get_graphs("dataset/" + dataset_name, "json")
    train_num = len(graphs)
    graphs += iterate_get_graphs("json_data/" + dataset_name + "/test", "json")
    test_num = len(graphs) - train_num
    val_num = test_num
    train_num -= val_num
    return train_num, val_num, test_num, graphs

def iterate_get_graphs(dir, file_format):
    graphs = []
    for file in get_file_paths(dir, file_format):
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
#             print(g)
#             exit()
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
    
    return graphs

def load_labels(dataset_name):
    path = "json_data/" + dataset_name + "/labels.json"
    global_labels = json.load(open(path, 'r'))
    features = iterate_get_graphs("json_data/" + dataset_name + "/train", "onehot") \
             + iterate_get_graphs("json_data/" + dataset_name + "/test", "onehot")
    print('Load one-hot label features (dim = {}) of {}.'.format(len(global_labels), dataset_name))
    return global_labels, features

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def load_ged(ged_dict, dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    path = "json_data/{}/{}".format(dataset_name, file_name)
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)