import os.path
from collections import defaultdict


def load_data(dataset_name):
    dataset = {}
    files = ['ws', 'ucs', 'ics', 'mcs']
    for file in files:
        file_path = 'data/%s/splits/%s.txt' % (dataset_name, file)
        for line in open(file_path):
            line_data = line.rstrip().split(' ')
            line_data = list(map(int, line_data))
            sid = line_data[0]
            items = line_data[2:]
            # remove the last two validation and testing items, use remaining items as training data
            if len(items) > 2:
                seq = items[0:-2]
            else:
                seq = []
            dataset[sid] = seq
    return dataset

def sequence2str(sequence):
    if len(sequence) == 0: return ''
    res = str(sequence[0])
    for i in range(1, len(sequence)):
        res += (' ' + str(sequence[i]))
    return res

def load_similar_items(file_path):
    similar_map = {}
    # file_path = 'data/%s/splits/similar_items.txt' % dataset_name
    for line in open(file_path):
        line_data = line.rstrip().split(' ')
        line_data = list(map(int, line_data))
        similar_map[line_data[0]] = line_data[1:]
    return similar_map

def find_similar_appearance(training, cs_similar_map):
    '''
    find all appearance of similar items to all cold-start items
    :param training: a map of sid->sequence for training set
    :param cs_similar_map: a map of cs_iid->[similar_ids]
    :return: a map from similar item to all appearances: similar_item_id->[(sid, idx)]
    '''
    similar_ids = set([])
    print(len(cs_similar_map.values()))
    for triple in list(cs_similar_map.values()):
        similar_ids = similar_ids.union(set(triple))
    appearances = defaultdict(list)
    for sid, sequence in training.items():
        for idx, iid in enumerate(sequence):
            if iid in similar_ids:
                appearances[iid].append((sid, idx))
    return appearances
