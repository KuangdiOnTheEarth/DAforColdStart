
# train/val/test data generation
import os
import sys
import copy
import torch
import random
from collections import defaultdict

import numpy as np


def cs_data_partition(folder_name):
    trainsamplenum = 0
    # the number of samples for model training, i.e. all sequences in warm-start and user-cold-start sets
    itemnum = 0
    # the model training is based on the samples from warm-start and user-cold-start sets
    # with the last item stored in `*_test`, second last item stored in `valid`, prefix sequence stored in `train`
    train = {}
    valid = {}
    ws_test = {}  # the augmented samples are considered as normal training samples, i.e. same as ws samples
    ucs_test = {}
    # the cold-start items should not appear in training set, so their model input for evaluation is stored separately
    ics_train = {}
    ics_valid = {}
    ics_test = {}
    mcs_test = {}

    train_map = {'da': train, 'ws': train, 'ucs': train, 'ics': ics_train, 'mcs': ics_train}
    valid_map = {'da': valid, 'ws': valid, 'ucs': valid, 'ics': ics_valid, 'mcs': ics_valid}
    test_map = {'da': ws_test, 'ws': ws_test, 'ucs': ucs_test, 'ics': ics_test, 'mcs': mcs_test}

    for set_name, test_set in test_map.items():
        file_path = 'data/%s/%s.txt' % (folder_name, set_name)
        # in case data augmentation set not exists
        if not os.path.exists(file_path):
            print("No data augmentation is detected")
            continue
        f = open(file_path, 'r')
        print("Reading %s set ..." % set_name)
        for line in f:
            line_data = line.rstrip().split(' ')
            line_data = list(map(int, line_data))
            sid = int(line_data[0])
            items = line_data[2:]
            # if set_name == 'ws' or set_name == 'ucs':  # only count the maximum sid in ws and ucs sets
            #     trainsamplenum = max(sid, trainsamplenum)
            itemnum = max(max(items), itemnum)
            if len(items) < 3:
                train_map[set_name][sid] = items
                valid_map[set_name][sid] = []
                test_map[set_name][sid] = []
            else:
                train_map[set_name][sid] = items[:-2]
                valid_map[set_name][sid] = []
                valid_map[set_name][sid].append(items[-2])
                test_map[set_name][sid] = []
                test_map[set_name][sid].append(items[-1])
    trainsamplenum = len(train)

    return [train_map, valid_map, test_map, trainsamplenum, itemnum]


def cs_evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    if args.cold_start:  # in cold-start mode, needs to additionally fetch dataset from dataset map
        train = train['ws']
        valid = valid['ws']
        test = test['ws']
        usernum = len(test) # the input `usernum`

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user