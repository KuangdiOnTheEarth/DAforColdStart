
# train/val/test data generation
import os
import sys
import copy
import torch
import random
from collections import defaultdict

import numpy as np

from SASRec_pytorch.utils import evaluate


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
    ics_test = {}
    # the cold-start items should not appear in training set, so their model input for evaluation is stored separately
    # mcs_train = {}
    # mcs_valid = {}
    mcs_test = {}

    train_map = {'da': train, 'ws': train, 'ucs': train, 'ics': train, 'mcs': train}
    valid_map = {'da': valid, 'ws': valid, 'ucs': valid, 'ics': valid, 'mcs': valid}
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
            if len(items) < 3: # if there are less than 3 items, use them as test set
                # train_map[set_name][sid] = items
                # valid_map[set_name][sid] = []
                # test_map[set_name][sid] = []
                test_map[set_name][sid] = []
                test_map[set_name][sid].append(items[-1])
                valid_map[set_name][sid] = []
                if len(items) == 2:
                    valid_map[set_name][sid].append(items[-2])
                train_map[set_name][sid] = []
            else:
                train_map[set_name][sid] = items[:-2]
                valid_map[set_name][sid] = []
                valid_map[set_name][sid].append(items[-2])
                test_map[set_name][sid] = []
                test_map[set_name][sid].append(items[-1])
    trainsamplenum = len(train)

    return [train_map, valid_map, test_map, trainsamplenum, itemnum]


def cs_evaluate(model, dataset, args):
    if not args.cold_start:
        print("Inappropriate use: cs_evaluate should only be called in cold_start mode")
        print("redirect to evaluate() function used for normal case ...")
        return evaluate(model, dataset, args)

    [train_map, valid_map, test_map, samplenum, itemnum] = copy.deepcopy(dataset)

    NDCG_map = {}
    HR_map = {}
    total_samples = 0 # used to calculate avg NDCG and HR
    avg_NDCG_numerator = 0 # used to calculate avg NDCG and HR
    avg_HR_numerator = 0 # used to calculate avg NDCG and HR

    for set_name in ['ws', 'ucs', 'ics', 'mcs']:
        train = train_map[set_name]
        valid = valid_map[set_name]
        test = test_map[set_name]
        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0

        sid_list = list(test.keys())
        samplenum = len(test)
        if samplenum>10000:
            index_list = random.sample(range(0, samplenum), 10000)
            temp = []
            for idx in index_list:
                temp.append(sid_list[idx])
            sid_list = temp

        for sid in sid_list:
            if len(test[sid]) < 1: continue
            # if (len(train[sid]) + len(valid[sid])) < 1 or len(test[sid]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[sid][0] if len(valid[sid]) > 0 else 0
            idx -= 1

            # when evaluating user cold-start, only use a fixed-number of items as model input sequence
            # the real # input item is (ucs_input_num + 1), since the validation item always exists in input
            # ucs_input_num = 4
            for i in reversed(train[sid]):
                # if set_name == 'ucs' and ucs_input_num <= 0: break
                # ucs_input_num -= 1
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(train[sid])
            rated.add(0)
            item_idx = [test[sid][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            predictions = -model.predict(*[np.array(l) for l in [[sid], [seq], item_idx]])
            predictions = predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

        NDCG_map[set_name] = NDCG / valid_user
        HR_map[set_name] = HT / valid_user

        avg_NDCG_numerator += NDCG_map[set_name] * valid_user
        avg_HR_numerator += HR_map[set_name] * valid_user
        total_samples += valid_user

    # calculate the weighted average over all samples across test sets
    NDCG_map['avg'] = avg_NDCG_numerator / total_samples
    HR_map['avg'] = avg_HR_numerator / total_samples

    return NDCG_map, HR_map