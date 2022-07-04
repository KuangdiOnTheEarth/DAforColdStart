
# train/val/test data generation
from collections import defaultdict


def cs_data_partition(folder_name):
    samplenum = 0
    itemnum = 0
    # the model training is based on the samples from warm-start and user-cold-start sets
    # with the last item stored in `*_test`, second last item stored in `valid`, prefix sequence stored in `train`
    train = {}
    valid = {}
    ws_test = {}
    ucs_test = {}
    # the cold-start items should not appear in training set, so their model input for evaluation is stored separately
    ics_train = {}
    ics_valid = {}
    ics_test = {}
    mcs_test = {}

    # container_map = {'ws': WS_Sample, 'ics': ICS_Sample, 'ucs': UCS_Sample, 'mcs': MCS_Sample}
    train_map = {'ws': train, 'ucs': train, 'ics': ics_train, 'mcs': ics_train}
    valid_map = {'ws': valid, 'ucs': valid, 'ics': ics_valid, 'mcs': ics_valid}
    test_map = {'ws': ws_test, 'ucs': ucs_test, 'ics': ics_test, 'mcs': mcs_test}

    for set_name, test_set in test_map.items():
        f = open('data/%s/%s.txt' % (folder_name, set_name), 'r')
        for line in f:
            line_data = line.rstrip().split(' ')
            line_data = list(map(int, line_data))
            sid = int(line_data[0])
            items = line_data[2:]
            samplenum = max(sid, samplenum)
            itemnum = max(max(items), itemnum)
            if len(items) < 3:
                train_map[set_name][sid] = items
                valid_map[set_name][sid] = []
                test_set[set_name][sid] = []
            else:
                train_map[set_name][sid] = items[:-2]
                valid_map[set_name][sid] = []
                valid_map[set_name][sid].append(items[-2])
                test_set[set_name][sid] = []
                test_set[set_name][sid].append(items[-1])

    return [train_map, valid_map, test_map, samplenum, itemnum]