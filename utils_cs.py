
# train/val/test data generation
from collections import defaultdict


def cs_data_partition(folder_name):
    samplenum = 0
    itemnum = 0
    WS_Sample = defaultdict(list)
    ICS_Sample = defaultdict(list)
    UCS_Sample = defaultdict(list)
    MCS_Sample = defaultdict(list)
    train = {}
    valid = {}
    ws_test = {}
    ics_test = {}
    ucs_test = {}
    mcs_test = {}
    container_map = {'ws': WS_Sample, 'ics': ICS_Sample, 'ucs': UCS_Sample, 'mcs': MCS_Sample}
    test_map = {'ws': ws_test, 'ics': ics_test, 'ucs': ucs_test, 'mcs': mcs_test}

    # read warm-start samples
    for file_name, container in container_map.items():
        f = open('data/%s/%s.txt' % (folder_name, file_name), 'r')
        for line in f:
            s, u, i = line.rstrip().split(' ')
            s = int(s)
            i = int(i)
            samplenum = max(u, samplenum)
            itemnum = max(i, itemnum)
            container[s].append(i)

    for set_name, container in container_map.items():
        for sid, item_list in container.items():
            nfeedback = len(item_list)
            if nfeedback < 3:
                train[sid] = item_list
                valid[sid] = []
                ws_test[sid] = []
                ics_test[sid] = []
                ucs_test[sid] = []
                mcs_test[sid] = []
            else:
                other_sets = set(test_map.keys()).difference(set(set_name))
                train[sid] = item_list[:-2]
                valid[sid] = []
                valid[sid].append(item_list[-2])

                for other in other_sets:
                    other[sid] = []

    for sample in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]