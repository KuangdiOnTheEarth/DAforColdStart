import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True)
parser.add_argument('--raw_dataset')
parser.add_argument('--cs_dataset')

def get_cold_user_item(fname, cs_user_prop=0.2, cs_item_prop=0.2):
    user_count = {}  # number of interactions for each user
    item_count = {}  # number of interactions on each item
    User = defaultdict(list)  # the list of interacted items for each user
    Item = defaultdict(list)  # the list of users who interacted with each item
    ucs_seq = {}  # test set for evaluating user cold-start
    ics_seq = {}  # test set for evaluating item cold-start
    mcs_seq = {}  # test set for evaluating mixed cold-start cases: cold-start user interacted with cold-start items
    ws_seq = {}   # test set for evaluating warm-start cases

    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        User[u].append(i)
        Item[i].append(u)
    num_user = len(User)
    num_item = len(Item)

    sorted_user_list = sorted(User, key=lambda key: len(User[key]))  # user-ids sorted by # interactions from he/she
    sorted_item_list = sorted(Item, key=lambda key: len(Item[key]))  # item-ids sorted by # interactions on it
    cs_user_list = sorted_user_list[:int(num_user * cs_user_prop)]
    cs_item_list = sorted_item_list[:int(num_item * cs_item_prop)]

    print("Users:")
    print("total users:" + str(num_user) +
          "; # interactions between: " + str(len(User[sorted_user_list[0]])) + " ~ " + str(
        len(User[sorted_user_list[-1]])))

    # find out training samples that are both user & item cold-start
    cs_item_seq_set = set()
    for iid in cs_item_list:
        user_set = set(Item[iid])
        cs_item_seq_set = cs_item_seq_set.union(user_set)
    mixed_set = set(cs_user_list).intersection(cs_item_seq_set)
    for uid in mixed_set:
        mcs_seq[uid] = User.pop(uid)
        if uid == 3822: print("3822 discovered")

    # collect the cold-start users (with least iterations)
    ucs_max, ucs_min = 0, 10 ** 6
    for uid in set(cs_user_list).difference(mixed_set):
        ucs_seq[uid] = User.pop(uid)
        ucs_max, ucs_min = max(ucs_max, len(ucs_seq[uid])), min(ucs_min, len(ucs_seq[uid]))

    # collect the cold-start items (with least iterations)
    ics_max, ics_min = 0, 10**6
    discard_count = 0
    for uid in cs_item_seq_set.difference(mixed_set):
        # sequence cut-off after the last cold-start item -> allow test on cold-start item
        seq = User.pop(uid)
        i = len(seq) - 1
        while seq[i] not in cs_item_list: i -= 1
        if i > ucs_max:     # all item-cold-start sample should be non-user-cold-start
            ics_seq[uid] = seq[0:i+1]
            ics_max, ics_min = max(ics_max, len(ics_seq[uid])), min(ics_min, len(ics_seq[uid]))
        else:
            discard_count += 1
    # the remaining sequences are for warm-start case
    ws_seq = User

    print("cold-start users: " + str(len(cs_user_list)))
    print("without overlap (exclude mixed cases): " + str(len(ucs_seq)) +
          "; sequence length between: " + str(ucs_min) + " ~ " + str(ucs_max))

    print("\nItems:")
    print("total items:" + str(num_item) +
          "; # interacted users between: " + str(len(Item[sorted_item_list[0]])) + " ~ " + str(len(Item[sorted_item_list[-1]])))
    print("cold-start items: " + str(len(cs_item_list)) +
          "; # interacted users between: " + str(len(Item[cs_item_list[0]])) + " ~ " + str(len(Item[cs_item_list[-1]])))
    print("# sequences for item-cold-start: " + str(len(cs_item_seq_set)))
    print("without overlap (exclude mixed cases): " + str(len(ics_seq)) + "; (" + str(discard_count) + " samples discarded" +
          "; sequences length between: " + str(ics_min) + " ~ " + str(ics_max))

    print("\nMixed:")
    print("samples that are both cold-start user and item: " + str(len(mixed_set)))

    # writing the split data sets into files in same folder
    # each line in the format: sample_id user_id item_id
    sample_id = 0
    file_list = {
        "ws": ws_seq, "ucs": ucs_seq, "ics": ics_seq, "mcs": mcs_seq
    }
    directory = 'data/' + fname
    if not os.path.exists(directory):
        os.makedirs(directory)
    for name, dataset in file_list.items():
        f = open('data/%s/%s.txt' % (fname, name), 'w')
        for uid, item_list in dataset.items():
            sample_id += 1
            for iid in item_list:
                f.write('%d %d %d\n' % (sample_id, uid, iid))
        f.close()


if __name__ == '__main__':
    get_cold_user_item("ml-1m", cs_user_prop=0.2, cs_item_prop=0.1)