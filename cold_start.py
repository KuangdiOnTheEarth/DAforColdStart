import argparse
import os
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True)
parser.add_argument('--raw_dataset')
parser.add_argument('--cs_dataset')

def sequence_cut(sequence, csi_list):
    '''
    Cut the sequence from the head to the first cold-start items
    This function is used to generate the mixed-cold-start samples,
    whose length is shorter than the user-cold-start threshold, and contain one and only one cold-start item at tail.
    '''
    # i = 0  # index of the second last cold-start item
    # j = len(sequence) - 1  # index of the last cold-start item
    # # the sequence will be cut between [i+1, j] (both sides included)
    # while sequence[j] not in csi_list: j -= 1
    # i = j - 1
    # while i >= 0 and sequence[i] not in csi_list: i -= 1
    j = 0
    while sequence[j] not in csi_list: j += 1
    return sequence[0:j + 1]

def get_cold_user_item(dataset, cs_user_prop=0.2, cs_item_prop=0.2, ucs_max_len_prop=0.2):
    user_count = {}  # number of interactions for each user
    item_count = {}  # number of interactions on each item
    User = defaultdict(list)  # the list of interacted items for each user
    Item = defaultdict(list)  # the list of users who interacted with each item
    ucs_seq = {}  # test set for evaluating user cold-start
    ics_seq = {}  # test set for evaluating item cold-start
    mcs_seq = {}  # test set for evaluating mixed cold-start cases: cold-start user interacted with cold-start items
    ws_seq = {}   # test set for evaluating warm-start cases

    # f = open('data/%s.txt' % fname, 'r')
    # for line in f:
    #     u, j = line.rstrip().split(' ')
    #     u = int(u)
    #     j = int(j)
    #     User[u].append(j)
    #     Item[j].append(u)

    f = open('data/%s/formatted/sequences.txt' % dataset, 'r')
    for line in f:
        line_seg = line.rstrip().split(' ')
        u = int(line_seg[0])
        for j in line_seg[1:]:
            User[u].append(int(j))
            Item[int(j)].append(u)

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

    ucs_len_threshold = len(User[cs_user_list[-1]])
    ucs_max_len = int(ucs_max_len_prop * ucs_len_threshold)
    # find out training samples that are both user & item cold-start
    # cut these sequence so they only keep one cold-start item at the tail
    mcs_max, mcs_min = 0, 10 ** 6
    cs_item_seq_set = set()
    for iid in cs_item_list:
        user_set = set(Item[iid])
        cs_item_seq_set = cs_item_seq_set.union(user_set)
    mixed_set = set(cs_user_list).intersection(cs_item_seq_set)
    for uid in mixed_set:
        seq = User.pop(uid)
        mcs_seq[uid] = sequence_cut(seq, cs_item_list)
        mcs_max, mcs_min = max(mcs_max, len(mcs_seq[uid])), min(mcs_min, len(mcs_seq[uid]))

    # collect the cold-start users (with least iterations)
    ucs_max, ucs_min = 0, 10 ** 6
    for uid in set(cs_user_list).difference(mixed_set):
        # ucs_seq[uid] = User.pop(uid)
        temp_seq = User.pop(uid)
        ucs_seq[uid] = temp_seq[0:ucs_max_len] if ucs_max < len(temp_seq) else temp_seq
        ucs_max, ucs_min = max(ucs_max, len(ucs_seq[uid])), min(ucs_min, len(ucs_seq[uid]))

    # collect the cold-start items (with least iterations)
    ics_max, ics_min = 0, 10**6
    for uid in cs_item_seq_set.difference(mixed_set):
        # sequence cut-off after the last cold-start item -> allow test on cold-start item
        seq = User.pop(uid)
        seq_cut = sequence_cut(seq, cs_item_list)
        if len(seq_cut) > ucs_max:     # all item-cold-start sample should be non-user-cold-start
            ics_seq[uid] = seq_cut
            ics_max, ics_min = max(ics_max, len(ics_seq[uid])), min(ics_min, len(ics_seq[uid]))
        else:
            mcs_seq[uid] = seq_cut

    # the remaining sequences are for warm-start case
    ws_seq = User

    print("-- Dataset split finished --")
    print("Warm-Start:       %d samples" % len(ws_seq))
    print("User-Cold-Start:  %d samples, length between %d ~ %d" % (len(ucs_seq), ucs_min, ucs_max))
    print("Item-Cold-Start:  %d samples, length between %d ~ %d" % (len(ics_seq), ics_min, ics_max))
    print("Mixed-Cold-Start: %d samples, length between %d ~ %d" % (len(mcs_seq), mcs_min, mcs_max))

    # print("cold-start users: " + str(len(cs_user_list)))
    # print("without overlap (exclude mixed cases): " + str(len(ucs_seq)) +
    #       "; sequence length between: " + str(ucs_min) + " ~ " + str(ucs_max))
    #
    # print("\nItems:")
    # print("total items:" + str(num_item) +
    #       "; # interacted users between: " + str(len(Item[sorted_item_list[0]])) + " ~ " + str(len(Item[sorted_item_list[-1]])))
    # print("cold-start items: " + str(len(cs_item_list)) +
    #       "; # interacted users between: " + str(len(Item[cs_item_list[0]])) + " ~ " + str(len(Item[cs_item_list[-1]])))
    # print("# sequences for item-cold-start: " + str(len(cs_item_seq_set)))
    # print("without overlap (exclude mixed cases): " + str(len(ics_seq)) + "; (" + str(discard_count) + " samples discarded" +
    #       "; sequences length between: " + str(ics_min) + " ~ " + str(ics_max))
    #
    # print("\nMixed:")
    # print("samples that are both cold-start user and item: " + str(len(mixed_set)))

    # writing the split data sets into files in same folder
    # each line in the format: sample_id user_id item_id
    sample_id = 0
    file_list = {
        "ws": ws_seq, "ucs": ucs_seq, "ics": ics_seq, "mcs": mcs_seq
    }
    # the ws and ucs must be placed at the beginning,
    # as they will be used in model training, where the sampler required continuous sid starting from 1

    directory = os.path.join('data', dataset, "splits")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for name, dataset in file_list.items():
        f = open('%s/%s.txt' % (directory, name), 'w')
        for uid, item_list in dataset.items():
            sample_id += 1
            line_str = str(sample_id) + ' ' + str(uid)
            sequence_str = ''
            for iid in item_list:
                sequence_str += (' ' + str(iid))
            f.write('%s%s\n' % (line_str, sequence_str))
        f.close()

    # write the list of cold-start items into file
    f = open('%s/%s.txt' % (directory, 'cs_item_list'), 'w')
    for iid in cs_item_list:
        f.write('%d\n' % iid)
    f.close()

    # print the meta information into file
    f = open('%s/%s.txt' % (directory, 'meta'), 'w')
    f.write("Original dataset: %d users and %d items\n" % (num_user, num_item))
    f.write("-------------------------------------------------\n")
    f.write("Cold-Start User Proportion: %f\n" % cs_user_prop)
    f.write("Cold-Start Item Proportion: %f\n" % cs_item_prop)
    f.write("-------------------------------------------------\n")
    f.write("Warm-Start:\t\t\t%d samples\n" % len(ws_seq))
    f.write("User-Cold-Start:\t%d samples, length between %d ~ %d\n" % (len(ucs_seq), ucs_min, ucs_max))
    f.write("Item-Cold-Start:\t%d samples, length between %d ~ %d\n" % (len(ics_seq), ics_min, ics_max))
    f.write("Mixed-Cold-Start:\t%d samples, length between %d ~ %d\n" % (len(mcs_seq), mcs_min, mcs_max))
    f.close()

if __name__ == '__main__':
    get_cold_user_item("ml-1m", cs_user_prop=0.2, cs_item_prop=0.2, ucs_max_len_prop=0.2)
    # get_cold_user_item("Steam", cs_user_prop=0.2, cs_item_prop=0.2, ucs_max_len_prop=0.2)
    # get_cold_user_item("Video", cs_user_prop=0.2, cs_item_prop=0.2, ucs_max_len_prop=0.2)