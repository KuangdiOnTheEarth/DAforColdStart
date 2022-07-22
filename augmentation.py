import argparse
import math
import os.path
import random
from copy import deepcopy

import utils_da

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--method', required=True, type=str)
parser.add_argument('--percentage', default=0.2, type=float)

# settings for Sequence Split method
parser.add_argument('--max_len', default=30, type=int)

# settings for Random Split method
parser.add_argument('--cut_points', default=5, type=int)

# settings for Synonym Replacement method
parser.add_argument('--replace_percentage', default=0.1, type=float)
parser.add_argument('--max_replace', default=5, type=int)
parser.add_argument('--augmentation_file', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    da_methods = ['SeqSplit', 'RandSplit', 'SynRep', 'Mixed']
    if args.method not in da_methods:
        print("Illegal data augmentation method, select from " + str(da_methods))

    if args.method == 'SeqSplit':
        fname = '{}.da.{}.per={}.maxlen={}.txt'
        fname = fname.format(args.dataset, args.method, args.percentage, args.max_len)
    elif args.method == 'RandSplit':
        fname = '{}.da.{}.per={}.cut_points={}.txt'
        fname = fname.format(args.dataset, args.method, args.percentage, args.cut_points)
    elif args.method == 'SynRep':
        fname = '{}.da.{}.rep_per={}.max_rep={}.txt'
        fname = fname.format(args.dataset, args.method, args.replace_percentage, args.max_replace)
    output_path = os.path.join('data', args.dataset, 'augmentation', fname)
    output_f = open(output_path, 'w')

    # training set of all splits (validation and testing items are removed)
    dataset = utils_da.load_data(args.dataset)  # sid -> sequence
    raw_num = len(dataset)
    generated_count = 0
    da_id = raw_num

    if args.method == 'SeqSplit':
        src_num = int(len(dataset) * args.percentage)  # number of samples DA methods will be applied on
        src_sid_list = random.sample(range(1, len(dataset)+1), src_num)

        print("%d training samples are loaded, with percentage %f, "
              "%d samples are selected" % (len(dataset), args.percentage, src_num))

        for sid in src_sid_list:
            src_seq = dataset[sid]

            # if args.method == 'SeqSplit':
            #     if len(src_seq) < 2: continue
            #     generated_count += min(len(src_seq), args.max_len)
            #     new_sample = str(src_seq[0])
            #     for i in range(1, min(len(src_seq), args.max_len)):
            #         da_id += 1
            #         new_sample += (' ' + str(src_seq[i]))
            #         output_f.write("%d %d %s\n" % (da_id, sid, new_sample))

            org_len = len(src_seq)
            # cutoff = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
            cutoff = [0.05, 0.15, 0.25, 0.35, 0.45]
            new_lens = set([int(org_len*i) for i in cutoff])
            for i in new_lens:
                if i < 2: continue
                da_id += 1
                new_sample = str(src_seq[0])
                for j in range(1,i):
                    new_sample += (' ' + str(src_seq[j]))
                output_f.write("%d %d %s\n" % (da_id, sid, new_sample))  # sample_id, original_seq_id, sequence

    elif args.method == 'RandSplit':
        aug_num = int(len(dataset) * args.percentage)  # number of samples DA methods will be applied on
        print("%d training samples are loaded, with percentage %f, "
              "%d samples are selected" % (len(dataset), args.percentage, aug_num))
        cur_num = 0
        used_iid_list = []
        len_sum = 0
        while cur_num < aug_num:
            cur_num += 1
            sid = random.randint(1, len(dataset))
            while (sid in used_iid_list) or len(dataset[sid]) < 2*(args.cut_points+1):
                sid = random.randint(1, len(dataset))
            used_iid_list.append(sid)
            src_seq = dataset[sid]
            len_sum += len(src_seq)
            org_len = len(src_seq)
            cut_points = [2*i for i in random.sample(range(1, int(len(src_seq)/2)), args.cut_points)]
            cut_points.sort()
            print("len=%d, cut at: %s" % (org_len, cut_points))
            for i in range(-1, len(cut_points)):
                da_id += 1
                if i == -1:
                    new_sample = utils_da.sequence2str(src_seq[0:cut_points[0]])
                elif i == len(cut_points)-1:
                    new_sample = utils_da.sequence2str(src_seq[cut_points[-1]:])
                    print(new_sample)
                else:
                    new_sample = utils_da.sequence2str(src_seq[cut_points[i]:cut_points[i+1]])
                output_f.write("%d %d %s\n" % (da_id, sid, new_sample))  # sample_id, original_seq_id, sequence
        print("AvgLen of DA set: %d" %(len_sum / (da_id-raw_num)))


    elif args.method == 'SynRep':
        print("Augmenting using Synonym Replacement, with R=%f" % args.replace_percentage)
        cs_similar_map = utils_da.load_similar_items(args.augmentation_file)  # cs_id -> [similar_id1, similar_id2, similar_id3]
        print("len(cs_similar_map) = %d" % len(cs_similar_map))
        similar_appearances = utils_da.find_similar_appearance(dataset, cs_similar_map)
        print("len(similar_appearance) = %d" % len(similar_appearances))
        for cs_id, similar_id_list in cs_similar_map.items():
            position_lists = [similar_appearances[sim_id] for sim_id in similar_id_list]
            possible_positions = set()
            for plist in position_lists:
                possible_positions = possible_positions.union(set(plist))
            possible_positions = list(possible_positions)

            app_num = len(possible_positions)  # total number of appearances of similar items of this cs item
            pos_num = min(args.max_replace, math.ceil(app_num * args.replace_percentage))  # number of appearances need to be replaced
            print("cs item %d: %d appearances for similar items, %d to be replaced" % (cs_id, app_num, pos_num))
            selected_app_idx_list = random.sample(range(0, app_num), pos_num)
            for app_idx in selected_app_idx_list:
                da_id += 1
                sid, rep_idx = possible_positions[app_idx]  # (sid, pos_idx)
                seq = deepcopy(dataset[sid])
                seq[rep_idx] = cs_id
                new_sample_str = utils_da.sequence2str(seq)
                output_f.write("%d %d %s\n" % (da_id, sid, new_sample_str))

    print("%d samples are generated" % (da_id-raw_num))
    output_f.close()
