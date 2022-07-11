import argparse
import os.path
import random

import utils_da

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--method', required=True, type=str)
parser.add_argument('--percentage', default=0.2, type=float)

# settings for Sequence Split method
parser.add_argument('--max_len', default=30, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    da_methods = ['SeqSplit', 'SynRep', 'Mixed']
    if args.method not in da_methods:
        print("Illegal data augmentation method, select from " + str(da_methods))

    fname = '{}.da.{}.per={}.maxlen={}.txt'
    fname = fname.format(args.dataset, args.method, args.percentage, args.max_len)
    output_path = os.path.join('data', args.dataset, 'augmentation', fname)
    output_f = open(output_path, 'w')

    # training set of all splits (validation and testing items are removed)
    dataset = utils_da.load_data(args.dataset)

    raw_num = len(dataset)
    src_num = int(len(dataset) * args.percentage)  # number of samples DA methods will be applied on
    src_sid_list = random.sample(range(1, len(dataset)+1), src_num)

    print("%d training samples are loaded, with percentage %f, "
          "%d samples are selected" % (len(dataset), args.percentage, src_num))

    generated_count = 0
    da_id = raw_num
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

        if args.method == 'SeqSplit':
            org_len = len(src_seq)
            cutoff = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
            new_lens = set([int(org_len*i) for i in cutoff])
            print(new_lens)

            for i in new_lens:
                if i < 2: continue
                da_id += 1
                new_sample = str(src_seq[0])
                for j in range(1,i):
                    new_sample += (' ' + str(src_seq[j]))
                output_f.write("%d %d %s\n" % (da_id, sid, new_sample))


    print("%d samples are generated" % (da_id-raw_num))
    output_f.close()
