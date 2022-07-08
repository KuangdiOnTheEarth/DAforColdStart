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
    dataset = utils_da.load_data(os.path.join('data', args.dataset, 'splits'))

    src_num = int(len(dataset) * args.percentage)  # number of samples DA methods will be applied on
    src_list = random.sample(range(0, len(dataset)), src_num)

    print("%d training samples are loaded, with percentage %d, "
          "%d samples are selected" % (len(dataset), args.percentage, src_num))

    da_count = 0

    for idx in src_list:
        src_seq = dataset[idx]

        if args.method == 'SeqSplit':
            for i in range(len(src_seq)):
                pass
    output_f.close()
