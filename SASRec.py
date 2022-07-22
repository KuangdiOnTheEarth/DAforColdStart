import os
import time
from copy import deepcopy

import torch
import argparse

from SASRec_pytorch.model import SASRec
from SASRec_pytorch.utils import *
from SASRec_pytorch.utils_cs import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--cold_start', default=False, type=str2bool)
parser.add_argument('--da_file', default='', type=str)
parser.add_argument('--output_dir', default='output', type=str)

args = parser.parse_args()
if not os.path.isdir(os.path.join(args.output_dir, args.dataset + '_' + args.train_dir)):
    os.makedirs(os.path.join(args.output_dir, args.dataset + '_' + args.train_dir))
with open(os.path.join(args.output_dir, args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    if args.cold_start:
        print('-- Cold-Start Evaluation mode is activated --')
        print('Loading dataset ...')
        dataset = cs_data_partition(args.dataset, args.da_file)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset
        user_train = user_train['ws']
        user_valid = user_valid['ws']
    else:
        print('Loading dataset ...')
        dataset = data_partition(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(args.output_dir, args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write("%s - %s\nMaxLen=%d, Dropout=%f\nAugmentation dataset: %s\n"
            % ("SASRec", args.dataset, args.maxlen, args.dropout_rate, args.da_file))
    f.flush()
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        if args.cold_start:
            NDCG10_map, HR10_map, NDCG30_map, HR30_map = cs_evaluate(model, dataset, args)
            print('\nWarm-Start:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)'
                  % (NDCG10_map['ws'], HR10_map['ws'], NDCG30_map['ws'], HR30_map['ws'])) # include data augmentation samples
            print('User-CS:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)'
                  % (NDCG10_map['ucs'], HR10_map['ucs'], NDCG30_map['ucs'], HR30_map['ucs']))
            print('Item-CS:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)'
                  % (NDCG10_map['ics'], HR10_map['ics'], NDCG30_map['ics'], HR30_map['ics']))
            print('Mixed-CS:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)'
                  % (NDCG10_map['mcs'], HR10_map['mcs'], NDCG30_map['mcs'], HR30_map['mcs']))
            print('Weighted-Average:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)'
                  % (NDCG10_map['avg'], HR10_map['avg'], NDCG30_map['avg'], HR30_map['avg']))
        else:
            t_test = evaluate(model, dataset, args)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    best_valid = 0
    best_epoch = 0
    best_evaluation_results = None
    best_model_state_dict = None
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')

            if args.cold_start:
                t_valid = evaluate_valid(model, dataset, args)
                log_str = ''
                log_str += ('\nepoch:%d, time: %f(s):\nValid: (NDCG@10: %.4f, HR@10: %.4f)\n' % (epoch, T, t_valid[0], t_valid[1]))
                evaluation_results = cs_evaluate(model, dataset, args)
                NDCG10_map, HR10_map, NDCG30_map, HR30_map = evaluation_results
                log_str += 'test sets:\n'
                log_str += ('Warm-Start: (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)\n'
                            % (NDCG10_map['ws'], HR10_map['ws'], NDCG30_map['ws'], HR30_map['ws']))  # include data augmentation samples
                log_str += ('User-CS:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)\n'
                            % (NDCG10_map['ucs'], HR10_map['ucs'], NDCG30_map['ucs'], HR30_map['ucs']))
                log_str += ('Item-CS:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)\n'
                            % (NDCG10_map['ics'], HR10_map['ics'], NDCG30_map['ics'], HR30_map['ics']))
                log_str += ('Mixed-CS:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)\n'
                            % (NDCG10_map['mcs'], HR10_map['mcs'], NDCG30_map['mcs'], HR30_map['mcs']))
                log_str += ('Weighted-Average:\t (NDCG@10: %.4f, HR@10: %.4f) (NDCG@30: %.4f, HR@30: %.4f)\n'
                            % (NDCG10_map['avg'], HR10_map['avg'], NDCG30_map['avg'], HR30_map['avg']))
                print(log_str)
                f.write(log_str)
                f.flush()

                if t_valid[0] > best_valid:
                    best_epoch = epoch
                    best_valid = t_valid[0]
                    best_evaluation_results = evaluation_results
                    best_model_state_dict = deepcopy(model.state_dict())
            else:
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))
                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()

            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = os.path.join(args.output_dir, args.dataset + '_' + args.train_dir)
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.cs={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.cold_start)
            # torch.save(model.state_dict(), os.path.join(folder, fname))
            torch.save(best_model_state_dict, os.path.join(folder, fname))
            f.write("\nBest epoch: %d, validation NDCG@10=%f" % (best_epoch, best_valid))

            metrics = ['NDCG@10', 'HR@10', 'NDCG@30', 'HR@30']
            splits = ['ws', 'ucs', 'ics', 'mcs', 'avg']
            print("Evaluation Metric: WS, UCS, ICS, MCS, AVG")
            for i in range(4):
                metric_name = metrics[i]
                res = best_evaluation_results[i]
                f.write("\n%s: %.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (metric_name, res['ws'], res['ucs'], res['ics'], res['mcs'], res['avg']))
            f.flush()
    
    f.close()
    sampler.close()
    print("Done")
