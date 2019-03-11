#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import train_al
import csv
import sys
import data_loaders


csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-num-avg', type=int, default=10, help='number of runs to average over [default=10]')
parser.add_argument('-result-path', type=str, default='results', help='where to store results [default: results]')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('-dataset', type=str, default='twitter', choices=['twitter', 'news', 'imdb', 'ag'], help='dataset [default: twitter]')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]') #changed from int and -1 to str and cpu
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# active learning 
parser.add_argument('-method', type=str, default=None, choices=['random','entropy','vote', 'dropout'],
                    help='active learning query strategy [default: None]')
parser.add_argument('-rounds', type=int, default=500, help='rounds of active querying [default: 500]')
parser.add_argument('-inc', type=int, default=1, help='number of instances added to training data at each round [default: 1]')
parser.add_argument('-num-preds', type=int, default=5, help='number of predictions made when computing dropout uncertainty [default:5]')
#parser.add_argument('-test-method', action='store_true', default=False, help='testing active learning method [default: False]')
parser.add_argument('-criterion', type=float, default=100, help='stopping criterion, accuracy [default:100]')
parser.add_argument('-test-inc', type=bool, default=False, help='testing number of instances added')
parser.add_argument('-test-preds', type=bool, default=False, help='testing number of predictions (dropout, vote entropy)')
parser.add_argument('-hist', type=bool, default=False, help='whether to make a uncertainty histogram')
parser.add_argument('-cluster', type=bool, default=False, help='whether to cluster the data or not [default:False]')
parser.add_argument('-randomness', type=float, default=0.05, help='percentage of randomness when selecting instances [default: 0.05]')
args = parser.parse_args()
    
# defining text and label fields
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)

# load data
if args.dataset == 'twitter':
    print('\nLoading Twitter data ...')
    train_set, train_iter, val_set, val_iter, test_set, test_iter = data_loaders.twitter('data', text_field, label_field, args, 
                                                                                         device=torch.device('cpu'), repeat=False)
    
elif args.dataset == 'news':
    print('\nLoading 20 Newsgroup data ...')
    train_set, train_iter, val_set, val_iter, test_set, test_iter = data_loaders.news('data', text_field, label_field, args, 
                                                                                      device=torch.device('cpu'), repeat=False)
    
elif args.dataset == 'imdb':
    print('\nLoading IMDB movie review data ...')
    train_set, train_iter, val_set, val_iter, test_set, test_iter = data_loaders.imdb('data/imdb', text_field, label_field, args, 
                                                                                      device=torch.device('cpu'), repeat=False)
    
elif args.dataset == 'ag':
    print('\nloading AG News data ...')
    train_set, train_iter, val_set, val_iter, test_set, test_iter = data_loaders.ag('data/ag', text_field, label_field, args, 
                                                                                    device=torch.device('cpu'), repeat=False)
    
else: 
    print('\nDataset is not defined.')
    sys.exit()


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
if args.method is not None: args.save_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
else: args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.result_path = os.path.join(args.result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.isdir(args.result_path): os.makedirs(args.result_path)
if args.randomness < 0: args.randomness = 0
if args.randomness > 1: args.randomness /= 100

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

for avg_iter in range(args.num_avg):
    
# model
    print('\nDefining model ...')
    cnn = model.CNN_Text(args)
    torch.save(cnn, 'snapshot/{}_untrained.pt'.format(args.dataset))
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot,map_location='cpu'))

    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()
    
# train, test, or predict
    if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.test:
        train.evaluate(test_iter, cnn, args)
    elif args.method is not None:
        train_al.train_with_al(train_set,val_set,test_set,cnn, text_field, label_field, avg_iter, args)
    else:
        print()
        try:
            train.train(train_iter, val_iter, cnn, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('Exiting from training early')

