#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import csv
import sys
import pandas as pd

import model
import data_loaders
import helpers
import train


csv.field_size_limit(sys.maxsize)


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 50]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=250, help='iteration numbers to stop without performance increasing [default: 250]')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-num-avg', type=int, default=10, help='number of runs to average over [default=10]')
parser.add_argument('-result-path', type=str, default='results', help='where to store results [default: results]')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
parser.add_argument('-dataset', type=str, default='imdb', choices=['imdb', 'ag'], help='dataset [default: imdb]')
parser.add_argument('-data-path', type=str, default='data', help='path to where data is stored [default: data]')
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
# active learning 
parser.add_argument('-method', type=str, default=None, help='active learning query strategy [default: None]')
parser.add_argument('-rounds', type=int, default=500, help='rounds of active querying [default: 500]')
parser.add_argument('-inc', type=int, default=1, help='number of instances added to training data at each round [default: 1]')
parser.add_argument('-num-preds', type=int, default=5, help='number of predictions made when computing dropout uncertainty [default:5]')
#parser.add_argument('-test-method', action='store_true', default=False, help='testing active learning method [default: False]')
parser.add_argument('-criterion', type=float, default=100, help='stopping criterion, accuracy [default:100]')
parser.add_argument('-test-inc', type=bool, default=False, help='testing number of instances added')
parser.add_argument('-test-preds', type=bool, default=False, help='testing number of predictions (dropout, vote entropy)')
parser.add_argument('-hist', type=bool, default=False, help='whether to make a uncertainty histogram')
parser.add_argument('-cluster', type=bool, default=False, help='whether to cluster the data or not [default:False]')
parser.add_argument('-randomness', type=float, default=0, help='percentage of randomness when selecting instances [default: 0]')
args = parser.parse_args()

# defining additional arguments
args.now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
if args.method is not None: args.save_dir = os.path.join(args.method, args.now)
else: args.save_dir = os.path.join(args.save_dir, args.now)
args.result_path = os.path.join(args.result_path, args.now)
if not os.path.isdir(args.result_path): os.makedirs(args.result_path)
if args.randomness < 0: args.randomness = 0
elif args.randomness > 1: args.randomness /= 100

# creating new path for later
while os.path.isdir(os.path.join(args.data_path, args.dataset, args.now)):
    args.now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

os.makedirs(os.path.join(args.data_path, args.dataset, args.now))

# copying validation set to new path
val_df = pd.read_csv('{}/{}/val.csv'.format(args.data_path, args.dataset), header=None, names=['text', 'label'])
val_df.to_csv('{}/{}/{}/val.csv'.format(args.data_path, args.dataset, args.now), header=False, index=False)

for avg_iter in range(args.num_avg):
    
    print('\nRun {}\n'.format(avg_iter))
    
    # setting datapath to original datasets
    args.datapath = os.path.join(args.data_path, args.dataset)
    
    filename = os.path.join(args.result_path, '{}_{}_{}.csv'.format(args.method, args.dataset, avg_iter))
    helpers.write_result(filename, 'w', ['Train Size', 'loss', 'accuracy', 'total {}'.format(args.dataset), 'time'], args)
    total = 0
    time = 0
    
    for al_iter in range(args.rounds):
    
        # defining text and label fields
        text_field = data.Field(lower=True)
        label_field = data.Field(sequential=False)

        # load data
        print('\nLoading {} data ...\n'.format(args.dataset))
        train_set, val_set, test_set = data_loaders.ds_loader(args.datapath, text_field, label_field, args)
        train_iter = data.BucketIterator(train_set, batch_size=args.batch_size, device=torch.device('cpu'), repeat=False)
        val_iter = data.BucketIterator(val_set, batch_size=args.batch_size, device=torch.device('cpu'), repeat=False)
        test_iter = data.Iterator(test_set, batch_size=args.batch_size, train=False, shuffle=False, sort=False, sort_within_batch=False,  device=torch.device('cpu'), repeat=False)
        train_df = pd.read_csv('{}/train.csv'.format(args.datapath), header=None, names=['text', 'label'])
        test_df = pd.read_csv('{}/test.csv'.format(args.datapath), header=None, names=['text', 'label'])


        # update args and print
        args.embed_num = len(text_field.vocab)
        args.class_num = len(label_field.vocab) - 1
        # setting datapath to updated datasets
        args.datapath = os.path.join(args.data_path, args.dataset, args.now) #updated dataset in new folder


        print("\nParameters:")
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))
    
        # model
        print('\nDefining model ...')
        cnn = model.CNN_Text(args)


        if args.cuda:
            torch.cuda.set_device(args.device)
            cnn = cnn.cuda()
            
        # training model, reporting results, model set to train() and eval() in the functions 
        train.train(train_iter, val_iter, cnn, args)
        acc, loss = train.evaluate(val_iter, cnn, args)
        helpers.write_result(filename, 'a', [len(train_set), loss, acc, total, time], args)
        
        # active learning
        total, time = train.al(test_iter, train_df, test_df, cnn, al_iter, args)
        
    print('\nDone with run {} of active learning with {}. Results are stored in {}.\n'.format(avg_iter, args.method, args.datapath))

        
        
print('\nDone with {} runs of {} active learning loops with {}. Results are stored in {}.\n'.format(args.num_avg, args.rounds, args.method, args.datapath))
