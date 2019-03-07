#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchtext
import torchtext.data as data
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def get_feature(data, text_field, args):
    feature = data.text
    feature = [[text_field.vocab.stoi[x] for x in feature]]
    if len(feature[0]) < 1: 
        feature = torch.tensor([float('nan')])
    else: 
        feature = torch.tensor(feature)
        if feature.shape[1] < max(args.kernel_sizes):
            feature = torch.cat((feature, torch.zeros((1, max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
            with torch.no_grad(): feature = autograd.Variable(feature)
            if args.cuda: feature = feature.cuda()
    return feature

def get_output(data, text_field, model, args):
    model.eval()
    feature = data.text
    feature = [[text_field.vocab.stoi[x] for x in feature]]
    if len(feature[0]) < 1: logit = torch.tensor([float('nan')])
    else:
        feature = torch.tensor(feature)
        if feature.shape[1] < max(args.kernel_sizes): 
            feature = torch.cat((feature, torch.zeros((1,max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
        with torch.no_grad(): feature = autograd.Variable(feature)
        if args.cuda:
            feature = feature.cuda()
        
        logit = model(feature)

    return logit

'''
def update_datasets(train, test, subset, args):
    fields = train.fields
    test = list(test)
    new_train = list(train)
    new_test = []
    for i in range(len(test)):
        if not (i in subset): new_test.append(test[i])
        else: new_train.append(test[i])
    return data.Dataset(new_train,fields), data.Dataset(new_test,fields)
'''

def update_datasets(path, train_df, test_df, subset, args):
    train_df.append(test_df.iloc[subset])
    test_df.drop(subset)
    train_df.to_csv('{}/train.csv'.format(path), index=False, header=False)
    test_df.to_csv('{}/test.csv'.format(path), index=False, header=False)
    
    
def clustering(data, args):
    '''
        input:
        data: dataframe column containing text
    '''
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    kmeans = KMeans(n_clusters=args.inc).fit_predict(X)
    kmeans = torch.tensor(kmeans)
    if args.cuda: kmeans = kmeans.cuda()
    return kmeans