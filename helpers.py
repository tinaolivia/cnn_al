#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchtext
import torchtext.data as data
import sys
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
def clustering(data, text_field, args):
    data = data.text
    features = []
    for example in data:
        feature = [text_field.vocab.stoi[x] for x in example]
        features.append(feature)
    kmeans = KMeans(n_clusters=args.inc).fit_predict(features)
    kmeans = torch.tensor(kmeans)
    #features = [[text_field.vocab.stoi[x] for x in data[i].text for i in range(len(data))]]
    #kmeans = torch.tensor(KMeans(n_clusters=args.inc).fit_predict(features))
    #vec = TfidfVectorizer(lowercase=False, preprocessor=None, tokenizer=None, stop_words='english')
    #data = [data[i].text for i in range(len(data))]
    #data = vec.fit_transform(data)
    #kmeans = torch.tensor(KMeans(n_clusters=args.inc).fit_predict(data))
    if args.cuda: kmeans = kmeans.cuda()
    return kmeans
    '''
    
def clustering(data, text_field, args):
    data = data.text
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit(data)
    kmeans = KMeans(n_clusters=args.inc).fit_predict(X)
    kmeans = torch.tensor(kmeans)
    if args.cuda: kmeans = kmeans.cuda()
    return kmeans