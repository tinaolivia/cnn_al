#!/usr/bin/env python3
import torch
import torch.autograd as autograd
import csv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def get_feature(data, text_field, args):
    '''
        input:
        data: data sample
        text_field: torchtext.data Field object
    '''
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
    '''
        input:
        data: data sample
        text_field: torchtext.data Field object
        model: model to predict data label
    '''
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


def update_datasets(path, train_df, test_df, subset, args):
    '''
        input:
        path: path to where the data is stored
        train_df: dataframe containing the train data
        test_df: dataframe containing the test data
        subset: subset of test to be added to train
    '''
    print('before: ', len(train_df), len(test_df))
    train_df = train_df.append(test_df.iloc[subset])
    test_df = test_df.drop(subset)
    print('after: ', len(train_df), len(test_df))
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

def write_result(filename, mode, result, args):
    '''
        input:
        filename: path and filename
        mode: writing mode, w=crate new file, a=append to existing
        result: list containing results to add to file
    '''
    with open(filename, mode=mode) as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(result)
