#!/usr/bin/env python3
import torch
import torch.autograd as autograd
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def get_single_probs(data_ex, text_field, model, act_func, args):
    '''
        input:
        data_ex: single data example
        text_field: text field
        model: trained model
        act_func: activation function
    '''
    feature = data_ex.text
    feature = [[text_field.vocab.stoi[x] for x in feature]]
    feature = torch.tensor(feature).cuda()
    if feature.shape[1] < max(args.kernel_sizes):
        feature = torch.cat((feature, torch.zeros((1, max(args.kernel_sizes)-feature.shape[1]), dtype=torch.long).cuda()), dim=1).cuda()
    with torch.no_grad(): feature = autograd.Variable(feature).cuda()
    logit = act_func(model(feature))
    return logit

def get_probs(batch, text_field, model, act_func, args):
    '''
        input:
        data_iter: iterator object (not bucketiterator)
        model: trained model for classification, set to eval() or train() outside
        dim: dimension of output (batch_size, class_num, num_preds)
    '''
    text_field.pad(batch.text)
    feature = batch.text.cuda() 
    #feature.data.t_()
    logit = act_func(model(feature))
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
    
def randomness(subset, range_, args):
    Bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.randomness]))
    for i,element in enumerate(subset):
        B = int(Bern.sample())
        if B == 0:
            print('\nUpdating subset element {}'.format(i))
            temp = subset[i]
            while (temp in subset):
                temp = int(torch.randint(high=range_, size=(1,)))
            subset[i] = temp
    return subset
    
def clustering(data, args):
    '''
        input:
        data: dataframe column containing text
    '''
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    kmeans = MiniBatchKMeans(n_clusters=args.inc).fit_predict(X)
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
