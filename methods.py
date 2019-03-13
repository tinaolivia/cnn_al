#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import helpers

def random(data, args):
    subset = []
    size = len(data)
    nsel = min(size,args.inc)
    random_perm = torch.randperm(size)
    n = 0
    for i in random_perm:
        if not(i in subset):
            subset.append(int(i))
            n += 1
            if n >= nsel: break
    return subset

def entropy(data, model, act_func, args):
    model.eval()
    subset = []
    top_e = float('-inf')*torch.ones(args.inc)
    if args.cuda: top_e, act_func = top_e.cuda(), act_func.cuda()
    ind = torch.zeros(args.inc)
    text_field = data.fields['text']
    if args.hist: 
        hist = torch.zeros(40)
        if args.cuda: hist = hist.cuda()
    
    for i,example in enumerate(data):
        if i % int(len(data)/100) == 0: print(i)
        logit = helpers.get_output(example, text_field, model, args)
        if (torch.max(torch.isnan(logit)) == 1): 
            entropy = torch.tensor([-1]) 
            print('NaN returned from get_output, iter {}'.format(i))
        else:
            if args.cuda: logit = logit.cuda()
            logPy = act_func(logit)
            entropy = -(logPy*torch.exp(logPy)).sum()
            if args.hist: hist[int(entropy*10)] += 1
        if args.cuda: entropy = entropy.cuda()            
        if entropy.double() > torch.min(top_e).double():
            min_e, idx = torch.min(top_e, dim=0)
            top_e[int(idx)] = entropy.double()
            ind[int(idx)] = i
                                    
    print('Top entropy: ', top_e)
    total_entropy = top_e.sum()
    for i in ind:
        subset.append(int(i))
        
    if args.hist: return subset, args.inc, total_entropy, hist
    else: return subset, total_entropy   

def entropy_w_clustering(data, df, model, log_softmax, args):
    '''
        input:
        data: dataset (torchtext dataset)
        df: dataframe column containing raw text documents
        model: trained model to calculate entropies
        log_softmax: log softmax activation function
    '''
    model.eval()
    subset = []
    kmeans = helpers.clustering(df, args)
    text_field = data.fields['text']
    top_e = -torch.ones(args.inc)
    top_ind = torch.empty(args.inc)
    if args.cuda: top_e = top_e.cuda()
    
    for j in range(args.inc):
        for i, example in enumerate(data):
            if kmeans[i] == j:
                logit = helpers.get_output(example, text_field, model, args)
                logPy = log_softmax(logit)
                entropy = -(logPy*torch.exp(logPy)).sum()
                #if args.cuda: entropy = entropy.cuda()
                if entropy > top_e[j]:
                    top_e[j] = entropy
                    top_ind[j] = i
                    
    for i in range(args.inc):
        subset.append(top_ind[i])
        
    return subset, top_e.sum()
                
            
def dropout(data, model, act_func, args):
    if args.cuda: model = model.cuda()
    model.train()
    subset = []
    text_field = data.fields['text']
    top_var = float('-inf')*torch.ones(args.inc)
    if args.cuda: top_var = top_var.cuda()
    ind = torch.zeros(args.inc)
    if args.hist: 
        hist = torch.zeros(1000)
        if args.cuda: hist = hist.cuda()
    for i, example in enumerate(data):
        if i % int(len(data)/100) == 0: print(i)
        probs = torch.empty((args.num_preds, args.class_num))
        if args.cuda: probs = probs.cuda()
        feature = helpers.get_feature(example, text_field, args)
        if torch.max(torch.isnan(feature)) == 1: 
            var = torch.tensor([float('-inf')])
            print('NaN returned from get_feature, iter {}'.format(i))
        else:
            if args.cuda: feature, act_func = feature.cuda(), act_func.cuda()
            
            for j in range(args.num_preds):
                if args.cuda: probs[j,:] = act_func(model(feature)).cuda()
                else: probs[j,:] = act_func(model(feature)) 
            var = torch.abs(probs - probs.mean(dim=0)).sum() # absolute value or squared here?
        if args.cuda: var = var.cuda()    
        if args.hist: hist[int(var)] += 1        
        if var > torch.min(top_var):
            min_var, idx = torch.min(top_var,dim=0)
            top_var[int(idx)] = var
            ind[int(idx)] = i
    
    print('Top variance: {}'.format(top_var))
    total_var = top_var.sum()
    for i in ind:
        subset.append(int(i))
        
    model.eval()
        
    if args.hist: return subset,  total_var, hist
    else: return subset, total_var 


def vote(data, model,  args):
    model.train()
    subset = []
    top_ve = float('-inf')*torch.ones(args.inc)
    if args.cuda: top_ve = top_ve.cuda()
    ind = torch.zeros((args.inc))
    text_field = data.fields['text']
    for i, example in enumerate(data):
        if i % int(len(data)/100) == 0: print(i)
        preds = torch.zeros(args.num_preds)
        votes = torch.zeros(args.class_num)
        feature = helpers.get_feature(example, text_field, args)
        if torch.max(torch.isnan(feature)) == 1: 
            ventropy = torch.tensor([float('-inf')])
            print('NaN returned from get_feature, iter {}'.format(i))
        else:
            if args.cuda: preds, votes, feature = preds.cuda(), votes.cuda(), feature.cuda()
        
            for j in range(args.num_preds):
                _, preds[j] = torch.max(model(feature),1)
            
            for j in range(args.class_num):
                votes[j] = (preds == j).sum()/args.num_preds
            
            ventropy = -(votes*torch.log(votes)).sum()
        
        if args.cuda: ventropy = ventropy.cuda()    
        if ventropy > torch.min(top_ve):
            min_ve, idx = torch.min(top_ve,dim=0)
            top_ve[int(idx)] = ventropy
            ind[int(idx)] = i 
            
    print('Top vote entropy: {}'.format(top_ve))
    total_ve = top_ve.sum()
    for i in ind:
        subset.append(int(i))
        
    model.eval()

    return subset, total_ve
