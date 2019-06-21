#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import helpers

def random(args):
    '''
        input:
        data_size: size of the dataset 
    '''
    random_perm = torch.randperm(args.test_size)
    subset = list(random_perm[:args.inc].numpy())
    return subset

def entropy(data, model, log_softmax, args, df=None):
    '''
        input:
        data: dataset
        model: trained model
        log_softmax: log softmax activation function        
        df: dataframe of dataset
    '''
    model.eval()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df['text'], args)
    top_e = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    
    for i, example in enumerate(data):
        logPys = helpers.get_single_probs(example, text_field, model, log_softmax, args).cuda()
        entropy_ = -(logPys*torch.exp(logPys)).sum().cuda()
        if args.cluster and df is not None:
            if entropy_ > top_e[kmeans[i]]: top_e[kmeans[i]], top_ind[kmeans[i]] = entropy_, i
        else:
            if entropy_ > top_e.min():
                min_e, idx = torch.min(top_e, dim=0)
                top_e[idx], top_ind[idx] = entropy_, i
        
    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_e.cpu().detach().numpy().sum()


def margin(data, model, softmax, args, df=None):
    '''
        input:
        data: dataset
        model: trained model
        softmax: softmax activation function
        df: dataframe of dataset
    '''
    model.eval()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df['text'], args) 
    top_m = torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    
    for i, example in enumerate(data):
        logits = helpers.get_single_probs(example, text_field, model, softmax, args).cuda()
        logits = logits.sort(descending=True)[0].cuda()
        margin_ = logits[0][0] - logits[0][1]
        if args.cluster and (df is not None):
            if margin_ < top_m[kmeans[i]]: top_m[kmeans[i]], top_ind[kmeans[i]] = margin_, i
        else:
            if margin_ < top_m.max():
                max_m, idx = torch.max(top_m, dim=0)
                top_m[idx], top_ind[idx] = margin_, i
                
    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_m.cpu().detach().numpy().sum()
        


def variation_ratio(data, model, softmax, args, df = None):
    '''
        input:
        data: dataset
        model: trained model
        softmax: softmax activation function
        df: dataframe of dataset
    '''
    model.eval()
    if args.cluster and (df is not None):
        kmeans = helpers.clustering(df['text'], args)
        test_k = kmeans.cpu().detach().numpy()
        for i in range(args.inc):
            for j in range(len(data)):
                if test_k[j] == i:
                    print(i, 'check')
                    break
    top_var = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    
    for i, example in enumerate(data):
        logits = helpers.get_single_probs(example, text_field, model, softmax, args).cuda()
        var = 1 - logits.max()
        if args.cluster and (df is not None):
            if var > top_var[kmeans[i]]: top_var[kmeans[i]], top_ind[kmeans[i]] = var, i
        else:
            if var > top_var.min():
                min_var, idx = torch.min(top_var, dim=0)
                top_var[idx], top_ind[idx] = var, i

    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_var.cpu().detach().numpy().sum()
                
#--------------------------------------------------------------------------------------------------
# Dropout methods
    
def dropout_variability(data, model, softmax, args, df=None):
    '''
        input:
        data: dataset
        model: trained model
        softmax: softmax activation function
        df: dataframe of dataset
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df['text'], args)
    top_var = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    
    for i, example in enumerate(data):
        probs = []
        for j in range(args.num_preds): probs.append(helpers.get_single_probs(example, text_field, model, softmax, args))
        probs = torch.stack(probs).cuda()
        mean = probs.mean(dim=0)
        var = torch.pow(probs - mean, 2).sum().cuda()
        if args.cluster and (df is not None):
            if var > top_var[kmeans[i]]: top_var[kmeans[i]], top_ind[kmeans[i]] = var, i
        else: 
            if var > top_var.min():
                min_var, idx = torch.min(top_var, dim=0)
                top_var[idx], top_ind[idx] = var, i
    
    model.eval()            
    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_var.cpu().detach().numpy().sum()
    

def dropout_entropy(data, model, softmax, args, df=None):
    '''
        input:
        data: dataset
        model: trained model
        softmax: softmax activation function
        df: dataframe of dataset
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df['text'], args)
    top_e = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    
    for i, example in enumerate(data):
        probs = []
        for j in range(args.num_preds): probs.append(helpers.get_single_probs(example, text_field, model, softmax, args))
        probs = torch.stack(probs).cuda()
        mean = probs.mean(dim=0).cuda()
        entropy = -(mean*torch.log(mean)).sum().cuda()
        if args.cluster and (df is not None):
            if entropy > top_e[kmeans[i]]: top_e[kmeans[i]], top_ind[kmeans[i]] = entropy, i
        else:
            if entropy > top_e.min():
                min_e, idx = top_e.min(dim=0)
                top_e[idx], top_ind[idx] = entropy, i
                
    model.eval()
    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_e.cpu().detach().numpy().sum()

def dropout_margin(data, model, softmax, args, df=None):
    '''
        input:
        data: dataset
        model: trained model
        softmax: softmax activation function
        df: dataframe of dataset
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df['text'], args)
    top_m = torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    
    for i, example in enumerate(data):
        probs = []
        for j in range(args.num_preds): probs.append(helpers.get_single_probs(example, text_field, model, softmax, args))
        probs = torch.stack(probs).cuda()
        mean = probs.mean(dim=0).sort(descending=True)[0].cuda()
        margin = mean[0][0] - mean[0][1]
        if args.cluster and (df is not None):
            if margin < top_m[kmeans[i]]: top_m[kmeans[i]], top_ind[kmeans[i]] = margin, i
        else:
            if margin < top_m.max():
                max_m, idx = torch.max(top_m, dim=0)
                top_m[idx], top_ind[idx] = margin, i
                
    model.eval()
    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_m.cpu().detach().numpy().sum()

def dropout_variation(data, model, softmax, args, df=None):
    '''
        input:
        data: dataset
        model: trained model
        softmax: softmax activation function
        df: dataframe of dataset
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df['text'], args)
    top_var = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    text_field = data.fields['text']
    for i, example in enumerate(data):
        probs = []
        for j in range(args.num_preds): probs.append(helpers.get_single_probs(example, text_field, model, softmax, args))
        probs = torch.stack(probs).cuda()
        mean = probs.mean(dim=0).cuda()
        var = 1 - mean.max()
        if args.cluster and (df is not None):
            if var > top_var[kmeans[i]]: top_var[kmeans[i]], top_ind[kmeans[i]] = var, i
        else:
            if var > top_var.min():
                min_var, idx = torch.min(top_var, dim=0)
                top_var[idx], top_ind[idx] = var, i
                
    model.eval()
    subset = list(top_ind.cpu().numpy())
    if args.cluster:
        for i in range(len(subset)):
            subset[i] = int(subset[i])
    return subset, top_var.cpu().detach().numpy().sum()
        
