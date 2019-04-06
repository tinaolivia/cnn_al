#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import helpers

def random(data_size, args):
    '''
        input:
        data_size: size of the dataset 
    '''
    random_perm = torch.randperm(data_size)
    subset = list(random_perm[:args.inc].numpy())
    return subset

def random_w_clustering(data_size, df, args):
    '''
        input:
        data_size: size of dataset
        df: dataframe with raw text documents
    '''
    subset=[]
    kmeans = helpers.clustering(df, args)
    random_perm = torch.randperm(data_size)
    for i in range(args.inc):
        for j in range(data_size):
            if kmeans[random_perm[j]] == i:
                subset.append(random_perm[j])
                break
    return subset

def entropy(data_iter, model, log_softmax, args, df=None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        log_softmax: log softmax activation function        
        df: dataframe column of text data
    '''
    model.eval()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args) 
    top_e = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter): 
        logPys = helpers.get_preds(batch, model, log_softmax, args).cuda()
        entropies = -(logPys*torch.exp(logPys)).sum(dim=1).cuda()            
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and entropies[j] > torch.min(top_e):
                        top_e[k] = entropies[j]
                        top_ind[k] = i*args.batch_size + j
            else:
                if entropies[j] > torch.min(top_e):
                    min_e, idx = torch.min(top_e, dim=0)
                    top_e[idx] = entropies[j]
                    top_ind[int(idx)] = i*args.batch_size + j
                
    subset = list(top_ind.cpu().numpy())
    return subset, top_e.cpu().detach().numpy().sum()


def margin(data_iter, model, softmax, args, df=None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        softmax: softmax activation function
        df: dataframe column of text data
    '''
    model.eval()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args) 
    min_m = torch.ones(args.inc).cuda()
    min_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter):
        logits = helpers.get_preds(batch, model, softmax, args).cuda()
        logits = logits.sort(descending=True)[0]
        margins = logits[:,0] - logits[:,1]
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and margins[j] < min_m[k]:
                        min_m[k] = margins[j]
                        min_ind[k] = i*args.batch_size + j 
            else:
                if margins[j] < torch.max(min_m): #, dim=0)[0]:
                    max_m, idx = torch.max(min_m, dim=0)
                    min_m[idx] = margins[j]
                    min_ind[idx] = i*args.batch_size + j
                
    subset = list(min_ind.cpu().numpy())
    return subset, min_m.cpu().detach().numpy().sum()
        


def variation_ratio(data_iter, model, softmax, args, df = None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        softmax: softmax activation function
        df: dataframe column of text data
    '''
    model.eval()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    top_var = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter):
        logits = helpers.get_preds(batch, model, softmax, args).cuda()
        var = 1 - logits.max(dim=1)[0]
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and var[j] > top_var[k]:
                        top_var[k] = var[j]
                        top_ind[k] = i*args.batch_size + j
            else:
                if var[j] > torch.min(top_var):
                    min_var, idx = torch.min(top_var, dim=0)
                    top_var[idx] = var[j]
                    top_ind[idx] = i*args.batch_size + j
                
    subset = list(top_ind.cpu().numpy())
    return subset, top_var.cpu().detach().numpy().sum()


#--------------------------------------------------------------------------------------------------
# Dropout methods
    
def dropout_variability(data_iter, model, softmax, args, df=None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        softmax: softmax activation function
        df: dataframe column of text data
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    top_var = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter):
        probs = torch.empty((args.num_preds, len(batch), args.class_num)).cuda()
        for j in range(args.num_preds):
            probs[j] = helpers.get_preds(batch, model, softmax, args).cuda()
        mean = probs.mean(dim=0)
        var = torch.pow(probs - mean, 2).sum(dim=1).cuda()
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and var[j] > top_var[k]:
                        top_var[k] = var[j]
                        top_ind[k] = i*args.batch_size + j
            else:
                if var[j] > torch.min(top_var):
                    min_var, idx = torch.min(top_var, dim=0)
                    top_var[idx] = var[j]
                    top_ind[idx] = i*args.batch_size + j
                
    model.eval()
    subset = list(top_ind.cpu().numpy())
    return subset, top_var.cpu().detach().numpy().sum()

def dropout_entropy(data_iter, model, softmax, args, df=None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        softmax: softmax activation function
        df: dataframe column of text data
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    top_e = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter):
        probs = torch.empty((args.num_preds, len(batch), args.class_num)).cuda()
        for j in range(args.num_preds):
            probs[j] = helpers.get_preds(batch, model, softmax, args).cuda()
        mean = probs.mean(dim=0).cuda()
        entropies = -(mean*torch.log(mean)).sum(dim=1).cuda()
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and entropies[j] > top_e[k]:
                        top_e[k] = entropies[j]
                        top_ind[k] = i*args.batch_size + j
            else:
                if entropies[j] > torch.min(top_e):
                    min_e, idx = torch.min(top_e, dim=0)
                    top_e[idx] = entropies[j]
                    top_ind[idx] = i*args.batch_size + j
    
    model.eval()
    subset = list(top_ind.cpu().numpy())
    return subset, top_e.cpu().detach().numpy().sum()

def dropout_margin(data_iter, model, softmax, args, df=None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        softmax: softmax activation function
        df: dataframe column of text data
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    min_m = torch.ones(args.inc).cuda()
    min_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter):
        probs = torch.empty((args.num_preds, len(batch), args.class_num)).cuda()
        for j in range(args.num_preds):
            probs[j] = helpers.get_preds(batch, model, softmax, args).cuda()
        mean = probs.mean(dim=0).sort(descending=True)[0].cuda()
        margins = mean[:,0] - mean[:,1]
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and margins[j] < min_m[k]:
                        min_m[k] = margin[j]
                        min_ind[k] = i*args.batch_size + j
            else:
                if margins[j] < torch.max(min_m):
                    max_m, idx = torch.max(min_m, dim=0)
                    min_m[idx] = margins[j]
                    min_ind[idx] = i*args.batch_size + j
    
    model.eval()
    subset = list(min_ind.cpu().numpy())
    return subset, min_m.cpu().detach().numpy().sum()
    

def dropout_variation(data_iter, model, softmax, args, df=None):
    '''
        input:
        data_iter: data iterator
        model: trained model
        softmax: softmax activation function
        df: dataframe column of text data
    '''
    model.train()
    if args.cluster and (df is not None): kmeans = helpers.clustering(df, args)
    top_var = -torch.ones(args.inc).cuda()
    top_ind = torch.empty(args.inc).cuda()
    
    for i, batch in enumerate(data_iter):
        probs = torch.empty((args.num_preds, len(batch), args.class_num)).cuda()
        for j in range(args.num_preds):
            probs[j] = helpers.get_preds(batch, model, softmax, args).cuda()
        mean = probs.mean(dim=0).cuda()
        var = 1 - mean.max(dim=1)[0]
        for j in range(len(batch)):
            if args.cluster and (df is not None):
                for k in range(args.inc):
                    if kmeans[i*args.batch_size+j] == k and var[j] > top_var[k]:
                        top_var[k] = var[j]
                        top_ind[k] = i*args.batch_size + j
            else:
                if var[j] > torch.min(top_var):
                    min_var, idx = torch.min(top_var, dim=0)
                    top_var[idx] = var[j]
                    top_ind[idx] = i*args.batch_size + j
                
    model.eval()
    subset = list(top_ind.cpu().numpy())
    return subset, top_var.cpu().detach().numpy().sum()
        
