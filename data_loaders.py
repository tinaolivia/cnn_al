#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torchtext.data as data

def twitter(text_field, label_field, args, **kargs):
    datafields = [("text", text_field), ("label", label_field)]
    trn, val, tst = data.TabularDataset.splits(path='data', train='train.csv', validation='val.csv',test='test.csv',
                                               format='csv', fields=datafields)
    
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    train_iter = data.BucketIterator(trn, batch_size=args.batch_size,**kargs)
    val_iter = data.BucketIterator(val, batch_size=args.batch_size,**kargs)
    test_iter = data.BucketIterator(tst, batch_size=args.batch_size,**kargs)
    
    return trn, train_iter, val, val_iter, tst, test_iter

def news(text_field, label_field, args, **kargs):
    datafields = [("text", text_field),("label", label_field)]
    trn, val, tst = data.TabularDataset.splits(path='data', train='news_train.csv', validation='news_val.csv',
                                               test='news_test.csv', format='csv', fields=datafields)
    
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    train_iter = data.BucketIterator(trn, args.batch_size, **kargs)
    val_iter = data.BucketIterator(val, args.batch_size, **kargs)
    test_iter = data.BucketIterator(tst, args.batch_size, **kargs)
    
    return trn, train_iter, val, val_iter, tst, test_iter

def imdb(text_field, label_field, args, **kargs):
    datafields = [("text",text_field),("label",label_field)]
    trn, val, tst = data.TabularDataset.splits(path='data/imdb', train='train.csv', validation='val.csv', 
                                               test='test.csv', format='csv', fields=datafields)
    
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    train_iter = data.BucketIterator(trn, args.batch_size, **kargs)
    val_iter = data.BucketIterator(val, args.batch_size, **kargs)
    test_iter = data.BucketIterator(tst, args.batch_size, **kargs)
    
    return trn, train_iter, val, val_iter, tst, test_iter

def ag(text_field, label_field, args, **kargs):
    datafields = [("text",text_field), ("label",label_field)]
    trn, val, tst = data.TabularDataset.splits(path='data/ag', train='train.csv', validation='val.csv', 
                                               test='test.csv', format='csv', fields=datafields)
    
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    train_iter = data.BucketIterator(trn, args.batch_size, **kargs)
    val_iter = data.BucketIterator(val, args.batch_size, **kargs)
    test_iter = data.BucketIterator(val, args.batch_size, **kargs)
    
    return trn, train_iter, val, val_iter, tst, test_iter