#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torchtext.data as data

def twitter(path, text_field, label_field, args, **kargs):
    datafields = [("text", text_field), ("label", label_field)]
    trn, val, tst = data.TabularDataset.splits(path=path, train='train.csv', validation='val.csv',test='test.csv',
                                               format='csv', fields=datafields)
    
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    train_iter = data.BucketIterator(trn, batch_size=args.batch_size,**kargs)
    val_iter = data.BucketIterator(val, batch_size=args.batch_size,**kargs)
    test_iter = data.BucketIterator(tst, batch_size=args.batch_size,**kargs)
    
    return trn, train_iter, val, val_iter, tst, test_iter

def news(path, text_field, label_field, args, **kargs):
    datafields = [("text", text_field),("label", label_field)]
    trn, val, tst = data.TabularDataset.splits(path=path, train='news_train.csv', validation='news_val.csv',
                                               test='news_test.csv', format='csv', fields=datafields)
    
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    train_iter = data.BucketIterator(trn, args.batch_size, **kargs)
    val_iter = data.BucketIterator(val, args.batch_size, **kargs)
    test_iter = data.BucketIterator(tst, args.batch_size, **kargs)
    
    return trn, train_iter, val, val_iter, tst, test_iter


def ds_loader(text_field, label_field, args):
    datafields = [("text", text_field), ("label", label_field)]
    trn, val, tst = data.TabularDataset.splits(path=args.datapath, train='train.csv',validation='val.csv', test='test.csv',format='csv', fields=datafields)
    return trn, val, tst
