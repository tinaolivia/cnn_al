import os
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import methods
import helpers

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            
            if steps % args.test_interval == 0:
                dev_acc, dev_loss = evaluate(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, '{}best'.format(args.dataset), steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

    dev_acc, dev_loss = evaluate(dev_iter, model, args)
    save(model, args.save_dir, 'snapshot', steps)
    
def al(test_set, train_df, test_df, model, al_iter, args):

    # defining activation functions
    log_softmax = nn.LogSoftmax(dim=1).cuda()
    softmax = nn.Softmax(dim=1).cuda()
        
    # querying instances
    if args.method == 'random':
        subset = methods.random(test_set, args)
        print('\nIter {}, selected {} instances at random.\n'.format(al_iter, len(subset)))
    
    elif args.method == 'entropy':
        if args.cluster: subset, total_entropy = methods.entropy_w_clustering(test_set, test_df['text'], model, log_softmax, args)
        else: subset, total_entropy = methods.entropy(test_set, model, log_softmax, args)
        print('\nIter {}, selected {} instances according to entropy uncertainty, total entropy {}.\n'.format(al_iter, len(subset), total_entropy))
        
    elif args.method == 'dropout':
        subset, total_var = methods.dropout(test_set, model, softmax, args)
        print('\nIter {}, selected {} instances according to dropout uncertainty, total variance {}.\n'.format(al_iter, len(subset), total_var))
    
    else:
        print('No method selected.')
        sys.exit()
        
    # adding randomness 
    if args.method != 'random' and args.randomness > 0:
        print('\nInitial subset: {}'.format(subset))
        subset = helpers.randomness(subset, len(test_set), args)
        print('\nUpdated subset: {}'.format(subset))

        
    # updating datasets
    print('\nSubset: {}\n'.format(subset))
    print('\nUpdating datasets ...\n')
    helpers.update_datasets(args.datapath, train_df, test_df, subset, args)


def evaluate(data_iter, model, args):
    model.eval()
    corrects = 0
    avg_loss = 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = float(100.0 * corrects/size)
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy, avg_loss


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    #x = autograd.Variable(x, volatile=True)
    with torch.no_grad():    
        x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print('x: ',x)
    output = model(x)
    softmax = torch.nn.Softmax(dim=1)
    print(softmax(output))
    print('output: ',output)
    _, predicted = torch.max(output, 1)
    return predicted.data

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
