import os
import datetime
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torchtext
import torchtext.data as data
import methods
import helpers
import csv


def train(train_iter, dev_iter, model, round_, args):
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

                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))

             
    save(model, save_dir=os.path.join(args.method,args.save_dir), save_prefix='al', steps=steps, round_=round_, args=args, al=True)
                

def train_with_al(train_set, val_set, test_set, model, avg_iter, args):
    
    log_softmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    if args.cuda: log_softmax = log_softmax.cuda()
    val_iter = data.BucketIterator(test_set, batch_size=args.batch_size, device=-1, repeat=False)
    
    initial_acc, initial_loss = evaluate(val_iter, model, args)
    initial_acc = initial_acc.cpu()
    
    if args.hist:
        with open('histogram/{}_{}.csv'.format(args.method,args.dataset), mode='w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
            csvwriter.writerow(['Uncertainty Histograms'])
    else:        
        if args.test_inc: 
            with open('accuracies/{}_{}_{}inc.csv'.format(args.method, args.dataset, args.inc), mode='w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['Train Size', 'Accuracy', 'Loss'])
                csvwriter.writerow([len(train_set) , initial_acc.numpy(), initial_loss])
        elif args.test_preds: 
            with open('accuracies/{}_{}_{}preds.csv'.format(args.method, args.dataset, args.num_preds), mode='w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['Train Size', 'Accuracy', 'Loss'])
                csvwriter.writerow([len(train_set) , initial_acc.numpy(), initial_loss])
        else:
            with open('accuracies/{}_{}_{}.csv'.format(args.method, args.dataset, avg_iter), mode='w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(['Train Size', 'Accuracy'])
                csvwriter.writerow([len(train_set) , initial_acc.numpy()])
            
    
    
    al_iter = 0
    acc = initial_acc
    
    while acc < args.criterion and al_iter < args.rounds:
        
        subset = []
        
        print('\nTrain: {}, Validation: {}, Test: {} \n'.format(len(train_set),len(val_set), len(test_set)))
    
    
        if args.method == 'random':
            subset,n = methods.random(test_set, subset, args)
            print('\nIter {}, selected {} samples at random\n'.format(al_iter, n))
            
        if args.method == 'entropy':
            if args.hist: subset, n, total_entropy, hist = methods.entropy(test_set, model, subset, log_softmax, args)
            elif args.cluster: subset, n, total_entropy = methods.entropy_w_cluster(test_set, model, subset, log_softmax, args)  
            else: subset, n, total_entropy = methods.entropy(test_set, model, subset, log_softmax, args)                
            print('\nIter {}, selected {} by entropy uncertainty, total entropy {}\n'.format(al_iter, n, total_entropy))
        
        if args.method == 'dropout':
            if args.hist: subset, n, total_var, hist = methods.dropout(test_set, model, subset, softmax, args)
            else: subset, n, total_var = methods.dropout(test_set, model, subset, softmax, args)
            print('\nIter {}, selected {} samples with dropout, total variability {}\n'.format(al_iter, n, total_var))
                    
        if args.method == 'vote':
            subset, n, total_ve = methods.vote(test_set, model, subset, args)
            print('\nIter {}, selected {} by dropout and vote entropy, total vote entropy {}\n'.format(al_iter, n, total_ve))
        
        if args.method != 'random' and args.randomness > 0:
            print('\nInitial subset: {}'.format(subset))
            Bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.randomness]))
            for i,element in enumerate(subset):
                B = int(Bern.sample())
                if B == 0:
                    print('\nUpdating subset element {}'.format(i))
                    temp = subset[i]
                    while (temp in subset):
                        temp = int(torch.randint(high=len(test_set), size=(1,)))
                    subset[i] = temp
        
        print('\nSubset: {}'.format(subset))
        print('\nUpdating datasets ...')
        train_set, test_set = helpers.update_datasets(train_set, test_set, subset, args)
        
        print('\nTrain: {}, Validation: {}, Test: {} \n'.format(len(train_set),len(val_set), len(test_set)))
        
        train_iter = data.BucketIterator(train_set, batch_size=args.batch_size, device=-1, repeat=False)
        
        print('\nLoading initial model for dataset {} ...'.format(args.dataset))
        model.load_state_dict(torch.load(args.snapshot))
        
        print('\nTraining new model ...')
        train(train_iter, val_iter, model, al_iter, args)
        
        print('\n\nLoading model {}, method {}'.format(al_iter, args.method))
        model.load_state_dict(torch.load('{}/{}/al_{}_{}.pt'.format(args.method, args.save_dir, args.dataset, al_iter)))
        
        acc, loss = evaluate(val_iter, model, args)
        acc = acc.cpu()
        
        if args.hist:
            hist = hist.cpu()
            with open('histogram/{}_{}.csv'.format(args.method,args.dataset), mode='a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
                csvwriter.writerow([hist.numpy()])
        
        else:
            if args.test_inc: 
                with open('accuracies/{}_{}_{}inc.csv'.format(args.method,args.dataset, args.inc), mode='a') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow([len(train_set), acc.numpy(), loss])
            elif args.test_preds:
                with open('accuracies/{}_{}_{}preds.csv'.format(args.method,args.dataset, args.num_preds), mode='a') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow([len(train_set), acc.numpy(), loss])
            else: 
                with open('accuracies/{}_{}_{}.csv'.format(args.method,args.dataset, avg_iter), mode='a') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csvwriter.writerow([len(train_set), acc.numpy()])
                
        

        al_iter += 1
        


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
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy, avg_loss


def save(model, save_dir, save_prefix, steps, round_, args, al=False):
    if al: 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_{}_{}.pt'.format(save_prefix, args.dataset, round_)
        torch.save(model.state_dict(), save_path)
    else: 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)
    
