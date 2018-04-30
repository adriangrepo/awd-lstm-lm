import argparse
import time
import math
import os
import hashlib
import numpy as np
from splitcross import SplitCrossEntropyLoss
import torch

import torch.nn as nn
from torch.autograd import Variable

import data
import model

from utils import batchify, get_batch, repackage_hidden

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

randomhash = ''.join(str(time.time()).split('.'))

eval_batch_size = 10
test_batch_size = 1

def set_args_defaults():
    '''
       alpha=2, batch_size=20, beta=1, bptt=70, clip=0.25, cuda=True, data='data/penn', dropout=0.4, dropoute=0.1, dropouth=0.25, dropouti=0.4, emsize=400, epochs=5, log_interval=200, lr=30, model='LSTM', nhid=1150, nlayers=3, nonmono=5, optimizer='sgd', resume='', save='PTB.pt', seed=141, tied=True, wdecay=1.2e-06, wdrop=0.5, when=[-1]
    '''
    args = {
        'description':'PyTorch PennTreeBank RNN/LSTM Language Model',
         'data': 'data/penn/',
         'model': 'LSTM',
         'emsize':400,
         'nhid':1150,
         'nlayers':3,
         'lr':30,
         'clip':0.25,
         'epochs':8000,
         'batch_size':80,
         'bptt':70,
         'dropout':0.4,
         'dropouth':0.3,
         'dropouti':0.65,
         'dropoute':0.1,
         'wdrop':0.5,
         'seed':1111,
         'nonmono':5,
         'cuda': True,
         'log-interval':200,
         'randomhash': randomhash,
         'save':randomhash+'.pt',
         'alpha':2,
         'beta':1,
         'wdecay':1.2e-6,
         'resume':'',
         'optimizer': 'sgd',
         'when':[-1],
        'log_interval':200
    }
    return args

def ptb_words_lstm(args):
    #override relevant keys
    args['batch_size']=20
    args['data']= 'data/penn'
    args['dropouti']= 0.4
    args['dropouth']= 0.25
    args['seed']= 141
    args['epochs']= 5
    args['save'] = 'PTB.pt'
    return args

def ptb_words_finetune(args):
    args['batch_size']=20
    args['data']= 'data/penn'
    args['dropouti']= 0.4
    args['dropouth']= 0.25
    args['seed']= 141
    args['epochs']= 5
    args['save'] = 'PTB.pt'
    return args

def ptb_words_pointer(args):
    args['data']= 'data/penn'
    args['window']= 500
    args['theta']= 1.0
    args['lambdasm']= 0.1
    args['save'] = 'PTB.pt'
    args['bptt'] = 5000
    return args

def wikitext2_words_main(args):
    args['data']= 'data/wikitext-2'
    args['dropouth']= 0.2
    args['seed']= 1882
    args['epochs']= 750
    args['save'] = 'WT2.pt'
    return args

def wikitext2_words_finetune(args):
    args['data']= 'data/wikitext-2'
    args['dropouth']= 0.2
    args['seed']= 1882
    args['epochs']= 750
    args['save'] = 'WT2.pt'
    return args

def wikitext2_words_pointer(args):
    args['data']= 'data/wikitext-2'
    args['window']= 3785
    args['theta']= 0.662
    args['lambdasm']= 0.1279
    args['save'] = 'WT2.pt'
    args['bptt'] = 2000
    return args

def args_to_dot(args):
    args = dotdict(args)
    args.tied = True
    print(args.keys())
    return args

def model_save(fn, modl, criterion, optimizer):
    with open(fn, 'wb') as f:
        torch.save([modl, criterion, optimizer], f)

def model_load(fn):
    #global modl, criterion, optimizer
    with open(fn, 'rb') as f:
        modl, criterion, optimizer = torch.load(f)
    return modl, criterion, optimizer

def run_loader(args):
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)
    return corpus, train_data, val_data, test_data


def run_model_builder(corpus, args):
    # Build the model
    criterion = None

    ntokens = len(corpus.dictionary)
    print(f'ntokens: {ntokens}')
    modl = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

    optimizer = None
    if args.resume:
        print('Resuming model ...')
        modl, criterion, optimizer = model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        modl.dropouti, modl.dropouth, modl.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            from weight_drop import WeightDrop
            for rnn in modl.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

    if args.cuda:
        modl = modl.cuda()
        criterion = criterion.cuda()

    params = list(modl.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)
    return criterion, modl, optimizer, params


def evaluate(modl, criterion, corpus, data_source, args, batch_size = 10):
    '''
    :param modl:
    :param criterion:
    :param corpus:
    :param data_source:
    :param args:
    :param batch_size:
    :return: total_loss[0] / len(data_source)
    '''
    # Turn on evaluation mode which disables dropout.
    modl.eval()
    if args.model == 'QRNN':
        modl.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = modl.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = modl(data, hidden)
        total_loss += len(data) * criterion(modl.decoder.weight, modl.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(epoch, modl, corpus, train_data, args, params, optimizer, criterion):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN':
        modl.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = modl.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        modl.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = modl(data, hidden, return_h=True)
        raw_loss = criterion(modl.decoder.weight, modl.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
    return optimizer

def run_training(modl, corpus, train_data, args, params, val_data, eval_batch_size, criterion):
    # Training code
    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            optimizer = train(epoch, modl, corpus, train_data, args, params, optimizer, criterion)
            #at end of each epoch (here 3 batches making up 663)
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in modl.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(modl, criterion, corpus, val_data, args)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save, modl, criterion, optimizer)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in modl.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(modl, criterion, corpus, val_data, args, eval_batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save, modl, criterion, optimizer)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(modl.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch), modl, criterion, optimizer)
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return best_val_loss

def run_on_test(args, test_data, test_batch_size, corpus):

    # Load the best saved model.
    modl, criterion, optimizer = model_load(args.save)

    # Run on test data.
    test_loss = evaluate(modl, criterion, corpus, test_data, args, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)

def workflow():
    print('workflow')
    args = set_args_defaults()
    args = ptb_words_lstm(args)
    args = args_to_dot(args)
    corpus, train_data, val_data, test_data = run_loader(args)
    criterion, modl, optimizer, params = run_model_builder(corpus, args)
    best_val_loss = run_training(modl, corpus, train_data, args, params, val_data, eval_batch_size, criterion)
    print(f'best_val_loss: {best_val_loss}')
    run_on_test(args, test_data, test_batch_size, corpus)

if __name__ == "__main__":
    workflow()