#!/bin/python2
# coding: utf-8
"""
This code implements a k-fold cross validation performance analysis of 
convolutional neural networks for tumor classification using Theano and 
Scikit Learn. 

The DNN part is largely based on Theano's tutorial 
http://deeplearning.net/software/theano/tutorial/examples.html
and on the deeplearning.net tutorials
http://deeplearning.net/tutorial/

Copyright 2017 by Branislav Gerazov.

See the file LICENSE for the licence associated with this software.

Author:
   Branislav Gerazov, April 2017
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import cPickle as pickle
import theano as th
import theano.tensor as T
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano.d3viz as d3v
#from PIL import Image
import os
import sys

import dlutils as dl
from theano.tensor.shared_randomstreams import RandomStreams
import argparse


#%% parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", choices=['spectrograms','spectrograms_full'],
                    default='spectrograms', dest='dataset',
                    help='dataset used')

parser.add_argument("-r", "--regulizer", 
                    choices=['l2', 'dropout', 'l2dropout', 'early'],
                    default='dropout', dest='regulizer',
                    help='regulizer used')
parser.add_argument("--l2", type=float,
                    default=0.001, dest='l2',
                    help='l2 regularization coefficient')
parser.add_argument("--drop", type=int,
                    default=0.5, dest='drop',
                    help='dropout prcnt')

parser.add_argument("-l", "--learning", choices=['const', 'linear', 'adam'],
                    default='adam', dest='learning',
                    help='learning rate used')

parser.add_argument("-c","--cnn", type=int,
                    default=2, dest='cnn',
                    help='Number of convolutional layers')
parser.add_argument("-k","--kernel", type=int,
                    default=3, dest='kernel',
                    help='Kernel size')
parser.add_argument("-f","--filters", type=int,
                    default=50, dest='number_of_filters',
                    help='Number of filters')
parser.add_argument("-b","--border", type=str,
                    default='valid', dest='border',
                    help='Border padding')

parser.add_argument("--all", dest='all_folds',
                    help='Do all 10 folds',
                    action='store_true')
parser.set_defaults(all_folds=False)


args = parser.parse_args()
dataset = args.dataset 
learning = args.learning
regulizer = args.regulizer
l2_value = args.l2
cnn_n_layers = args.cnn
number_of_filters = args.number_of_filters
kernel = args.kernel
border = args.border
all_folds = args.all_folds

#%% create output folder
output_folder = 'results_cnn_shape_{}_{}cnn_{}f_{}k_{}bord_{}'.format(
                regulizer, cnn_n_layers, number_of_filters, kernel, 
                border, dataset)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
else:
    count = 0
    while os.path.isdir(output_folder+'_'+str(count)):
        count += 1
    output_folder = output_folder+'_'+str(count)
    os.mkdir(output_folder)
output_folder = output_folder+'/'    

#%% time it
t0 = time.time()

#%% set random seeds
rng = np.random.RandomState(42)    
srng = RandomStreams(rng.randint(42))

#%% load data
if dataset == 'spectrograms':
    with open('sim_data_specs_resized.pkl','rb') as f:
        # pickle contents: targets_shape, targets_size, spectrograms_all
        target, _, data = pickle.load(f)    
        
        n_samples, n_rows, n_cols = data.shape
        n_feats = (n_rows, n_cols)
        rand_sample = rng.permutation(n_samples)
        data = data[rand_sample,:,:]  # mix data up
        target = target[rand_sample]
        shape_input = (n_rows, n_cols)
        pca_components=0
        use_scaler = True
        
elif dataset == 'spectrograms_full':
    with open('sim_data_specs.pkl','rb') as f:
        # pickle contents: targets_shape, targets_size, spectrograms_all
        target, _, data = pickle.load(f)    
        data = data[:,:,4:]  # the data is not trimmed
        n_samples, n_rows, n_cols = data.shape
        n_feats = (n_rows, n_cols)
        rand_sample = rng.permutation(n_samples)
        data = data[rand_sample,:,:]  # mix data up
        target = target[rand_sample]
        shape_input = (n_rows, n_cols)
        pca_components=0
        use_scaler = True

else:
    print('wrong data choice!')    
    
#%% convert to floatX for GPU
data = np.asarray(data, dtype=th.config.floatX)
target = np.asarray(target, dtype=th.config.floatX)

#%% define depth and width loops
master_stats = []

n_layers_range = np.array([2, 5, 9])  # fc layers
for n_layers in n_layers_range:
    print()
    print()
    print('############################################################')
    print('############################################################')
    print('Number of layers testing {}'.format(n_layers))

    inner_stats = []
    
    n_neurons_range = np.array([300, 1000])
    for n_neurons_hl in n_neurons_range:
        print()
        print()
        print('****************************************')
        print('****************************************')
        print('Number of layers testing {}'.format(n_layers))
        print('Number of neurons testing {}'.format(n_neurons_hl))    
        print()
        
#%% init cnn parameters
        # cnn layers
        layer_n_filters = [number_of_filters] * cnn_n_layers  # LeNet [20, 50]
        filter_sizes = [(kernel,kernel)] * cnn_n_layers  # LeNet 5x5 * 2
        borders = [border] * cnn_n_layers
        pool_strides = [(2,2)] * cnn_n_layers
        pool_activations = [T.nnet.relu] * cnn_n_layers
        if 'dropout' in regulizer:
            dropouts_cnn = [0.2] * cnn_n_layers 
        else:
            dropouts_cnn = [0] * cnn_n_layers
            
        cnn_layers = zip(layer_n_filters, filter_sizes, borders, 
                         pool_strides, pool_activations, dropouts_cnn)

        # fc layers
        layer_sizes = [n_neurons_hl] * (n_layers-1) + [1]
        activations = [T.nnet.relu] * (n_layers-1) + [T.nnet.sigmoid]
        if 'dropout' in regulizer:
            dropouts = [0.5] + [0.5] * (n_layers-1) 
            dropouts[-1] = 0  # no dropout in last layer!
        else:
            dropouts = [0] * n_layers 
        mlp_layers = zip(layer_sizes, activations, dropouts)
        dropouts = dropouts_cnn + dropouts
        
        # training hyper-params
        k_folds = 10
        test_perc = 1/k_folds
        do_all_folds = all_folds
        n_epochs = 6000
        batch_size = 32
        thresh = 0.5
        
        # learning algorithm
        if learning == 'const':
            learn = 0.01  # LeNet 0.01?
        elif learning == 'linear':            
            learn_0 = 0.03
            learn_final = 0.0003
            learn_stop_iter = 500
        elif learning == 'adam':  # after Goodfellow 16 Deep Learning
            learn = 0.00001  # default is 0.001
            mdecay_1 = 0.9  # exp decay for 1st moment
            mdecay_2 = 0.999  # exp decay for 2nd moment
            eps = 1e-8
        else:
            print('Learning method is not valid.')
       
        # regularisation
        reg_l1 = 0  # 0.01
        if 'l2' in regulizer:
            reg_l2 = l2_value
        else:
            reg_l2 = 0
        early_stopping = True
        early_validation_prcnt = 1/(k_folds-1)  # keep the same size as test set
        early_eval_iter = 1  # per epoch
        early_patience = 3000 # in epochs
        
        save_model = False  # save parameters of best model

        #%% make a train and test set 
        skf = StratifiedKFold(n_splits=k_folds, random_state=42)
        train_inds = []
        test_inds = []
        for train_ind, test_ind in skf.split(data, target):
            train_inds.append(train_ind)
            test_inds.append(test_ind)
            
        #%% Theano stuff
        print('Theano - init variables ...')
        x = T.tensor3('x')
        y = T.vector('y')

        #% construct theano graphs
        print('Theano - construct graphs...')
        n_train = train_ind.size
        n_batches = n_train // batch_size
        mlp, mlp_test = dl.construct_mlp(rng, srng, 
                                         x, (n_rows, n_cols), batch_size, 
                                         cnn_layers, mlp_layers, 
                                         dropouts)
        
        p_y = mlp[-1].output                             
        p_y = p_y.T
        y_pred = p_y > thresh

        # cross_entropy
        cross_ent = -y * T.log(p_y) - (1-y)*T.log(1-p_y)  
        # negative log likelihood
        # nll = -T.mean(T.log(p_y)[T.arange(y.shape[0]), y])
        
        weights = [layer.params[0] for layer in mlp]
        biases = [layer.params[1] for layer in mlp]
        
        # L1 regularisation
        L1 = reg_l1 * np.sum(np.asarray([abs(w).sum() for w in weights]))
        # L2 regularisation
        L2 = reg_l2 * np.sum(np.asarray([(w**2).sum() for w in weights]))
        
        y_cost = cross_ent.mean() + L1 + L2
        
        # compute gradient
        grads = [T.grad(y_cost, [w,b]) for w, b in zip(weights, biases)]
        w_grads = [grad[0] for grad in grads] 
        b_grads = [grad[1] for grad in grads]
        grad_norms = sum([T.mean(grad**2) for grad in w_grads])  # for monitoring 
                                                                 # gradient norm
        grad_norms = grad_norms / len(w_grads)
        grads = zip(weights, w_grads) + zip(biases, b_grads)
        
        # y_err = T.mean(T.neq(y_pred, y))
        y_acc = T.mean(T.eq(y_pred, y))
        
        #% test graph
        p_y_test = mlp_test[-1].output                             
        p_y_test = p_y_test.T
        y_pred_test = p_y_test > thresh
        # y_err = T.mean(T.neq(y_pred, y))
        y_acc_test = T.mean(T.eq(y_pred_test, y))
        
        #%% init shared input data to precompile graphs
        print('Theano - load rand data to GPU...')
        test_samples = int(n_samples * test_perc)
        x_test, y_test = dl.share_data(np.zeros((test_samples, 
                                                  n_feats[0],
                                                  n_feats[1])), 
                                          np.zeros(test_samples))
        train_samples = n_samples - test_samples
        if not early_stopping:
            x_train, y_train = dl.share_data(np.zeros((train_samples, 
                                                        n_feats[0],
                                                        n_feats[1])), 
                                          np.zeros(train_samples))
        else:
            val_samples = int(train_samples * early_validation_prcnt)
            train_samples = train_samples - val_samples
            x_train, y_train = dl.share_data(np.zeros((train_samples,
                                                        n_feats[0],
                                                        n_feats[1])), 
                                          np.zeros(train_samples))
            x_val, y_val = dl.share_data(np.zeros((val_samples,
                                                    n_feats[0],
                                                    n_feats[1])), 
                                          np.zeros(val_samples))
        
        #%% compile
        print('Theano - compile graphs...')
        batch_ind = T.lscalar(name='batch_ind')  # to index batches 
                                                 # already in memory
        
        if learning == 'const':
            updates = [(p, p - learn * p_grad) for p, p_grad in grads]  
            
        elif learning == 'linear':
            iteration = th.shared(np.asarray(0, dtype=th.config.floatX), 
                                  name='iteration', borrow=True) 
            learn_coef = (iteration / n_batches)/learn_stop_iter
            learn_a = T.switch(T.lt(learn_coef, 1), 
                               learn_coef, np.array(1, dtype=th.config.floatX))
            learn = (1 - learn_a) * learn_0 + learn_a * learn_final
            updates = [(iteration, iteration + 1)] + \
                      [(p, p - learn * p_grad) for p, p_grad in grads]
                      
        elif learning == 'adam':
            # following Goodfellow "Deep Learning" 2016 and 
            # Radford https://gist.github.com/Newmu/acb738767acb4788bac3            
            iteration = th.shared(np.asarray(0, dtype=th.config.floatX), 
                                  name='iteration', borrow=True) 
            iteration_new = iteration + 1
            scale = T.sqrt(1 - mdecay_2**iteration_new) / (1 - mdecay_1**iteration_new)
            learn_scaled = scale * learn
            updates = []
            for p, p_grad in grads:
                moment_1 = th.shared(np.zeros(p.get_value().shape).astype(
                                                    dtype=th.config.floatX), 
                                     name='moment_1_'+p.name, borrow=True)
                moment_2 = th.shared(np.zeros(p.get_value().shape).astype(
                                                    dtype=th.config.floatX),
                                     name='moment_2_'+p.name, borrow=True)
                moment_1_new = mdecay_1 * moment_1 + (1-mdecay_1) * p_grad
                moment_2_new = mdecay_2 * moment_2 + (1-mdecay_2) * T.sqr(p_grad)
                update = moment_1_new / (T.sqrt(moment_2_new) + eps)
                updates.append((moment_1, moment_1_new))    
                updates.append((moment_2, moment_2_new))
                updates.append((p, p - learn_scaled * update))
            
            updates.append((iteration, iteration_new))
                      
        thf_train = th.function([batch_ind], [y_pred, y_acc, grad_norms],
                                givens={x : x_train[batch_ind * batch_size : 
                                                    (batch_ind+1) * batch_size],
                                        y : y_train[batch_ind * batch_size : 
                                                    (batch_ind+1) * batch_size]},
                                updates=updates)
        
#        Test functions should be redone to use batches
        thf_accuracy_test = th.function([batch_ind], y_acc_test, 
                                       givens={x: x_test[batch_ind * batch_size : 
                                                    (batch_ind+1) * batch_size],
                                               y: y_test[batch_ind * batch_size : 
                                                    (batch_ind+1) * batch_size]})
        thf_pred_test = th.function([batch_ind], y_pred_test, 
                                       givens={x: x_test[batch_ind * batch_size : 
                                                    (batch_ind+1) * batch_size]})
        if early_stopping:
            thf_accuracy_val = th.function([batch_ind], y_acc_test, 
                                           givens={x: x_val[batch_ind * batch_size : 
                                                       (batch_ind+1) * batch_size],
                                                   y: y_val[batch_ind * batch_size : 
                                                       (batch_ind+1) * batch_size]})   

#%% plot graphs
#        print('Theano - plotting graphs and profiling...')
#        th.printing.pydotprint(thf_train, 'lis_mlp_adam_train.png')
##    #    os.system('gwenview lis_mlp_train.png')
#        th.printing.pydotprint(thf_pred_test, 'lis_mlp_test.png')
##    #    os.system('gwenview lis_mlp_test.png')
#        d3v.d3viz(thf_pred_test, 'lis_mlp_pred.html')
#        os.system('firefox lis_mlp_pred.html')
#    #    
##    #    th.printing.pydotprint(thf_train, 'lis_mlp_train.png')
##        #Image.open('lis_logreg.png').show()
##        #os.system('gwenview lis_logreg.png')
#        d3v.d3viz(thf_train, 'lis_mlp_adam_train.html')
#        os.system('firefox lis_mlp_adam_train.html')
#        predict_profiled = th.function([x], y_pred, profile=True)
#        x_profile = rng.normal(0, 1, (32, 28, 28)).astype(th.config.floatX)
##        y1 = thf_predict_test()
##    ##    thf_accuracy_test(y1)
#        y_profile = predict_profiled(x_profile)
#        d3v.d3viz(predict_profiled, 'lis_mlp_pred_profiled.html')
#        os.system('firefox lis_mlp_pred_profiled.html')
        
        #%% k-fold loop
        train_acc = np.full((k_folds,n_epochs), np.nan)
        grad_folds = np.full((k_folds,n_epochs), np.nan)
        test_acc = np.full((k_folds,n_epochs), np.nan)
        val_acc = np.full((k_folds,n_epochs), np.nan)
        prec = np.full(k_folds, np.nan)
        rec = np.full(k_folds, np.nan)
        acc = np.full(k_folds, np.nan)
        fscore = np.full(k_folds, np.nan)
        max_epoch = 0  # for early stopping plots
        #%% for cell by cell execution in Spyder
        for fold, (train_ind, test_ind) in enumerate(zip(train_inds, test_inds)):
        #train_ind, test_ind = zip(train_inds, test_inds)[0]
        #fold = 0
        #%%
            print()
            print('K-fold: {} / {}'.format(fold+1, k_folds))
            data_train, data_test, target_train, target_test = data[train_ind,:], \
                                                               data[test_ind,:], \
                                                               target[train_ind], \
                                                               target[test_ind]
            n_train = data_train.shape[0]
            n_test = data_test.shape[0]  
            
            #%% now split train data to train and validate
            if early_stopping:
                data_train, data_val, target_train, target_val = \
                                        train_test_split(data_train, target_train,
                                                         test_size=early_validation_prcnt,
                                                         stratify=target_train, 
                                                         random_state=42)
                n_train = data_train.shape[0]
                n_val = data_val.shape[0]
                n_test = data_test.shape[0]  
            
            #%% upload data to GPU
            print('Theano - load fold data to GPU...')
            x_train.set_value(data_train, borrow=True)
            y_train.set_value(target_train, borrow=True)
            x_test.set_value(data_test, borrow=True)
            y_test.set_value(target_test, borrow=True)
            if early_stopping:
                x_val.set_value(data_val, borrow=True)
                y_val.set_value(target_val, borrow=True)
                
            #%% reset w and b
            for layer in mlp:
                layer.init_params(rng, reset=True)
            
            #%% train
            print('Theano - train...')
            t_start = time.time()
            n_batches = train_samples//batch_size
            n_batches_test = test_samples//batch_size  
            batch_accs = np.zeros(n_batches)
            batch_grads = np.zeros(n_batches)
            batch_accs_test = np.zeros(n_batches_test)
            if early_stopping:
                n_batches_val = val_samples//batch_size
                batch_accs_val = np.zeros(n_batches_val)
            prediction = np.zeros(n_batches_test*batch_size)
            val_err_min = np.inf
            n_check = 0
            #max_epoch = 0  # for early stopping plots
            epoch_stop = 0
            for epoch in range(n_epochs):
#                print()
#                sys.stdout.write("\r"+'epoch: '+str(epoch)+'\r')
                #sys.stdout.write(".")
                #sys.stdout.flush()
                for batch in range(n_batches):
                    pred_batch, acc_batch, grad_batch = thf_train(batch)
                    batch_grads[batch] = grad_batch
                    batch_accs[batch] = acc_batch
                
                train_acc[fold, epoch] = np.mean(batch_accs)
                grad_folds[fold, epoch] = np.mean(batch_grads)
                
                if not early_stopping:
                    for batch in range(n_batches_test):
                        batch_accs_test[batch] = thf_accuracy_test(batch)
                        prediction[batch*batch_size:
                                   (batch+1)*batch_size] = thf_pred_test(batch)
                    test_acc[fold, epoch] = np.mean(batch_accs_test)
                    
                if early_stopping and (epoch+1) % early_eval_iter == 0:
                    for batch in range(n_batches_val):
                        batch_accs_val[batch] = thf_accuracy_val(batch)
                        
                    val_acc_curr = np.mean(batch_accs_val)
                    val_acc[fold, epoch] = val_acc_curr
                    val_err_curr = 1 - val_acc_curr
        #            print('epoch {}, validation error {:.4f}'.format(epoch, val_err_curr))
                    if val_err_curr < val_err_min:
                        val_err_min = val_err_curr
                        epoch_best = epoch
                        n_check = 0
                        # evaluate performance on test
                        for batch_test in range(n_batches_test):
                            batch_accs_test[batch_test] = thf_accuracy_test(batch_test)
                            prediction[batch_test*batch_size:
                                   (batch_test+1)*batch_size] = thf_pred_test(batch_test)
                        
                        test_acc_best = np.mean(batch_accs_test)
                        if save_model:  # save best model
                            with open('{}best_model_fold{}.pkl'.format(output_folder,fold), 
                                      'wb') as save_file:
                                p_best = [p.get_value(borrow=True) for p in weights + biases]
                                pickle.dump(p_best, save_file, -1)  
                    else:
                        n_check += 1
                        if n_check > early_patience:
                            epoch_stop = epoch
                            print()
                            print('Activated early stopping at epoch {}.'.format(epoch))
                            print('Best validation error at epoch {} with validation error of {:.4f} and test accuracy of {:.4f}'.format(
                                    epoch_best, val_err_min, test_acc_best))
                            break
            print()
            print('Time used for training {} s.'.format(time.time()-t_start))
            if early_stopping and epoch_stop > max_epoch:
                max_epoch = epoch_stop
            else:
                max_epoch = n_epochs
        
            #%% accuracy
            N = prediction.size
            target_test_batchwise = target_test[:N]  # not all samples might fit in
            TP = np.logical_and(prediction, 
                                target_test_batchwise)
            TP = np.sum(TP)
            FP = np.logical_and(prediction, 
                                np.logical_not(target_test_batchwise))
            FP = np.sum(FP)
            TN = np.logical_and(np.logical_not(prediction), 
                                np.logical_not(target_test_batchwise))
            TN = np.sum(TN)
            FN = np.logical_and(np.logical_not(prediction), 
                                target_test_batchwise)
            FN = np.sum(FN)
            
            print('Theano results fold {}'.format(fold+1))
#            print('TP = {} / {}'.format(TP, N))
#            print('TN = {} / {}'.format(TN, N))
#            print('FP = {} / {}'.format(FP, N))
#            print('FN = {} / {}'.format(FN, N))
            
            prec[fold] = TP / (TP + FP) # precision
            rec[fold] = TP / (TP + FN) # recall
            acc[fold] = (TP + TN) / N # accuracy
            fscore[fold] = 2 * prec[fold] * rec[fold] / (prec[fold] + rec[fold])
            print('accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, f-score = {:.4f}'.format(
                                                                                    acc[fold],
                                                                                    prec[fold], 
                                                                                    rec[fold], 
                                                                                    fscore[fold]))
            if not do_all_folds:
                print('Ending k-folds.')
                break
            
        #%% end of kfold loop
        stats = [np.nanmean(acc), np.nanmean(prec), np.nanmean(rec), np.nanmean(fscore)]
        print('==================')
        print('Theano average results all folds')
        print('accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, f-score = {:.4f}'.format(
                np.nanmean(acc), np.nanmean(prec), np.nanmean(rec), np.nanmean(fscore)))
        
        #%% plotting
        import warnings
        warnings.filterwarnings("ignore")
        plt.figure()
        plt.plot(train_acc.T, 'b', lw=4, alpha=.25)
        if not early_stopping:
            plt.plot(test_acc.T, 'r', lw=4, alpha=.25)
        else:
            plt.plot(np.arange(val_acc.shape[1])*early_eval_iter,
                     val_acc.T, 'g', lw=4, alpha=.25)    
        
        plt.plot(np.nanmean(train_acc, axis=0), 'k', lw=4)
        plt.plot(np.nanmean(train_acc, axis=0), 'b', lw=3, label='train')
        if not early_stopping:
            plt.plot(np.nanmean(test_acc, axis=0), 'k', lw=4)
            plt.plot(np.nanmean(test_acc, axis=0), 'r', lw=3, label='test')
        else:
            plt.plot(np.arange(val_acc.shape[1])*early_eval_iter,
                     np.nanmean(val_acc, axis=0), 'k', lw=4)
            plt.plot(np.arange(val_acc.shape[1])*early_eval_iter,
                     np.nanmean(val_acc, axis=0), 'g', lw=3, label='validation')
            
        plt.legend(loc=4)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid('on')
        #if not early_stopping:
        plt.axis([0,max_epoch,0,1])
        #else:
        #    plt.axis([0,epoch_stop,0,1])
        plt.savefig(output_folder+'cnn_acc_cnn{:2d}_fc{:2d}_neurons{:3d}_{}l2_{}dropout.png'.format(
                    len(cnn_layers), len(mlp_layers), n_neurons_hl, reg_l2, dropouts[-2]))
        
        plt.figure()
        plt.plot(grad_folds.T, 'g', lw=4, alpha=.25)
        
        plt.plot(np.mean(grad_folds, axis=0), 'k', lw=4)
        plt.plot(np.nanmean(grad_folds, axis=0), 'g', lw=3)
            
        plt.legend(loc=4)
        plt.ylabel('Gradient norm')
        plt.xlabel('Epoch')
        plt.grid('on')
        
        plt.savefig(output_folder+'cnn_grads_cnn{:2d}_fc{:2d}_neurons{:3d}_{}l2_{}dropout.png'.format(
                    len(cnn_layers), len(mlp_layers), n_neurons_hl, reg_l2, dropouts[-2]))
        #%% Inner hyper parameter loop
        inner_stats.append(stats)
    
    #%% end hyper parameter loop
    print()
    print('==================')
    print('==================')
    print('Inner stats for all widths.')
    for i, stats in enumerate(inner_stats):
        print('{:3d},{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                n_neurons_range[i], stats[0], stats[1], stats[2], stats[3]))
        
    inner_stats = np.asarray(inner_stats)
    plt.figure()
    plt.plot(n_neurons_range, inner_stats[:,0], 'b', lw=4, alpha=.25)
    plt.ylabel('Accuracy')
    plt.xlabel('Width (n_neurons)')
    plt.grid('on')
    filename = output_folder+'cnn_acc_cnn{:2d}_fc{:2d}_neurons{:3d}_{}l2_{}dropout.png'.format(
                    len(cnn_layers), len(mlp_layers), n_neurons_hl, reg_l2, dropouts[-2])
    hyper_params = [n_epochs, batch_size, n_feats, early_stopping, reg_l1, reg_l2, dropouts,
                    n_layers]
    plt.savefig(filename+'.png')
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump((hyper_params, n_neurons_range, inner_stats), f, -1)

    #%% Hyper parameter loop
    master_stats.append(inner_stats)

#%% end hyper parameter loop
print()
print('==================')
print('==================')
print('Master stats for all depths.')
for l, inner_stats in enumerate(master_stats):
    for i, stats in enumerate(inner_stats):
        print('{:2d}, {:3d},{:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(
                n_layers_range[l], n_neurons_range[i], stats[0], stats[1], stats[2], stats[3]))
    
master_stats = np.asarray(master_stats)
plt.figure()
for i in range(n_neurons_range.size):
    n_neurons = n_neurons_range[i]
    plt.plot(n_layers_range, master_stats[:,i,0], lw=4, alpha=.25, label=str(n_neurons))
plt.ylabel('Accuracy')
plt.xlabel('Depth (No. of hidden layers)')
plt.grid('on')
plt.legend(loc=1)
filename = output_folder+'cnn_acc_depth_width_epochs{}_{}l2_{}drop'.format(
            n_epochs, reg_l2, dropouts[-2])
plt.savefig(filename+'.png')

hyper_params = [n_epochs, batch_size, n_feats, 
                early_stopping, reg_l1, reg_l2,
                dropouts]
with open(filename+'.pkl', 'wb') as f:
    pickle.dump((hyper_params, n_layers_range, n_neurons_range, master_stats), f, -1)
    
#%%
print('Total run time is {} min.'.format((time.time()-t0)/60))
