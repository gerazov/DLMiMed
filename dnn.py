#!/bin/python2
# coding: utf-8
"""
This code implements a k-fold cross validation performance analysis of deep 
neural networks for tumor classification using Theano and Scikit Learn. 

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
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import cPickle as pickle
import theano as th
import theano.tensor as T
import time

# for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano.d3viz as d3v
import os
import sys

import dlutils as dl
from theano.tensor.shared_randomstreams import RandomStreams
import argparse


#%% parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", choices=['raw', 'spectrograms'],
                    default='raw', dest='dataset',
                    help='dataset used')
parser.add_argument("-p", "--pca", type=int,
                    default=50, dest='pca',
                    help='no. of pca components for raw signal')
parser.add_argument("-r", "--regulizer", 
                    choices=['l2', 'dropout', 'l2dropout', 'early'],
                    default='dropout', dest='regulizer',
                    help='regulizer used')
parser.add_argument("-l", "--learning", choices=['const', 'linear', 'adam'],
                    default='adam', dest='learning',
                    help='learning algorithm used')
parser.add_argument("--drop", type=float,
                    default=0.5, dest='drop',
                    help='dropout prcnt')

args = parser.parse_args()
dataset = args.dataset 
learning = args.learning
pca_components = args.pca
regulizer = args.regulizer
drop = args.drop

#%% create output folder
output_folder = 'results_dnn_shape_{}_{}_{}_{}pca_10folds'.format(
                dataset, regulizer, learning, pca_components)
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
if dataset == 'raw':
    with open('sim_data.pkl','rb') as f:
        # pickle contents: targets_shape, targets_size, signals
        target, _, data = pickle.load(f)
        n_samples, n_feats = data.shape
        rand_sample = rng.permutation(n_samples)
        data = data[rand_sample,:]  # mix data up
        target = target[rand_sample]
        n_feats = pca_components
        use_scaler = False

elif dataset == 'spectrograms':
    # this uses flattened versions of the spectrograms of the backscatter signals
    with open('sim_data_specs_resized.pkl','rb') as f:
        # pickle contents: targets_shape, targets_size, spectrograms_all
        target, _, data = pickle.load(f)    
        n_samples, n_rows, n_cols = data.shape
        rand_sample = rng.permutation(n_samples)
        data = data[rand_sample,:,:]  # mix data up
        target = target[rand_sample]
        data = np.reshape(data, (n_samples,-1))  # flatten
        n_feats = data.shape[1]
        pca_components=0
        use_scaler = True

else:
    print('wrong data choice!')    
    
#%% convert to floatX for GPU
data = np.asarray(data, dtype=th.config.floatX)
target = np.asarray(target, dtype=th.config.floatX)

#%% define depth and width loops
master_stats = []

n_layers_range = [3, 4, 5]
for n_layers in n_layers_range:
    print()
    print()
    print('############################################################')
    print('############################################################')
    print('Number of layers testing {}'.format(n_layers))

    inner_stats = []
    n_neurons_range = np.array([300, 1000, 1500, 3000])
    for n_neurons_hl in n_neurons_range:
        print()
        print()
        print('****************************************')
        print('****************************************')
        print('Number of layers testing {}'.format(n_layers))
        print('Number of neurons testing {}'.format(n_neurons_hl))    
        print()
        
#%% init dnn parameters
        cnn_layers = None
        layer_sizes = [n_neurons_hl] * (n_layers-1) + [1]  # two class output
        activations = [T.nnet.relu] * (n_layers-1) + [T.nnet.sigmoid]
        if 'dropout' in regulizer:
            dropouts = [drop] + [drop] * (n_layers-1)  # first layer can be different
            dropouts[-1] = 0  # no dropout in last layer!
        else:
            dropouts = [0] * n_layers 
        mlp_layers = zip(layer_sizes, activations, dropouts)
        
        # training hyper-parameters
        k_folds = 10
        test_perc = 1/k_folds
        do_all_folds = True
        n_epochs = 3000
        batch_size = 32
        thresh = 0.5  # for prediction class decision
        
        # learning algorithm
        if learning == 'const':
            learn = 0.001
        elif learning == 'linear':            
            learn_0 = 0.03
            learn_final = 0.0003
            learn_stop_iter = 500
        elif learning == 'adam':
            learn = 0.001  # 0.001 default
            mdecay_1 = 0.9  # exp decay for 1st moment
            mdecay_2 = 0.999  # exp decay for 2nd moment
            eps = 1e-8
        else:
            print('Learning method is not valid.')
        
        # regularisation
        reg_l1 = 0  # 0.01
        if 'l2' in regulizer:
            reg_l2 = .001
        else:
            reg_l2 = 0
        early_stopping = True
        early_validation_prcnt = 1/(k_folds-1)  # keep the same size as test set
        early_eval_iter = 1  # per epoch
        early_patience = 500 # in epochs
        
        save_model = False  # save parameters of best model
        save_feats = False  # save output of penultimate layer to be used as features


        #%% make a train and test set 
        skf = StratifiedKFold(n_splits=k_folds, random_state=42)
        train_inds = []
        test_inds = []
        for train_ind, test_ind in skf.split(data, target):
            train_inds.append(train_ind)
            test_inds.append(test_ind)
            
        #%% Theano stuff
        #% init shared input data to precompile graphs
        print('Theano - load rand data to GPU...')
        test_samples = int(n_samples * test_perc)
        x_test, y_test = lis.share_data(np.zeros((test_samples, n_feats)), 
                                          np.zeros(test_samples))
        train_samples = n_samples - test_samples
        if not early_stopping:
            x_train, y_train = lis.share_data(np.zeros((train_samples, n_feats)), 
                                          np.zeros(train_samples))
        else:
            val_samples = int(train_samples * early_validation_prcnt)
            train_samples = train_samples - val_samples
            x_train, y_train = lis.share_data(np.zeros((train_samples,
                                                        n_feats)), 
                                          np.zeros(train_samples))
            x_val, y_val = lis.share_data(np.zeros((val_samples,
                                                    n_feats)), 
                                          np.zeros(val_samples))
        if save_feats:
            x_all, y_all = lis.share_data(np.zeros((n_samples,
                                                    n_feats)), 
                                          np.zeros(n_samples))
            
        print('Theano - init variables ...')
        x = T.matrix('x')
        y = T.vector('y')
        
        #%% construct theano graphs
        print('Theano - construct graphs...')

        n_train = train_ind.size
        n_batches = n_train // batch_size
        
        mlp, mlp_test = dl.construct_dnn(rng, x, n_feats, mlp_layers, dropouts=dropouts)
        
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
        
        # test graph
        p_y_test = mlp_test[-1].output     
        if save_feats:
            end_feats = mlp_test[-2].output                        
        p_y_test = p_y_test.T
        y_pred_test = p_y_test > thresh
        # y_err = T.mean(T.neq(y_pred, y))
        y_acc_test = T.mean(T.eq(y_pred_test, y))
            
        #%% compile
        print('Theano - compile graphs...')
        batch_ind = T.scalar(name='batch_ind', dtype='int64')  # to index batches 
                                                               # already in memory
        
        if learning == 'const':
            updates = [(p, p - learn * p_grad) for p, p_grad in grads]  
            
        elif learning == 'linear':
            iteration = th.shared(np.array(0, dtype=th.config.floatX), 
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
            iteration = th.shared(np.array(0, dtype=th.config.floatX), 
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

#        Test functions should be redone to use batches - already done in cnn.py
        thf_predict_test = th.function([], y_pred_test, givens={x: x_test[:,:]})
        thf_accuracy_test = th.function([y_pred_test], y_acc_test, givens={y: y_test[:]})
        if early_stopping:
            thf_predict_val = th.function([], y_pred_test, givens={x: x_val[:,:]})
            thf_accuracy_val = th.function([y_pred_test], y_acc_test, givens={y: y_val[:]}) 
        if save_feats:
            thf_end_feats_all = th.function([], end_feats, givens={x: x_all[:,:]})    
            thf_end_feats_train= th.function([], end_feats, givens={x: x_train[:,:]})    
            thf_end_feats_val = th.function([], end_feats, givens={x: x_val[:,:]})    
            thf_end_feats_test = th.function([], end_feats, givens={x: x_test[:,:]})    

       #%% plot graphs
##        print('Theano - plotting graphs and profiling...')
#        th.printing.pydotprint(thf_train, 'lis_mlp_adam_train.png')
##    #    os.system('gwenview lis_mlp_train.png')
##        th.printing.pydotprint(thf_predict_test, 'lis_mlp_test.png')
##    #    os.system('gwenview lis_mlp_test.png')
##        d3v.d3viz(thf_predict_test, 'lis_mlp_pred.html')
##        os.system('firefox lis_mlp_pred.html')
##    #    
##    #    th.printing.pydotprint(thf_train, 'lis_mlp_train.png')
##        #Image.open('lis_logreg.png').show()
##        #os.system('gwenview lis_logreg.png')
#        d3v.d3viz(thf_train, 'lis_mlp_adam_train.html')
#        os.system('firefox lis_mlp_adam_train.html')
##    #    predict_profiled = th.function([x], y_pred, profile=True)
##    #    x_profile = rng.normal(0, 1, (train_samples, pca_components)).astype(th.config.floatX)
##    ##    y1 = thf_predict_test()
##    ##    thf_accuracy_test(y1)
##    #    y_profile = predict_profiled(x_profile)
##    #    d3v.d3viz(predict_profiled, 'lis_mlp_pred_profiled.html')
##    #    os.system('firefox lis_mlp_pred_profiled.html')
        
        #%% k-fold loop
        train_acc = np.full((k_folds,n_epochs), np.nan)
        grad_folds = np.full((k_folds,n_epochs), np.nan)
        test_acc = np.full((k_folds,n_epochs), np.nan)
        val_acc = np.full((k_folds,n_epochs), np.nan)
        prec = np.full(k_folds, np.nan)
        rec = np.full(k_folds, np.nan)
        acc = np.full(k_folds, np.nan)
        fscore = np.full(k_folds, np.nan)
        #%% for cell by cell execution in Spyder
        for fold, (train_ind, test_ind) in enumerate(zip(train_inds, test_inds)):
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
            
            #%% apply pca to the data
            if pca_components:
                print('applying PCA with {} components...'.format(pca_components))
                pca = PCA(n_components=pca_components, whiten=True, copy=True)
                pca.fit(data_train)
                # print(pca.explained_variance_ratio_) 
                data_pca_train = np.asarray(pca.transform(data_train), dtype=th.config.floatX)
                data_pca_test = np.asarray(pca.transform(data_test), dtype=th.config.floatX)
                if early_stopping:
                    data_pca_val = np.asarray(pca.transform(data_val), dtype=th.config.floatX)
                if save_feats:
                    data_pca = np.asarray(pca.transform(data), dtype=th.config.floatX)
            else:
                data_pca_train = data_train
                data_pca_test = data_test
                if early_stopping:
                    data_pca_val = data_val
                
            #%% apply scaler (not needed with PCA)
            if use_scaler:
                print('applying Robust scaler...')
                scaler = RobustScaler() 
                scaler.fit(data_pca_train)
                data_pca_train = np.asarray(scaler.transform(data_pca_train), 
                                            dtype=th.config.floatX)
                data_pca_test = np.asarray(scaler.transform(data_pca_test), 
                                           dtype=th.config.floatX)
                if early_stopping:
                    data_pca_val = np.asarray(scaler.transform(data_pca_val), 
                                              dtype=th.config.floatX)    
            
            #%% upload data to GPU
            print('Theano - load fold data to GPU...')
            x_train.set_value(data_pca_train, borrow=True)
            y_train.set_value(target_train, borrow=True)
            x_test.set_value(data_pca_test, borrow=True)
            y_test.set_value(target_test, borrow=True)
            if early_stopping:
                x_val.set_value(data_pca_val, borrow=True)
                y_val.set_value(target_val, borrow=True)
            if save_feats:
                x_all.set_value(data_pca, borrow=True)

            #%% reset w and b
            for layer in mlp:
                layer.init_params(rng, reset=True)
            
            #%% train
            print('Theano - train...')
            t_start = time.time()
            n_batches = n_train//batch_size
            n_batches_test = n_test//batch_size  # not implemented 

            batch_accs = np.zeros(n_batches)
            batch_grads = np.zeros(n_batches)
            val_err_min = np.inf
            n_check = 0
            max_epoch = 0  # for early stopping plots
            for epoch in range(n_epochs):
        #        print()
        #        print('epoch: {}/{} \r'.format(epoch, n_epochs), end='')
        #            sys.stdout.write("\r"+'epoch: '+str(epoch)+'\r')
        #            sys.stdout.flush()
                for batch in range(n_batches):
#                    print('.', end='')
                    pred_batch, acc_batch, grad_batch = thf_train(batch)
                    batch_grads[batch] = grad_batch
                    batch_accs[batch] = acc_batch
                train_acc[fold, epoch] = np.mean(batch_accs)
                grad_folds[fold, epoch] = np.mean(batch_grads)
                if not early_stopping:
                    prediction = thf_predict_test()
                    test_acc[fold, epoch] = thf_accuracy_test(prediction)
#                    sys.stdout.write(".")
#                    sys.stdout.flush()
#                    print(thf_debug())
                if early_stopping and (epoch+1) % early_eval_iter == 0:
                    prediction_val = thf_predict_val()
                    val_acc_curr = thf_accuracy_val(prediction_val)
                    val_acc[fold, epoch] = val_acc_curr
                    val_err_curr = 1 - val_acc_curr
        #            print('epoch {}, validation error {:.4f}'.format(epoch, val_err_curr))
#                    sys.stdout.write(".")
#                    sys.stdout.flush()
                    if val_err_curr < val_err_min:
                        val_err_min = val_err_curr
                        epoch_best = epoch
                        n_check = 0
                        # evaluate performance on test
                        prediction = thf_predict_test() 
                        
                        if save_model:  # save best model
                            with open('{}best_model_fold{}.pkl'.format(output_folder,fold), 
                                      'wb') as save_file:
                                p_best = [p.get_value(borrow=True) for p in weights + biases]
                                pickle.dump(p_best, save_file, -1) 
                        if save_feats:
                            with open('{}best_feats_fold{}.pkl'.format(output_folder,fold), 
                                      'wb') as save_file:
                                best_feats =  rand_sample, thf_end_feats_all(), \
                                              thf_end_feats_train(), thf_end_feats_val(), \
                                              thf_end_feats_test()
                                targets = target_train, target_val, target_test
                                pickle.dump((best_feats, targets), save_file, -1) 
                    else:
                        n_check += 1
                        if n_check > early_patience:
                            epoch_stop = epoch
                            print()
                            print('Activated early stopping at epoch {}.'.format(epoch))
                            print('Best validation error at epoch {} with error of {:.4f}'.format(
                                    epoch_best, val_err_min))
                            break

            print()
            print('Time used for training {} s.'.format(time.time()-t_start))
            if early_stopping and epoch_stop > max_epoch:
                max_epoch = epoch_stop
            else:
                max_epoch = n_epochs
        
            #%% accuracy
            N = target_test.size
            TP = np.logical_and(prediction, target_test)
            TP = np.sum(TP)
            FP = np.logical_and(prediction, np.logical_not(target_test))
            FP = np.sum(FP)
            TN = np.logical_and(np.logical_not(prediction), np.logical_not(target_test))
            TN = np.sum(TN)
            FN = np.logical_and(np.logical_not(prediction), target_test)
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
        plt.axis([0,max_epoch,0,1])
        plt.savefig(output_folder+'mlp_acc_layers{:2d}_neurons{:3d}_feats{:03d}_{}l2_{}dropout.png'.format(
                    len(mlp), n_neurons_hl, n_feats, reg_l2, dropouts[-2]))
        
        #%% plotting
        plt.figure()
        plt.plot(grad_folds.T, 'g', lw=4, alpha=.25)
        
        plt.plot(np.nanmean(grad_folds, axis=0), 'k', lw=4)
        plt.plot(np.nanmean(grad_folds, axis=0), 'g', lw=3)
            
        plt.legend(loc=4)
        plt.ylabel('Gradient norm')
        plt.xlabel('Epoch')
        plt.grid('on')
        plt.axis([0,max_epoch,0,np.nanmax(grad_folds)])
        plt.savefig(output_folder+'mlp_grads_layers{:2d}_neurons{:3d}_feats{:03d}_{}l2_{}dropout.png'.format(
                    len(mlp), n_neurons_hl, n_feats, reg_l2, dropouts[-2]))
        
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
    filename = output_folder+'mlp_acc_width_{:02d}layers_epochs{}_feats{:03d}_{}l2_{:.1f}drop'.format(
               n_layers, n_epochs, n_feats, reg_l2, dropouts[-2])
    hyper_params = [n_epochs, batch_size, n_feats, early_stopping, reg_l1, reg_l2, dropouts,
                    n_layers]
    plt.savefig(filename+'.png')
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump((hyper_params, n_neurons_range, inner_stats), f, -1)

    #%% Hyper parameter loop stats
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
filename = output_folder+'mlp_acc_depth_width_epochs{}_feats{:03d}_{}l2_{}drop'.format(
            n_epochs, n_feats, reg_l2, dropouts[-2])
plt.savefig(filename+'.png')

hyper_params = [n_epochs, batch_size, n_feats, 
                early_stopping, reg_l1, reg_l2,
                dropouts]
with open(filename+'.pkl', 'wb') as f:
    pickle.dump((hyper_params, n_layers_range, n_neurons_range, master_stats), f, -1)

#%%
print('Total run time is {} min.'.format((time.time()-t0)/60))
