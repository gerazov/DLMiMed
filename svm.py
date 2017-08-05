#!/bin/python2
# coding: utf-8
"""
This code implements SVM using the features from the DNN.

Copyright 2017 by Branislav Gerazov.

See the file LICENSE for the licence associated with this software.

Author:
   Branislav Gerazov, April 2017
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import cPickle as pickle
import time
from sklearn.svm import SVC

#%% time it
t0 = time.time()

#%% load data
feats = []
targets = []
for fold in range(10):
    with open('best_feats_shape/best_feats_fold{}.pkl'.format(fold),'rb') as f:
        best_feats, target = pickle.load(f) 
        rand_sample, feats_all, feats_train, feats_val, feats_test = best_feats 
        target_train, target_val, target_test = target 
        feats.append([feats_train, feats_val, feats_test])
        targets.append([target_train, target_val, target_test])

#%% master loop
Cs = np.array([300000])
gammas = np.array([0.0005, .001, .003, .01, .03, .1, .3, 1, 3]) 
accs_master = np.zeros((Cs.size, gammas.size, 1))

for i_C, C in enumerate(Cs[::-1]):
    print()
    print()
    print('############################################################')
    print('############################################################')
    print('C = {}'.format(C))
    for i_gamma, gamma in enumerate(gammas):
        print()
        print('****************************************')
        print('****************************************')
        print('Gamma = {}'.format(gamma))
        print()
        svm = SVC(C=C, kernel='rbf', gamma=gamma, 
                  probability=False, cache_size=1000, random_state=42)
        accs = []
        for fold in range(10):
            feats_train = np.vstack(feats[fold][0:2])
            targets_train = np.concatenate(targets[fold][0:2])
            feats_test = feats[fold][-1]
            targets_test = targets[fold][-1]
            
            svm.fit(feats_train, targets_train)
            acc = svm.score(feats_test, targets_test)
            print('Fold : {}, accuracy : {:.4f}'.format(fold, acc))
            accs.append(acc)
    
        #%
        print('Mean accuracy for all folds : {:.4f}'.format(np.mean(accs)))
        accs_master[i_C, i_gamma] = np.mean(accs)
        
#%% print out
print('############################################################')
print('############################################################')
print('final results')
for i_C, C in enumerate(Cs[::-1]):
    for i_gamma, gamma in enumerate(gammas):
        print('{},{},{}'.format(C, gamma, accs_master[i_C, i_gamma]))
    
    