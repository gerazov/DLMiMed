#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This code includes utility functions to build the neural networks.

The DNN part is largely based on Theano's tutorial 
http://deeplearning.net/software/theano/tutorial/examples.html
and on the deeplearning.net tutorials
http://deeplearning.net/tutorial/

Copyright 2017 by Branislav Gerazov.

See the file LICENSE for the licence associated with this software.

Author:
   Branislav Gerazov, April 2017
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
import os
from scipy import fftpack as fftp
from scipy import signal as sig
import scipy.interpolate as interp
import theano as th
import theano.tensor as T
from theano.tensor.signal import pool


class fc_layer(object):
    ''' 
    I wanted to avoid making a class but if you want to make the whole 
    network creation process dynamic you need it.
    '''
    def __init__(self, rng, srng, tensor_in, n_in, n_neurons=1, activation=None, 
                lvl=0, use_dropout=False, dropout=0, params=None):
        self.input = tensor_in
        self.n_in = n_in
        self.n_neurons = n_neurons
        self.activation = activation
        self.lvl = lvl
        
        if params is None:
            self.init_params(rng)
            w, b = self.w, self.b
        else:  # used to generate test net without dropout
            self.w, self.b = params
#                    self.w, self.b = params[0] * (1 - dropout), params[1]
            w, b = self.w, self.b
            
        output = T.dot(tensor_in, w) + b
        if activation is not None:
            output = activation(output) 
        # dropout using code from https://blog.wtf.sg/2014/07/23/dropout-using-theano/
        if use_dropout:
            mask = srng.binomial(size=output.shape,p=1-dropout, 
                                 dtype=th.config.floatX)
            output = T.switch(mask, output,0)
#            output = mask * output
#            output = T.switch(srng.normal(output.shape) > dropout,output,0)
        else:
            output = (1-dropout) * output  
#                    output = output
        self.params = [self.w, self.b]
        self.output = output     
        
    def init_params(self, rng, reset=False):
        ''' 
        Function to initialise weights and biases based on the activation func
        also used to reset them in the k-fold loop.
        '''
        n_in = self.n_in 
        n_neurons = self.n_neurons 
        activation = self. activation 
        lvl = self.lvl 
        W_bound = (6 / (n_in + n_neurons))**.5  # from deeplearning.net tutorial
        inits = {T.nnet.relu : ((rng.randn(n_in, n_neurons)* 0.1).astype(th.config.floatX), 
                                np.ones(n_neurons).astype(th.config.floatX)),
                 T.nnet.sigmoid : ((rng.uniform(-W_bound, W_bound,
                                                size=(n_in, n_neurons))*4).astype(th.config.floatX), 
                                   np.zeros(n_neurons).astype(th.config.floatX)),
                 T.tanh : (rng.uniform(-W_bound, W_bound,
                                       size=(n_in, n_neurons)).astype(th.config.floatX), 
                           np.zeros(n_neurons).astype(th.config.floatX))}
        if not reset:
            w = th.shared(inits[activation][0], name='w_'+str(lvl), borrow=True) 
            b = th.shared(inits[activation][1], name='b_'+str(lvl), borrow=True)
            self.w, self.b = w, b
        else:
            self.w.set_value(inits[activation][0], borrow=True)
            self.b.set_value(inits[activation][1], borrow=True)

class cnn_pool_layer(object):
    '''
    Convolutional NN layer with pooled output.
    '''
    def __init__(self, rng, srng, tensor_in, shape_input, 
                 shape_filter=(12, 1, 5, 5), 
                 border='valid', stride=(1, 1),
                 pool_stride=(2, 2),
                 lvl=0, activation=None,
                 use_dropout=False, dropout=0, 
                 params=None):
        '''
        shape_in : (batch size (b), input channels (c), 
                    input rows (i1), input columns (i2))
        shape_filter : (output channels (c1), input channels (c2), 
                        filter rows (k1), filter columns (k2))        
        border : 'valid', 'half', 'full' or zero padding (p_1, p_2)
        stride : stride along each axis
        pool_stride : stride of pooling layer
        '''
        self.input = tensor_in
        self.shape_input = shape_input
        self.shape_filter = shape_filter
        self.border = border
        self.stride = stride
        self.pool_stride = pool_stride
        self.lvl = lvl
        self.activation = activation
        
        if params is None:
            self.init_params(rng)
#                    filters, b = self.filters, self.b
        else:  # used to generate test net without dropout
#                    filters, b = params
            self.filters, self.b = params
            
        cnn_output = T.nnet.conv2d(tensor_in, self.filters, 
                                   input_shape=shape_input, 
                                   filter_shape=shape_filter, 
                                   border_mode=border, 
                                   subsample=stride) 
        # output.shape[2] == (i1 + 2 * p1 - k1) // s1 + 1
        # output.shape[3] == (i2 + 2 * p2 - k2) // s2 + 1
        pool_output = pool.pool_2d(cnn_output,
                                   ws=pool_stride,
                                   ignore_border=True,
                                   mode='max')  # can be sum, 
#                                                         average_inc_pad, 
#                                                         average_exc_pad 

        output = pool_output + self.b.dimshuffle('x', 0, 'x', 'x')
        # from deeplearning CNN tutorial:        
            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
        
        if activation is not None:
            output = activation(output) 
        
        # dropout using code from https://blog.wtf.sg/2014/07/23/dropout-using-theano/
        if use_dropout:
            mask = srng.binomial(size=output.shape,
                                 p=1-dropout, 
                                 dtype=th.config.floatX)
            output = T.switch(mask, output, 0)
        else:
            output = (1-dropout) * output  

        self.output = output 
        self.params = [self.filters, self.b]

    def init_params(self, rng, reset=False):
        ''' 
        Function to initialise filters and biases based on the activation func
        also used to reset them in the k-fold loop.
        '''
        shape_filter = self.shape_filter 
        pool_stride = self.pool_stride
        activation = self.activation 
        lvl = self.lvl 
        # according to deeplearning cnn tutorial:
        fan_in = np.prod(shape_filter [1:])
        fan_out = (shape_filter[0] * np.prod(shape_filter[2:]) //
                   np.prod(pool_stride))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        inits = {T.nnet.relu : ((rng.standard_normal(shape_filter)* 0.1).astype(
                                                                th.config.floatX), 
                                np.ones(shape_filter[0]).astype(th.config.floatX)),
                 T.nnet.sigmoid : ((rng.uniform(-W_bound, W_bound,
                                                size=(shape_filter))*4).astype(th.config.floatX), 
                                   np.zeros(shape_filter[0]).astype(th.config.floatX)),
                 T.tanh : (rng.uniform(-W_bound, W_bound,
                                       size=(shape_filter)).astype(th.config.floatX), 
                           np.zeros(shape_filter[0]).astype(th.config.floatX))}
        if not reset:
            filters = th.shared(inits[activation][0], name='filt_'+str(lvl), borrow=True) 
            b = th.shared(inits[activation][1], name='b_'+str(lvl), borrow=True)
            self.filters, self.b = filters, b
        else:
            self.filters.set_value(inits[activation][0], borrow=True)
            self.b.set_value(inits[activation][1], borrow=True)

def construct_mlp(rng, srng, x, n_feats, batch_size, cnn_layers, mlp_layers, dropouts):
    '''
    Construct nnet based on input parameters and include dropout for training.
    '''
    # reshape x into a tensor
    n_rows, n_cols = n_feats
    next_input = x.reshape((batch_size, 1, n_rows, n_cols))
    next_input_test = next_input
    next_shape_in = (batch_size, 1, n_rows, n_cols)
    next_channels_in = 1
    mlp = []
    mlp_test = []
    
    if cnn_layers is not None:
        for layer, (n_filters, filter_size, border, pool_stride,
                    activation, dropout) in enumerate(cnn_layers):
            new_layer = cnn_pool_layer(rng, srng, 
                                       next_input, 
                                       shape_input=next_shape_in, 
                                       shape_filter=(n_filters, 
                                                     next_channels_in, 
                                                     filter_size[0], 
                                                     filter_size[1]), 
                                       border=border, 
                                       stride=(1, 1),
                                       pool_stride=pool_stride,
                                       lvl=layer, 
                                       activation=activation,
                                       dropout=dropout, 
                                       params=None)
                                       
#                                   shape_in : (batch size (b), input channels (c), 
#                                               input rows (i1), input columns (i2))
#                                   shape_filter : (output channels (c1), input channels (c2), 
#                                                   filter rows (k1), filter columns (k2))        
#                                   border : 'valid', 'half', 'full' or zero padding (p_1, p_2)
#                                   stride : stride along each axis
#                                   pool_stride : stride of pooling layer

            mlp.append(new_layer)
            
            if np.any(dropouts):
                new_layer_test = cnn_pool_layer(rng, srng,
                                       next_input_test, 
                                       shape_input=next_shape_in, 
                                       shape_filter=(n_filters, 
                                                     next_channels_in, 
                                                     filter_size[0], 
                                                     filter_size[1]), 
                                       border=border, 
                                       stride=(1, 1),
                                       pool_stride=pool_stride,
                                       lvl=layer, 
                                       activation=activation,
                                       dropout=dropout, 
                                       params=new_layer.params,
                                       use_dropout=False)
                                       
                mlp_test.append(new_layer_test)
                next_input_test = new_layer_test.output
            
            next_input = new_layer.output
            
            if border == 'valid':
                pad = [0, 0]
            elif border == 'half':
                pad = np.asarray(filter_size) // 2
                
            in_rows, in_cols = next_shape_in[2:]
#            output.shape[2] == (i1 - k1 + 2 * p1) // s1 + 1
            out_rows = (in_rows - filter_size[0] + 2*pad[0] + 1) // pool_stride[0]
            out_cols = (in_cols - filter_size[1] + 2*pad[1] + 1) // pool_stride[1]
            next_shape_in = (batch_size, n_filters, out_rows, out_cols)
            print(next_shape_in)
            next_channels_in = n_filters
            last_cnn_layer = layer + 1  # to continue counting the fc layers
        
        next_input = next_input.flatten(2)
        next_input_test = next_input_test.flatten(2)
        next_size_in = np.prod(next_shape_in[1:])
        print(next_size_in)
        
    for layer, (n_neurons, activation, dropout) in enumerate(mlp_layers):
        new_layer = fc_layer(rng, srng,
                             next_input, next_size_in, 
                             n_neurons, 
                             activation, 
                             layer+last_cnn_layer, 
                             dropout=dropout)
        mlp.append(new_layer)
        
        if np.any(dropouts):
            new_layer_test = fc_layer(rng, srng,
                                      next_input_test, 
                                      next_size_in, 
                                      n_neurons, 
                                      activation, 
                                      layer+last_cnn_layer, 
                                      dropout=dropout,
                                      params=new_layer.params, 
                                      use_dropout=False)
            mlp_test.append(new_layer_test)
            next_input_test = new_layer_test.output
        
        next_input = new_layer.output
        next_size_in = n_neurons 
    
    if not np.any(dropouts):  # all are 0s
        mlp_test = mlp
    
    return mlp, mlp_test

def construct_dnn(rng, x, n_in0, mlp_layers, dropouts=0):
    '''
    A watered down version of construct_mlp which does not have 
    convolutional layers.
    '''
    next_input = x
    next_size_in = n_in0
    mlp = []
    
    if not np.any(dropouts):  # all are 0s
        for layer, (n_neurons, activation, dropout) in enumerate(mlp_layers):
            new_layer = fc_layer(rng, next_input, next_size_in, n_neurons, 
                                 activation, layer, dropout=0)
            mlp.append(new_layer)
            
            next_input = new_layer.output
            next_size_in = n_neurons 
        
        mlp_test = mlp
        
    else:
        mlp_test = []
        next_input_test = x
        for layer, (n_neurons, activation, dropout) in enumerate(mlp_layers):
            new_layer = fc_layer(rng, next_input, next_size_in, n_neurons, 
                                 activation, layer, 
                                 use_dropout=False, dropout=dropout)
            mlp.append(new_layer)
            
            new_layer_test = fc_layer(rng, next_input_test, next_size_in, n_neurons, 
                                      activation, layer, 
                                      use_dropout=False, dropout=dropout,
                                      params=new_layer.params)
            mlp_test.append(new_layer_test)
            
            next_input = new_layer.output
            next_input_test = new_layer_test.output
            next_size_in = n_neurons 
    
    return mlp, mlp_test

def share_data(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables
        adapted from http://deeplearning.net/tutorial/logreg.html#logreg
    """
    shared_x = th.shared(np.asarray(data_x, dtype=th.config.floatX),
                         borrow=borrow)
    shared_y = th.shared(np.asarray(data_y, dtype=th.config.floatX),
                         borrow=borrow)
#    return shared_x, T.cast(shared_y, 'int32')
    return shared_x, shared_y
    
    