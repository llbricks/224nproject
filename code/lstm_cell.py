#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

# What's the logger?
# logger = logging.getLogger("hw3.q3.1")
# logger.setLevel(logging.DEBUG)
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell():
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, lstm_size):
        print(lstm_size)
        self.state_size = lstm_size

    def __call__(self, inputs, state, scope=None):

        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        input_size = inputs.get_shape()[1]
        #print('Input size: ' , input_size)

        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            W_c = tf.get_variable("W_c",(input_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            U_c = tf.get_variable("U_c",(self.state_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            b_c = tf.get_variable("b_c",(self.state_size),initializer = tf.constant_initializer(0))

            W_o = tf.get_variable("W_o",(input_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            U_o = tf.get_variable("U_o",(self.state_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable("b_o",(self.state_size),initializer = tf.constant_initializer(0))

            W_i = tf.get_variable("W_i",(input_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            U_i = tf.get_variable("U_i",(self.state_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            b_i = tf.get_variable("b_i",(self.state_size),initializer = tf.constant_initializer(0))

            W_f = tf.get_variable("W_f",(input_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            U_f = tf.get_variable("U_f",(self.state_size,self.state_size), initializer = tf.contrib.layers.xavier_initializer())
            b_f = tf.get_variable("b_f",(self.state_size),initializer = tf.constant_initializer(0))

            o_t = tf.sigmoid(tf.batch_matmul(inputs,W_o) + tf.batch_matmul(state[0],U_o) + b_c)

            f_t = tf.sigmoid(tf.batch_matmul(inputs,W_f) + tf.batch_matmul(state[0],U_f) + b_f)
            i_t = tf.sigmoid(tf.batch_matmul(inputs,W_i) + tf.batch_matmul(state[0],U_i) + b_i)
            c_t_tilde = tf.tanh(tf.batch_matmul(inputs,W_c) + tf.batch_matmul(state[0],U_c) + b_c)
            c_t = state[1]*f_t + i_t*c_t_tilde

            h_t = o_t * tf.tanh(c_t)

            #o_t = tf.tanh(tf.matmul(inputs,U_o)+ r_t*tf.matmul(state,W_o) + b_o)

            new_state = [h_t,c_t]


            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        output = new_state
        return h_t, new_state
