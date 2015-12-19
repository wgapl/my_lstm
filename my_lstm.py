#! /usr/bin/env python
"""
File: my_lstm.py

Author: Thomas Wood (thomas@wgapl.com)

Description: a quick and dirty lstm layer based on description of lstm networks
at http://deeplearning.net/tutorial/lstm.html

"""

import numpy as np
from numpy import tanh
from numpy.random import random
from string import printable


def sigmoid(z):
    return 1./(1.+np.exp(-z))

def rand_mat(nrow, ncol, sigma, mu=0.0):
    return sigma*(2*np.random.random((nrow,ncol))-1.) + np.tile(mu,(nrow,ncol))

def gen_bag_hashtable():
    N = len(printable)
    table = {}
    for k in range(N):
        table[printable[k]] = k
    return table

def make_wordvector(s, table):
    N = len(printable)
    L = len(s)
    a = np.zeros((N,L))
    for k in range(L):
        a[ table[ s[k] ], k ] = 1
    return a

def make_string(x):
    s = []
    for k in range(x.shape[1]):
        s.append(printable[np.argmax(x[:,k])])
    return ''.join(s)

class RNNLayerLSTM:
    """
    There are four afferent weight matrices:

        W_i - used to update input gate
        W_c - used to update prelimiary candidate hidden state
        W_f - used to update forget gate
        W_o - used to upate output gate

    four recurrent weight matrices (U_i, U_c, U_f, U_o)

    and four bias vectors (b_i, b_c, b_f, b_o)

    along with a weight matrix for the candidate vector (V_o).

    There are also the persistent values used to step forward the lstm layer,
    the hidden state -- h_(t-1),    and
    the candidate vector -- C_(t-1)

    """
    def __init__(self, n_in, n_out, params, eps=0.001):

        self.n_input = n_in # dimension of the input vector x_t
        self.n_output = n_out
        ####---- LAYER PARAMETERS

        # W consists of four afferent weight matrices W_i, W_c, W_f, W_o
        ind_W = 4*n_in*n_out
        self.W = params[:ind_W].reshape((4*n_out, n_in))
        # U consists of four recurrent weight matrices U_i, U_c, U_f, U_o
        ind_U = ind_W + 4*n_out*n_out
        self.U = params[ind_W:ind_U].reshape((4*n_out, n_out))
        # bias consists of four biases b_i, b_c, b_f, b_o
        ind_bias = ind_U + 4*n_out
        self.bias = params[ind_U:ind_bias].reshape((4*n_out, ))
        # One more matrix just for the value of the candidate vector
        self.V_o = params[ind_bias:].reshape((n_out, n_out))

        ####---- LAYER STATES - (PERSISTENT)

        # h is the value of the hidden state of the layer
        self.h = eps*(2*random((n_in,))-1.)

        # X is the candidate value
        self.C = eps*(2*random((n_in,))-1.)

    def step(self, x):
        """
        Input Gate update rule:
        i_t = sigmoid(W_i*x_t + U_i*h_(t-1) + b_i)

        Preliminary Candidate hidden state update rule:
        Cprelim_t = tanh(W_c*x_t +U_c*h_(t-1) + b_c)

        Forget Gate update rule:
        f_t = sigmoid(W_f*x_t + U_f*h_(t-1) + b_f)

        Candidate hidden state update rule:
        C_t = i_t*Cprelim_t + f_t*C_(t-1)

        Output Gate update rule:
        o_t = sigmoid(W_o*x_t +U_o*h_(t-1) +V_o*C_t + b_o)

        Hidden state update rule:
        h_t = o_t * tanh(C_t)

        """

        # We have stacked the afferent and reccurent weight matrices to allow
        # us to easily compute the products of x and h with their respective
        # weight matrix with a single step.
        W_x = np.dot(self.W, x)#.reshape((self.W.shape[0],1))
        U_h = np.dot(self.U, self.h)

        n = self.n_output # for ease of reading and writing

        # Split the pre-calculated matrices up for easier access
        # Common practice for me when splitting up an array in this fashion
        # I will often go back through and remove unnecessary variables.

        # W_i_x = W_x[:n]
        # W_c_x = W_x[n:2*n]
        # W_f_x = W_x[2*n:3*n]
        # W_o_x = W_x[3*n:]
        #
        # U_i_h = U_h[:n]
        # U_c_h = U_h[n:2*n]
        # U_f_h = U_h[2*n:3*n]
        # U_o_h = U_h[3*n:]

        # i_t = sigmoid(W_i_x + U_i_h + self.bias[:n])
        # C_pre = tanh(W_c_x + U_c_h + self.bias[n:2*n])
        # f_t = sigmoid(W_f_x + U_f_h + self.bias[2*n:3*n])

        # self.C = i_t *  C_pre + f_t * self.C


        self.C = sigmoid(W_x[:n] + U_h[:n] + self.bias[:n]) \
        *  tanh(W_x[n:2*n] + U_h[n:2*n] + self.bias[n:2*n]) \
        + sigmoid(W_x[2*n:3*n] + U_h[2*n:3*n] + self.bias[2*n:3*n]) \
        * self.C

        # o_t = sigmoid(W_o_x + U_o_h + np.dot(self.V_o,self.C) + self.bias[3*n:])
        # self.h = o_t * tanh(self.C)
        self.h = sigmoid(W_x[3*n:] +U_h[3*n:] + \
        np.dot(self.V_o, self.C) + self.bias[3*n:]) * tanh(self.C)

        return self.h



if __name__ == "__main__":


    s = """0 a is the quick fox who jumped over the lazy brown dog's new sentence."""

    table = gen_bag_hashtable()
    v = make_wordvector(s, table)
    s0 = make_string(v)
    print s0
    x = make_wordvector(s[:-1], table) # training
    y = make_wordvector(s[1:], table) # target

    N = len(printable)
    n_params = 9*N*N+4*N

    params4 = 0.1*(2*random((n_params,))-1.)
    rnn4 = RNNLayerLSTM(N, N, params4, eps=0.01)

    params3 = 0.1*(2*random((n_params,))-1.)
    rnn3 = RNNLayerLSTM(N, N, params3, eps=0.01)

    params2 = 0.1*(2*random((n_params,))-1.)
    rnn2 = RNNLayerLSTM(N, N, params2, eps=0.01)

    params1 = 0.1*(2*random((n_params,))-1.)
    rnn1 = RNNLayerLSTM(N, N, params1, eps=0.01)

    # Initialize the value of the output sequence
    hx = np.zeros(y.shape)

    # Iterate over input sequence, each column is a sample.
    for k in range(x.shape[1]):
        y1 = rnn1.step(x[:,k])
        y2 = rnn2.step(y1)
        y3 = rnn3.step(y2)
        y4 = rnn4.step(y3)
        hx[np.argmax(y4),k] = 1


    print hx
    print make_string(hx)
    print np.linalg.norm(hx-y) #+ np.linalg.norm(params)/len(s)
