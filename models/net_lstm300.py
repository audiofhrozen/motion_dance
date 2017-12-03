from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class Network(chainer.Chain):
    def __init__(self, InitOpt):
        """ At the begining and during the training, the methods __init__, __call__ are used
            and the variable loss and accuracy.
            Inside methods are not called, so any name can be define for internal network processing.
        """
        units = InitOpt[0]
        dim = InitOpt[1]
        dim2 = InitOpt[2]
        super(Network, self).__init__(
            conv1 = L.Convolution2D(1, 16, ksize=(33, 2)),
            cvbn1 = L.BatchNormalization(16),
            conv2 = L.Convolution2D(16, 32, ksize=(33, 2)),
            cvbn2 = L.BatchNormalization(32),
            conv3 = L.Convolution2D(32, 64, ksize=(33, 2),),
            cvbn3 = L.BatchNormalization(64),
            conv4 = L.Convolution2D(64, dim, ksize=(33, 2)),
            cvbn4  = L.BatchNormalization(dim),
            fc01 = L.Linear(dim, dim),
            enc_lstm1 = L.LSTM(dim, units),
            enc_lstm2 = L.LSTM(units, units),
            enc_lstm3 = L.LSTM(units, units),
            fc02 = L.Linear(units, dim),   
            dec_lstm1 = L.LSTM(dim+dim2, units),
            dec_lstm2 = L.LSTM(units, units),
            dec_lstm3 = L.LSTM(units, units),
            out_signal = L.Linear(units, dim2), 
        )
        self.train = False

    def clear(self):
        self.loss = None
        self.accuracy = None
        self.y = None
        return

    def __call__(self, variables):
        self.clear()
        y = self.forward(variables[0], variables[2])
        self.loss = F.mean_squared_error(y, variables[1]) #F.mean_squared_error mean_absolute_error
        self.y = y
        return self.loss

    def forward(self, h, h1, printing = False):
        act = F.elu
        if printing: print('x', h.data.shape)
        if printing: print('step', h1.data.shape)
        h = act(self.cvbn1(self.conv1(h)))
        if printing: print('conv1', h.data.shape)
        h = act(self.cvbn2(self.conv2(h)))
        if printing: print('conv2', h.data.shape)
        h = act(self.cvbn3(self.conv3(h)))
        if printing: print('conv3', h.data.shape)
        h = act(self.cvbn4(self.conv4(h)))
        h = act(self.fc01(h)) 
        if printing: print('conv4', h.data.shape)
        h = self.enc_lstm1(h)
        if printing: print('elstm1', h.data.shape)
        h = self.enc_lstm2(h)
        if printing: print('elstm2', h.data.shape)
        h = self.enc_lstm3(h)
        if printing: print('elstm3', h.data.shape)
        h = F.tanh(self.fc02(h)) 
        if printing: print('encoder', h.data.shape)
        h = F.concat((h1,h))
        if printing: print('concat', h.data.shape)
        h = self.dec_lstm1(h)
        if printing: print('dlstm1', h.data.shape)
        h = self.dec_lstm2(h)
        if printing: print('dlstm2', h.data.shape)
        h = self.dec_lstm3(h)
        if printing: print('dlstm3', h.data.shape)
        h = act(self.out_signal(h))
        if printing: print('output', h.data.shape)
        return h

    def reset_state(self):
        self.enc_lstm1.reset_state()
        self.enc_lstm2.reset_state()
        self.enc_lstm3.reset_state() 
        self.dec_lstm1.reset_state()
        self.dec_lstm2.reset_state()
        self.dec_lstm3.reset_state()
