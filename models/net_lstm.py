from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from sys import stdout 


class Dancer(chainer.Chain):
    def __init__(self, InitOpt, encode):
        """ At the begining and during the training, the methods __init__, __call__ are used
            and the variable loss and accuracy.
            Inside methods are not called, so any name can be define for internal network processing.
        """
        units = InitOpt[0]
        dim = InitOpt[1]
        dim2 = InitOpt[2]
        super(Dancer, self).__init__()
        with self.init_scope():
            self.encode = encode(dim)
            self.enc_lstm1 = L.StatelessLSTM(dim, units)
            self.enc_lstm2 = L.StatelessLSTM(units, units)
            self.enc_lstm3 = L.StatelessLSTM(units, units)
            self.fc01 = L.Linear(units, dim)   
            self.dec_lstm1 = L.StatelessLSTM(dim+dim2, units)
            self.dec_lstm2 = L.StatelessLSTM(units, units)
            self.dec_lstm3 = L.StatelessLSTM(units, units)
            self.out_signal = L.Linear(units, dim2) 

        self.state = {'ec1': None,'ec2': None,'ec3': None, \
                'eh1': None, 'eh2': None,'eh3': None, \
                'dc1': None,'dc2': None,'dc3': None, \
                'dh1': None,'dh2': None,'dh3': None }

    def __call__(self, variables):
        context = variables[1]
        self.loss = loss = 0
        state = self.state
        for i in range(variables[0].shape[1]):
            state, y = self.forward(variables[0][:,i], context, state)
            loss += F.mean_squared_error(variables[2][:,i], y) #F.mean_squared_error mean_absolute_error
            context = y
        self.loss = loss
        stdout.write('loss={}\r'.format(chainer.cuda.to_cpu(loss.data)))
        stdout.flush()
        return self.loss

    def forward(self, h, h1, state):
        act = F.elu
        h = self.encode(h)

        ec1, eh1 = self.enc_lstm1(state['ec1'], state['eh1'], h)
        ec2, eh2 = self.enc_lstm2(state['ec2'], state['eh2'], eh1)
        ec3, eh3 = self.enc_lstm3(state['ec3'], state['eh3'], eh2)

        h = act(self.fc01(eh3)) 
        h = F.concat((h1,h))

        dc1, dh1 = self.dec_lstm1(state['dc1'], state['eh1'], h)
        dc2, dh2 = self.dec_lstm2(state['dc2'], state['eh2'], dh1)
        dc3, dh3 = self.dec_lstm3(state['dc3'], state['eh3'], dh2)

        h = act(self.out_signal(dh3))

        new_state =dict()
        for key in state:
            new_state[key] = locals()[key]
        return new_state, h

class CNNEncode(chainer.Chain):
    def __init__(self, dim):
        """ At the begining and during the training, the methods __init__, __call__ are used
            and the variable loss and accuracy.
            Inside methods are not called, so any name can be define for internal network processing.
        """
        dim = dim[0] if isinstance(dim, list) else dim
        super(CNNEncode, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, ksize=(33, 2))
            self.cvbn1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, ksize=(33, 2))
            self.cvbn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, ksize=(33, 2),)
            self.cvbn3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, dim, ksize=(33, 2))
            self.cvbn4  = L.BatchNormalization(dim)
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, h):
        act = F.elu
        h = act(self.cvbn1(self.conv1(h)))
        h = act(self.cvbn2(self.conv2(h)))
        h = act(self.cvbn3(self.conv3(h)))
        h = act(self.cvbn4(self.conv4(h)))
        return h


class RESEncode(chainer.Chain):
    def __init__(self, dim):
        dim = dim[0] if isinstance(dim, list) else dim
        super(RESEncode, self).__init__()
        with self.init_scope():
            self.rescv_11 = L.Convolution2D(1, 32, ksize=(15, 3), pad=(7,1), nobias=True)
            self.resbn_11 = L.BatchNormalization(32)
            self.rescv_12 = L.Convolution2D(32, 1, ksize=(15, 3), pad=(7,1), nobias=True)
            self.resbn_12 = L.BatchNormalization(1)
            self.conv1 = L.Convolution2D(1, 16, ksize=(33, 2))
            self.cvbn1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, ksize=(33, 2))
            self.cvbn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, ksize=(33, 2),)
            self.cvbn3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, dim, ksize=(33, 2))
            self.cvbn4  = L.BatchNormalization(dim)
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, h):
        act = F.elu
        hr = act(self.resbn_11(self.rescv_11(h))) 
        hr = act(self.resbn_12(self.rescv_12(hr)))
        h = act(hr+h)
        h = act(self.cvbn1(self.conv1(h)))
        h = act(self.cvbn2(self.conv2(h)))
        h = act(self.cvbn3(self.conv3(h)))
        h = act(self.cvbn4(self.conv4(h)))
        return h
