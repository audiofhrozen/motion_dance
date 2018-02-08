from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
import numpy as np
from sys import stdout 


class Dancer(chainer.Chain):
    def __init__(self, InitOpt, audiofeat):
        units, dim, out = InitOpt
        super(Dancer, self).__init__()
        with self.init_scope():
            self.audiofeat = audiofeat(dim)
            self.enc_lstm1 = L.StatelessLSTM(dim, units)
            self.enc_lstm2 = L.StatelessLSTM(units, units)
            self.enc_lstm3 = L.StatelessLSTM(units, units)
            self.fc01 = L.Linear(units, dim)   
            self.dec_lstm1 = L.StatelessLSTM(dim+out, units)
            self.dec_lstm2 = L.StatelessLSTM(units, units)
            self.dec_lstm3 = L.StatelessLSTM(units, units)
            self.out_signal = L.Linear(units, out) 

        self.state = {'ec1': None,'ec2': None,'ec3': None, \
                'eh1': None, 'eh2': None,'eh3': None, \
                'dc1': None,'dc2': None,'dc3': None, \
                'dh1': None,'dh2': None,'dh3': None }

    def __call__(self, variables):
        in_audio, context, nx_step = variables
        self.loss = loss = 0
        state = self.state
        for i in range(in_audio.shape[1]):
            h = self.audiofeat(in_audio[:,i])
            state, y = self.forward(state, context, h)
            loss += F.mean_squared_error(nx_step[:,i], y) #F.mean_squared_error mean_absolute_error
            context = y
        self.loss = loss
        stdout.write('loss={:.04f}\r'.format(float(chainer.cuda.to_cpu(loss.data))))
        stdout.flush()
        return self.loss

    def forward(self, state, h1, h):
        act = F.elu
        #new_state = [None]*len(self.state)
        ec1, eh1 = self.enc_lstm1(state['ec1'], state['eh1'], h)
        ec2, eh2 = self.enc_lstm2(state['ec2'], state['eh2'], eh1)
        ec3, eh3 = self.enc_lstm3(state['ec3'], state['eh3'], eh2)
        h = act(self.fc01(eh3))
        #new_state[0], new_state[1] = self.enc_lstm1(state[0], state[1], h)
        #new_state[2], new_state[3] = self.enc_lstm2(state[2], state[3], new_state[1])
        #new_state[4], new_state[5] = self.enc_lstm3(state[4], state[5], new_state[3])
        #h = act(self.fc01(new_state[5])) 
        h = F.concat((h1,h))
        dc1, dh1 = self.dec_lstm1(state['dc1'], state['dh1'], h)
        dc2, dh2 = self.dec_lstm2(state['dc2'], state['dh2'], dh1)
        dc3, dh3 = self.dec_lstm3(state['dc3'], state['dh3'], dh2)
        h = act(self.out_signal(dh3))
        #new_state[6], new_state[7] = self.dec_lstm1(state[6], state[7], h)
        #new_state[8], new_state[9] = self.dec_lstm2(state[8], state[9], new_state[7])
        #new_state[10], new_state[11] = self.dec_lstm3(state[10], state[11], new_state[9])
        #h = act(self.out_signal(new_state[11]))
        new_state = dict()
        for key in state:
            new_state[key] = locals()[key]
        return new_state, h