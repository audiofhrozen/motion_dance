from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
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
            self.dec_lstm1 = L.StatelessLSTM(dim + out, units)
            self.dec_lstm2 = L.StatelessLSTM(units, units)
            self.dec_lstm3 = L.StatelessLSTM(units, units)
            self.out_signal = L.Linear(units, out)

        self.state = {'ec1': None, 'ec2': None, 'ec3': None,
                      'eh1': None, 'eh2': None, 'eh3': None,
                      'dc1': None, 'dc2': None, 'dc3': None,
                      'dh1': None, 'dh2': None, 'dh3': None}
        self.delta_sgn = 0
        self.ot = 0
        self.h = 0

    def __call__(self, variables):
        xp = self.xp
        in_audio, curr_step, nx_step = variables
        batchsize = in_audio.shape[0]
        sequence = in_audio.shape[1]
        self.loss = loss = 0
        state = self.state
        self.ot = xp.std(curr_step, axis=1)
        for i in range(sequence):
            h, state, y = self.forward(state, curr_step, self.audiofeat(in_audio[:, i]), True)
            loss += F.mean_squared_error(nx_step[:, i], y)
            curr_step = y
            ot = xp.std(nx_step[:, i], axis=1) * batchsize  # y
            delta_sgn = xp.sign(ot - self.ot)
            if i > 0:
                labels = xp.minimum(xp.absolute(delta_sgn + self.delta_sgn), 1)
                labels = xp.asarray(labels, dtype=xp.int32)
                loss2 = F.contrastive(h, self.h, labels, margin=3.0) / sequence  # .mean_squared_error mean_absolute_error
                if float(chainer.cuda.to_cpu(loss2.data)) > 0.:
                    loss += loss2  # F.mean_squared_error mean_absolute_error
            self.h = h
            self.ot = ot
            self.delta_sgn = delta_sgn
        self.loss = loss
        stdout.write('loss={:.04f}\r'.format(float(chainer.cuda.to_cpu(loss.data))))
        stdout.flush()
        return self.loss

    def forward(self, state, h1, h, eval=False):
        act = F.elu
        ec1, eh1 = self.enc_lstm1(state['ec1'], state['eh1'], h)
        ec2, eh2 = self.enc_lstm2(state['ec2'], state['eh2'], eh1)
        ec3, eh3 = self.enc_lstm3(state['ec3'], state['eh3'], eh2)
        _h = F.tanh(self.fc01(eh3))
        h = F.concat((h1, _h))
        dc1, dh1 = self.dec_lstm1(state['dc1'], state['dh1'], h)
        dc2, dh2 = self.dec_lstm2(state['dc2'], state['dh2'], dh1)
        dc3, dh3 = self.dec_lstm3(state['dc3'], state['dh3'], dh2)
        h = act(self.out_signal(dh3))
        new_state = dict()
        for key in state:
            new_state[key] = locals()[key]
        if eval:
            return _h, new_state, h
        return new_state, h
