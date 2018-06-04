from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class CNNFeat(chainer.Chain):
    def __init__(self, dim):
        """At the begining and during the training, the methods __init__, __call__ are used

           and the variable loss and accuracy.

           Inside methods are not called, so any name can be define for internal network processing.

        """

        dim = dim[0] if isinstance(dim, list) else dim
        super(CNNFeat, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 16, ksize=(33, 2))
            self.cvbn1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, ksize=(33, 2))
            self.cvbn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, ksize=(33, 2),)
            self.cvbn3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, dim, ksize=(33, 2))
            self.cvbn4 = L.BatchNormalization(dim)
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, h):
        act = F.elu
        h = act(self.cvbn1(self.conv1(h)))
        h = act(self.cvbn2(self.conv2(h)))
        h = act(self.cvbn3(self.conv3(h)))
        h = act(self.cvbn4(self.conv4(h)))
        return h


class RESFeat(chainer.Chain):
    def __init__(self, dim):
        dim = dim[0] if isinstance(dim, list) else dim
        super(RESFeat, self).__init__()
        with self.init_scope():
            self.rescv_11 = L.Convolution2D(1, 32, ksize=(15, 3), pad=(7, 1), nobias=True)
            self.resbn_11 = L.BatchNormalization(32)
            self.rescv_12 = L.Convolution2D(32, 1, ksize=(15, 3), pad=(7, 1), nobias=True)
            self.resbn_12 = L.BatchNormalization(1)
            self.conv1 = L.Convolution2D(1, 16, ksize=(33, 2))
            self.cvbn1 = L.BatchNormalization(16)
            self.conv2 = L.Convolution2D(16, 32, ksize=(33, 2))
            self.cvbn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(32, 64, ksize=(33, 2),)
            self.cvbn3 = L.BatchNormalization(64)
            self.conv4 = L.Convolution2D(64, dim, ksize=(33, 2))
            self.cvbn4 = L.BatchNormalization(dim)
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, h):
        act = F.elu
        hr = act(self.resbn_11(self.rescv_11(h)))
        hr = act(self.resbn_12(self.rescv_12(hr)))
        h = act(hr + h)
        h = act(self.cvbn1(self.conv1(h)))
        h = act(self.cvbn2(self.conv2(h)))
        h = act(self.cvbn3(self.conv3(h)))
        h = act(self.cvbn4(self.conv4(h)))
        return h
