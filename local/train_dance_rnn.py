#!/usr/bin/python -u
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
try:
    warnings.filterwarnings('ignore')
except Exception as e:
    pass

import argparse
import chainer
from chainer import iterators
from chainer.optimizer import GradientClipping
from chainer.optimizer import GradientNoise
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
import imp
import importlib
import logging
import matplotlib
import numpy as np
import os
import platform
import signal
import sys


matplotlib.use('Agg')
model = None


def convert(batch, device):
    batchsize = len(batch)
    in_audio = [batch[x][0] for x in range(batchsize)]
    in_audio = np.concatenate(in_audio, axis=0)
    context = [batch[x][1] for x in range(batchsize)]
    context = np.concatenate(context, axis=0)
    nx_step = [batch[x][2] for x in range(batchsize)]
    nx_step = np.concatenate(nx_step, axis=0)

    if device >= 0:
        in_audio = chainer.cuda.to_gpu(in_audio)
        context = chainer.cuda.to_gpu(context)
        nx_step = chainer.cuda.to_gpu(nx_step)
    return [in_audio, context, nx_step]


def signal_handler(signal, frame):
    logging.warning('Previous Finish Training... ')
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    serializers.save_hdf5('{}/trained.model'.format(args.save), model)
    logging.info('Optimization Finished')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class BPTTUpdater(training.updater.StandardUpdater):
    """docstring for BPTTUpdater"""

    def __init__(self, train_iter, optimizer, bprop_len, device, converter):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device, converter=converter)
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        train_batch = self.converter(train_iter.next(), self.device)
        optimizer.target.cleargrads()
        loss = optimizer.target(train_batch)
        chainer.report({'loss': loss}, optimizer.target)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


def main():
    global model
    logging.info('Training model: {}'.format(args.network))
    net = imp.load_source('Network', args.network)
    audionet = imp.load_source('Network', './models/audio_nets.py')
    model = net.Dancer(args.initOpt, getattr(audionet, args.encoder))
    if args.gpu >= 0:
        if chainer.cuda.available:
            chainer.cuda.get_device_from_id(args.gpu).use()
            chainer.config.cudnn_deterministic = False
            if platform == 'Windows':
                import subprocess
                win_cmd = 'wmic path win32_VideoController get Name | findstr /C:"NVIDIA"'
                names_gpu = subprocess.check_output(win_cmd, shell=True).decode("utf-8")
                gpu_name = names_gpu.split('\r')[0]
            elif platform == "Linux":
                import os
                names_gpu = os.popen('lspci | grep NVIDIA | grep controller').read().split('\n')
                try:
                    _, gpu_name = names_gpu[args.gpu].split('[')
                    gpu_name, _ = gpu_name.split(']')
                except Exception as e:
                    gpu_name = ""
            else:
                raise OSError('OS not supported')
            logging.info('GPU: {} - {}'.format(args.gpu, gpu_name))
            model.to_gpu()
        else:
            logging.warning('No GPU was found, the training will be executed in the CPU')
            args.gpu = -1

    logging.info('Minibatch-size: {}'.format(args.batch))
    logging.info('# epoch: {}'.format(args.epoch))

    DBClass = importlib.import_module('inlib.dataset_hdf5')
    try:
        trainset = getattr(DBClass, args.dataset)(args.folder, args.sequence, 'train', args.init_step)
    except Exception as e:
        logging.warning('Cannot continue with the training, Failing Loading Data... ')
        raise TypeError(e)

    try:
        testset = getattr(DBClass, args.dataset)(args.folder, args.sequence, 'test', args.init_step)
    except Exception as e:
        logging.warning('Cannot find testing files, test stage will be skipped... ')
        testset = None

    def make_optimizer(net, alpha=0.0002, beta1=0.5):
        optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, amsgrad=True)
        optimizer.setup(net)
        logging.info('Adding Gradient Clipping Hook')
        optimizer.add_hook(GradientClipping(10.), 'hook_clip')
        logging.info('Adding Gradient Noise Hook')
        optimizer.add_hook(GradientNoise(0.01), 'hook_noise')
        return optimizer

    optimizer = make_optimizer(model)

    if args.workers > 1:
        train_iter = iterators.MultiprocessIterator(trainset, batch_size=args.batch, shuffle=True, n_processes=args.workers,
                                                    n_prefetch=args.workers)
    else:
        train_iter = iterators.SerialIterator(trainset, batch_size=args.batch, shuffle=True)

    if testset is not None:
        test_iter = iterators.SerialIterator(testset, batch_size=args.batch, repeat=False, shuffle=False)

    # TODO(nelson): Change later the steps
    updater = BPTTUpdater(train_iter, optimizer, None, args.gpu, converter=convert)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.save)
    trainer.extend(extensions.dump_graph('main/loss'))
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(extensions.LogReport())
    if testset is not None:
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu, converter=convert))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.extend(extensions.observe_lr())
    trainer.extend(CommandsExtension())
    save_args(args, args.save)
    trainer.run()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    serializers.save_hdf5('{}/trained_{}.model'.format(args.save, args.epoch), model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Program based on Chainer Framework')
    parser.add_argument('--dataset', '-d', type=str,
                        help='Dataset File')
    parser.add_argument('--batch', '-b', type=int,
                        help='Minibatch size', default=50)
    parser.add_argument('--init_step', type=int,
                        help='Initialize sequence', default=0)
    parser.add_argument('--encoder', '-o',
                        type=str, help='Encoder type')
    parser.add_argument('--epoch', '-e',
                        type=int, help='Training epochs', default=1)
    parser.add_argument('--folder', '-f',
                        type=str, help='Dataset Location')
    parser.add_argument('--gpu', '-g', type=int,
                        help='GPU id to use, if -1 the operations will be executed in CPU', default=0)
    parser.add_argument('--initOpt', '-i', nargs='+', type=int,
                        help='Model initial options')
    parser.add_argument('--network', '-n', type=str,
                        help='Network description in python format')
    parser.add_argument('--frequency', '-r', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--save', '-s', type=str,
                        help='Folder of the model to save')
    parser.add_argument('--workers', '-w', type=int,
                        help='Number of worker processes', default=1)
    parser.add_argument('--sequence', '-q', type=int,
                        help='Training sequence', default=1)
    parser.add_argument('--verbose', type=int,
                        help='Logging Verbose', default=0)
    args = parser.parse_args()
    platform = platform.system()
    v = chainer.__version__
    format = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    if args.verbose < 1:
        logging.basicConfig(
            level=logging.WARN, format=format)
        logging.warning('Skip DEBUG/INFO messages')
    else:
        logging.basicConfig(
            level=logging.INFO, format=format)
    logging.info('============= Training Program based on Chainer v{} ============'.format(v))
    main()
    sys.exit(0)
