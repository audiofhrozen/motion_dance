#!/usr/bin/python -u
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
# h5py has been recently printing a warning message. The filtering is to avoid that message.
import signal, timeit, json, imp, importlib, argparse, colorama, platform
import sys, os
import numpy as np
from sys import stdout 
from utillib.print_utils import print_info, print_warning, print_error

import chainer
from chainer import iterators, serializers, training
from chainer.training import extensions
import chainer.optimizers as O
from chainer.optimizer import GradientNoise, GradientClipping
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
import matplotlib
matplotlib.use('Agg')

def convert(batch, device):
  in_audio, context, nx_step = batch[0]
  for i in range(1, len(batch)):
    in_audio = np.concatenate((in_audio, batch[i][0]), axis=0)
    context = np.concatenate((context, batch[i][1]), axis=0)
    nx_step = np.concatenate((nx_step, batch[i][2]), axis=0)

  if device>= 0:
    in_audio = chainer.cuda.to_gpu(in_audio)
    context =  chainer.cuda.to_gpu(context)
    nx_step = chainer.cuda.to_gpu(nx_step)
  return [in_audio, context, nx_step]

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

model = None
 
def signal_handler(signal, frame):
    print_warning('Previous Finish Training... ')
    if not os.path.exists(args.save):
      os.makedirs(args.save)
    serializers.save_hdf5('{}/trained.model'.format(args.save), model)
    print_info('Optimization Finished')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)   
 
def main():
  global model
  print_info('Training model: {}'.format(args.network))
  net = imp.load_source('Network', args.network)
  audionet = imp.load_source('Network', './models/audio_nets.py')
  model = net.Dancer(args.initOpt, getattr(audionet, args.encoder))
  if args.gpu >= 0:
    if chainer.cuda.available:
      chainer.cuda.get_device_from_id(args.gpu).use()
      chainer.config.cudnn_deterministic = False
      names_gpu = os.popen('lspci | grep NVIDIA | grep controller').read().split('\n')
      try:
        _, gpu_name = names_gpu[args.gpu].split('[')
        gpu_name, _ = gpu_name.split(']')
      except:
        gpu_name = ''
      print_info('GPU: {} - {}'.format(args.gpu, gpu_name))
      model.to_gpu()
    else:
      print_warning('No GPU was found, the training will be executed in the CPU')
      args.gpu = -1

  print_info('Minibatch-size: {}'.format(args.batch))
  print_info('# epoch: {}'.format(args.epoch))
  
  DBClass = importlib.import_module('utillib.chainer.dataset_hdf5')
  try:
    trainset = getattr(DBClass,args.dataset)(args.folder, args.sequence, 'train', args.init_step)
  except Exception as e:
    print_warning('Cannot continue with the training, Failing Loading Data... ')
    raise TypeError(e)

  try:
    testset = getattr(DBClass,args.dataset)(args.folder, args.sequence, 'test', args.init_step)
  except Exception as e:
    print_warning('Cannot find testing files, test stage will be skipped... ')
    testset = None

  def make_optimizer(net, alpha=0.0002, beta1=0.5):
    optimizer = O.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(net)
    print_info('Adding Gradient Clipping Hook')
    optimizer.add_hook(GradientClipping(10.), 'hook_clip')
    print_info('Adding Gradient Noise Hook')
    optimizer.add_hook(GradientNoise(0.01), 'hook_noise')
    return optimizer

  optimizer = make_optimizer(model)

  train_iter = iterators.MultiprocessIterator(trainset, batch_size=args.batch, shuffle=True, n_processes=args.workers, \
      n_prefetch=args.workers) 

  if testset is not None:
    test_iter = iterators.SerialIterator(testset, batch_size=args.batch, repeat=False, shuffle=False)

  updater = BPTTUpdater(train_iter, optimizer, None, args.gpu, converter=convert) # TODO: Change latter the steps
  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.save)
  trainer.extend(extensions.dump_graph('main/loss'))
  frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
  trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
  trainer.extend(extensions.LogReport())

  if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))
  trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'elapsed_time']))
  trainer.extend(extensions.ProgressBar())

  trainer.extend(extensions.observe_lr())
  #trainer.extend(CommandsExtension())
  save_args(args, args.save)
  trainer.run()

  if not os.path.exists(args.save):
    os.makedirs(args.save)

  serializers.save_hdf5('{}/trained_{}.model'.format(args.save, args.epoch), model)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Training Program based on Chainer Framework')
  parser.add_argument('--dataset', '-d', type=str, help='Dataset File')
  parser.add_argument('--batch', '-b', type=int, help='Minibatch size', default=50)
  parser.add_argument('--init_step', type=int, help='Initialize sequence', default=0)
  parser.add_argument('--encoder', '-o', type=str, help='Encoder type')
  parser.add_argument('--epoch', '-e', type=int, help='Training epochs', default=1)
  parser.add_argument('--folder', '-f', type=str, help='Dataset Location')
  parser.add_argument('--gpu', '-g', type=int, help='GPU id to use, if -1 the operations will be executed in CPU', default=0)
  parser.add_argument('--initOpt', '-i',  nargs='+', type=int, help='Model initial options')
  parser.add_argument('--network', '-n', type=str, help='Network description in python format')
  parser.add_argument('--frequency', '-r', type=int, default=-1, help='Frequency of taking a snapshot')
  parser.add_argument('--save', '-s', type=str, help='Folder of the model to save')
  parser.add_argument('--workers', '-w', type=int, help='Number of worker processes', default=1)
  parser.add_argument('--sequence', '-q', type=int, help='Training sequence', default=1)
  args = parser.parse_args()
  colorama.init()
  platform = platform.system()
  if platform == 'Windows':
    slash='\\'
  elif platform == 'Linux':
    slash='/'
  else:
    raise OSError('OS not supported')
  v = chainer.__version__
  print_info('============= Training Program based on Chainer v{} ============'.format(v))
  main()
  sys.exit(0)
