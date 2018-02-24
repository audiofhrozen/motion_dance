#!/usr/bin/python -u
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import signal, timeit, json, imp, importlib, argparse
import sys, os, h5py, six, threading
from time import sleep
import numpy as np
from datetime import datetime, timedelta
from random import shuffle
from sys import stdout 
from utillib.print_utils import print_info, print_warning, print_error

import chainer
from chainer import computational_graph as Graph
from chainer import iterators, serializers, training
import chainer.optimizers as O
import chainer.links as L
import chainer.functions as F
from chainer.optimizer import GradientNoise, GradientClipping, WeightDecay
from chainer.dataset import concat_examples
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args
from chainer.training import extensions

def convert(batch, device):
  in_audio, amin, amax = batch[0]
  for i in range(1, len(batch)):
    in_audio = np.concatenate((in_audio, batch[i][0]), axis=0)
    amin = np.concatenate((amin, batch[i][1]), axis=0)
    amax = np.concatenate((amax, batch[i][2]), axis=0)

  if device>= 0:
      in_audio = chainer.cuda.to_gpu(in_audio)
      amin =  chainer.cuda.to_gpu(amin)
      amax =  chainer.cuda.to_gpu(amax)
  return in_audio, amin, amax

class Discriminator(chainer.Chain):
  def __init__(self, InitOpt, wscale=0.2):
    dim = InitOpt[0]
    super(Discriminator, self).__init__()
    with self.init_scope():
      self.l1 = L.Linear(dim, 256, initialW=wscale)
      self.l2 = L.Linear(256, 16, initialW=wscale)
      self.out = L.Linear(16, 1, initialW=wscale)

  def __call__(self, x):
    h = F.elu(self.l1(x))
    h = F.elu(self.l2(h))
    return self.out(h)

class GANUpdater(chainer.training.updater.StandardUpdater):
  def __init__(self, *args, **kwargs):
    self.gen, self.dis = kwargs.pop('models')
    super(GANUpdater, self).__init__(*args, **kwargs)

  def loss_dis(self, dis, y_fake, y_real):
    batchsize = len(y_fake)
    L1 = F.sum(F.softplus(-y_real)) / batchsize
    L2 = F.sum(F.softplus(y_fake)) / batchsize
    #L3 = F.sum(F.softplus(-y_real2)) / args.batch
    loss = L1 + L2 #+ L3
    chainer.report({'loss': loss}, dis)
    return loss

  def loss_gen(self, gen, y_fake, mse):
    batchsize = len(y_fake)
    L4 = F.sum(F.softplus(-y_fake)) / batchsize
    #L5 = F.sum(F.softplus(y_real2)) / args.batch
    loss = L4  + mse  #+ L5 + mse1 + mse2 +mse3
    chainer.report({'loss': loss}, gen)
    return loss

  def update_core(self):
    gen_optimizer = self.get_optimizer('gen')
    dis_optimizer = self.get_optimizer('dis')

    batch = self.get_iterator('main').next()
    gen, dis = self.gen, self.dis
    batchsize = len(batch)
    noisy, amin, amax = self.converter(batch, self.device)
    feats = gen(noisy)
    mse = F.mean_squared_error(amin, feats) 

    _range = amax-amin
    y_real = dis(_range)#*_range)

    _rng1 = amax-feats
    y_fake = dis(_rng1)#*_rng1)
    #clean_feat = gen(clean)
    #y_real2 = dis(clean_feat)
    #mse1 = F.mean_squared_error(target, noisy_feat)
    #mse2 = F.mean_squared_error(target, clean_feat)
    #se3 = F.mean_squared_error(noisy_feat, clean_feat)
    dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
    gen_optimizer.update(self.loss_gen, gen, y_fake, mse)

def signal_handler(signal, frame):
    print_warning('Previous Finish Training... ')
    serializers.save_hdf5('{}/trained.model'.format(args.save), gen)
    print_info('Optimization Finished')
    sys.exit(0)

def main():
  global gen
  print_info('Training model: {}'.format(args.network))
  net = imp.load_source('Network', args.network)

  gen = getattr(net,args.generator)(args.initOpt)#net.Generator(args.initOpt)
  dis = Discriminator(args.initOpt)

  if args.gpu >= 0:
    if chainer.cuda.available:
      chainer.cuda.get_device_from_id(args.gpu).use()
      names_gpu = os.popen('lspci | grep NVIDIA | grep controller').read().split('\n')
      try:
        _, gpu_name = names_gpu[args.gpu].split('[')
        gpu_name, _ = gpu_name.split(']')
      except:
        gpu_name = ''
      print_info('GPU: {} - {}'.format(args.gpu, gpu_name))
      gen.to_gpu()
      dis.to_gpu()
    else:
      print_warning('No GPU was found, the training will be executed in the CPU')
      args.gpu = -1

  print_info('Minibatch-size: {}'.format(args.batch))
  print_info('# epoch: {}'.format(args.epoch))
  
  DBClass = importlib.import_module('utillib.chainer.dataset_hdf5') # TODO: Need to move to args

  try:
    config = { 'gpu' : args.gpu , 'folder' : args.folder}
    trainset = getattr(DBClass,args.dataset)(config, 'train')
    #TrainUpdater = getattr(DBClass,'TrainUpdater')
  except Exception as e:
    print_warning('Cannot continue with the training, Failing Loading Data... ')
    raise TypeError(e)

  try:
    testset = getattr(DBClass,args.dataset)(config, 'test')
  except Exception as e:
    print_warning('Cannot find testing files, test stage will be skipped... ')
    testset = None

  def make_optimizer(net, alpha=0.0002, beta1=0.5):
    optimizer = O.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(net)
    print_info('Adding Gradient WeightDecay Hook')
    optimizer.add_hook(WeightDecay(0.0001), 'hook_dec')
    print_info('Adding Gradient Noise Hook')
    optimizer.add_hook(GradientNoise(0.01), 'hook_noise')
    return optimizer
 
  opt_dis = make_optimizer(dis)
  opt_gen = make_optimizer(gen)

  train_iter = iterators.MultiprocessIterator(trainset, batch_size=args.batch, shuffle=True, n_processes=args.workers, \
      n_prefetch=args.workers)

  if testset is not None:
    test_iter = iterators.SerialIterator(testset, batch_size=config.trbatch, repeat=False, shuffle=False)

  updater = GANUpdater(models=(gen, dis), 
                       iterator=train_iter, 
                       optimizer={'gen' : opt_gen , 'dis' : opt_dis },
                       device=args.gpu, 
                       converter=convert) 

  trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.save)

  display_interval = (args.display, 'iteration')
  trainer.extend(extensions.dump_graph('gen/loss'))
  frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
  trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
  trainer.extend(extensions.LogReport(trigger=display_interval))
  trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',]), trigger=display_interval)

  #if extensions.PlotReport.available():
  #  trainer.extend(extensions.PlotReport(['gen/loss'], 'epoch', file_name='loss_gen.png'))
  #  trainer.extend(extensions.PlotReport(['dis/loss'], 'epoch', file_name='loss_dis.png'))
  trainer.extend(extensions.ProgressBar())

  #trainer.extend(extensions.observe_lr())
  trainer.extend(CommandsExtension())
  save_args(args, args.save)
  trainer.run()

  if not os.path.exists(args.save):
    os.makedirs(args.save)

  serializers.save_hdf5('{}/generator_trained.model'.format(args.save), gen)
  print('done')



if __name__=='__main__':
  gen = None
  signal.signal(signal.SIGINT, signal_handler)  
  parser = argparse.ArgumentParser(description='Training Program based on Chainer Framework')
  parser.add_argument('--dataset', '-d', type=str, help='Dataset File')
  parser.add_argument('--display', '-y', type=int, help='Interval to display', default=10000)
  parser.add_argument('--folder', '-f', type=str, help='Dataset Location')
  parser.add_argument('--batch', '-b', type=int, help='Minibatch size', default=50000)
  parser.add_argument('--gpu', '-g', type=int, help='GPU id to use, if -1 will use CPU', default=0)
  parser.add_argument('--epoch', '-e', type=int, help='Training epochs', default=10)
  parser.add_argument('--workers', '-w', type=int, help='Number of worker processes', default=1)
  parser.add_argument('--save', '-s', type=str, help='name of the model to save')
  parser.add_argument('--initOpt', '-i',  nargs='+', type=int, help='Model initial options')
  parser.add_argument('--network', '-n', type=str, help='Network description in python format')
  parser.add_argument('--pretrain', '-p', type=str, help='Pretrained model')
  parser.add_argument('--frequency', '-r', type=int, default=-1, help='Frequency of taking a snapshot')
  parser.add_argument('--generator', '-t', type=str, help='Class Generator')
  args = parser.parse_args()
  v = chainer.__version__
  print_info('============= Training Program based on Chainer v{} ============'.format(v))
  main()
  sys.exit(0)
