#!/usr/bin/python -u
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, imp, importlib, argparse
import sys, os, h5py
import numpy as np

import chainer
from chainer import serializers, cuda, Variable
from utillib.print_utils import print_info, print_warning, print_error

def main():
  print_info('Searching in {}/{} for files:'.format(args.folder, args.stage))
  list_file = glob.glob('{}/{}_preaudio*'.format(args.folder, args.stage))
  print_info('---Postprocessing audio dataset - denoise target & encode stage')

  print_info('Trained model: {}'.format(args.network))
  net = imp.load_source('Network', args.network) 
  encoder = getattr(net, args.encoder)
  model = net.Dancer(args.initOpt, encoder)
  if (args.gpu >= 0) and cuda.available:
    cuda.get_device_from_id(args.gpu).use()
    xp = cuda.cupy
    using_gpu = True
    model.to_gpu()
  else:
    xp = np
    using_gpu = False
  serializers.load_hdf5(args.pretrain, model)

  prefix = '{}/train_postaudio_'.format(args.folder)
  try:
    for filename in glob.glob('{}*'.format(prefix)):
      os.remove(filename)
  except Exception as e:
    pass

  for i in range(len(list_file)):
    with h5py.File(list_file[i], 'r') as f:
      clean_data = f['Clean']
      shape = clean_data.shape[0]
      snr_data = f['snr_45']
      with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for j in range(0,shape, args.batch):
          out = model.encode(Variable(xp.asarray(clean_data[j:j+args.batch])))
          out1 = model.encode(Variable(xp.asarray(snr_data[j:j+args.batch])))
          if using_gpu:
            out = cuda.to_cpu(out.data)
            out1 = cuda.to_cpu(out1.data)
          else:
            out = out.data 
            out1 = out1.data 
          if not 'target' in locals():
            dims = [shape]  
            dims +=  out.shape[1:]
            targetc = np.zeros(dims, dtype=np.float32)
            targetn = np.zeros(dims, dtype=np.float32)
          targetc[j:j+args.batch] = out
          targetn[j:j+args.batch] = out1
      keys = f.keys()
      nsnr = len(keys)
      for j in range(nsnr):
        outfile = '{}f{:03d}.h5'.format(prefix, i*nsnr + j)
        with h5py.File(outfile, 'a') as h:
          ds = h.create_dataset('input', data=f[keys[j]])
          ds = h.create_dataset('cleanfeats', data=targetc)
          #ds = h.create_dataset('marg_max', data=targetn)
  print_info('PostAudio done')

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Training Program based on Chainer Framework')
  parser.add_argument('--folder', '-f', type=str, help='Dataset Location')
  parser.add_argument('--batch', '-b', type=int, help='Minibatch size', default=50000)
  parser.add_argument('--encoder', '-E', type=str, help='Encoder type')
  parser.add_argument('--gpu', '-g', type=int, help='GPU id to use, if -1 will use CPU', default=0)
  parser.add_argument('--dims', '-d',  type=int, help='Model initial options')
  parser.add_argument('--network', '-n', type=str, help='Network description in python format')
  parser.add_argument('--pretrain', '-p', type=str, help='Pretrained model')
  parser.add_argument('--initOpt', '-i',  nargs='+', type=int, help='Model initial options')
  parser.add_argument('--stage', '-s', type=str, help='Stage')
  args = parser.parse_args()
  main()