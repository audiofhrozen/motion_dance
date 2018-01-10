#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function


import chainer, argparse, imp
from chainer import serializers
from utillib.print_utils import print_info

def main():
  print_info('Training model: {}'.format(args.network))
  net = imp.load_source('Network', args.network)
  encoder = getattr(net, args.encoder)
  model = net.Dancer(args.initOpt, encoder)
  serializers.load_hdf5(args.endtoend, model)
  serializers.load_hdf5(args.denoised, model.encode)
  serializers.save_hdf5('{}/optimized.model'.format(args.save), model)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Merge Trained Models')
  parser.add_argument('--network', '-n', type=str, help='Network description in python format')
  parser.add_argument('--encoder', '-E', type=str, help='Encoder type')
  parser.add_argument('--endtoend', '-e', type=str, help='End-to-End trained Model')
  parser.add_argument('--denoised', '-d', type=str, help='Denoised trained Model')
  parser.add_argument('--save', '-s', type=str, help='Folder of the model to save')
  parser.add_argument('--initOpt', '-i',  nargs='+', type=int, help='Model initial options')
  args = parser.parse_args()
  main()