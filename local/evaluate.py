#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, os, h5py, imp, six, argparse, ConfigParser, timeit
from time import localtime, strftime
import numpy as np
from python_utils.audio import STFT, single_spectrogram
import scipy.io.wavfile as wavfile
import chainer
from chainer import cuda, serializers, Variable
from python_utils.audio import *
from python_utils.print_utils import print_info
from data_preproc import format_data
from motion_config import Configurations 
from transforms3d.euler import euler2quat, quat2euler
from colorama import Fore, Style
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='Evaluation Motion')
parser.add_argument('--folder', '-f', type=str, help='Specify network folder')
parser.add_argument('--evals', '-v', type=str, help='Evaluation to execute')
parser.add_argument('--data', '-d', type=str, help='Data folder')
parser.add_argument('--exp', '-e', type=str, help='Experiment type')
parser.add_argument('--rot', '-r', type=str, help='Rotation type')
args = parser.parse_args()
config = ConfigParser.ConfigParser()

def render_motion(_motion, _config):
  motion = (_motion - _config.intersec_pos)/_config.slope_pos
  axis_motion = np.zeros(( len(_config.parts),motion.shape[0],3), dtype=np.float32)
  for i in range(len(_config.parts)):
    if i ==0:
      axis_motion[i,:] = gnrl_conf.pos_scale*motion[:,0:3] 
    else:
      for j in range(motion.shape[0]):
        if gnrl_conf.rot_type=='euler':
          axis_motion[i, j,:] = rad2deg(motion[j,i*3:i*3+3]) 
        elif gnrl_conf.rot_type=='quat':
          axis_motion[i, j,:] = rad2deg(np.asarray([quat2euler(motion[j,i*4-1:i*4+3])])) 
        else:
          print_error('Incorrect type of rotation')
          raise TypeError()
  return axis_motion

def eval_next_step(_model, fileslist):
  print_info('Evaluating next step...')

  for j in range(len(fileslist)):
    for i in range(len(gnrl_conf.snr_lst)):
      _audio, _motion, _,  _name = format_data(fileslist[j],gnrl_conf.snr_lst[i], gnrl_conf)
      _model.reset_state()    
      _output = np.zeros((_motion.shape[0]-1, _motion.shape[1]), dtype=np.float32)
      _start = timeit.default_timer()
      for k in range(_motion.shape[0]-1):
        if k == 0:
          _current_step=_motion[0:1,:]
        else:
          _current_step=_output[k-1:k,:]
        _output[k] = _model.forward(Variable(xp.asarray(_audio[k:k+1])), 
          Variable(xp.asarray(_current_step))).data.get()
      _time = timeit.default_timer() -_start
      _motion = render_motion(_motion[1:], gnrl_conf) 
      _output = render_motion(_output, gnrl_conf)

      _out_file = '{}/{}_{}_next_step.h5'.format(gnrl_conf.out_folder,_name,gnrl_conf.snr_lst[i])
      print(str(_time), ':', mean_squared_error(_motion, _output))
      if os.path.exists(_out_file):
          os.remove(_out_file)
      with h5py.File(_out_file, 'a') as f:
          ds = f.create_dataset('true', data=_motion)
          ds = f.create_dataset('predicted', data=_output)
  return


def main():
  list_htr_files = glob.glob('{}/MOCAP/HTR/test_{}_*.htr'.format(gnrl_conf.data_folder, gnrl_conf.exp))
  net_folder = args.folder
  
  print(Fore.GREEN+'Evaluation training...')   
  print(Fore.GREEN+'Loading model from {}...'.format(net_folder))

  with open(gnrl_conf.correct_file) as f:
    corrections = f.read().split('\r\n')[:-1]
  corrections = [x.split('\t') for x in corrections]
  gnrl_conf.correct_name = [x[0] for x in corrections]
  gnrl_conf.correct_frm = [int(x[1]) for x in corrections]

  net = imp.load_source('Network', '{}/network.py'.format(net_folder))         
  ls_model = glob.glob('{}/train/*.model'.format(net_folder))
  model_file = ls_model[0]
  model = net.Network(InitOpt)
  serializers.load_hdf5(model_file, model)
  model.to_gpu()
  model.train = False

  pos_max = np.zeros((1,gnrl_conf.pos_dim), dtype=np.float32)
  pos_min = np.zeros((1,gnrl_conf.pos_dim), dtype=np.float32)
  gnrl_conf.file_pos_minmax = '{}/{}_pos_minmax_{}.h5'.format(gnrl_conf.data_folder, gnrl_conf.exp, gnrl_conf.rot_type)
  with h5py.File(gnrl_conf.file_pos_minmax, 'r') as f:
    np.copyto(pos_min, f['minmax'][0,:])
    np.copyto(pos_max, f['minmax'][1,:])
  audio_max = np.zeros((1, 129), dtype=np.float32)
  audio_min = np.ones((1, 129), dtype=np.float32) * -120.

  gnrl_conf.out_folder = '{}/evaluation'.format(net_folder)
  print('### Preparing dataset...')
  min_val = np.ones((pos_max.shape), dtype=np.float32)* gnrl_conf.rng_pos[0]
  max_val = np.ones((pos_max.shape), dtype=np.float32)* gnrl_conf.rng_pos[1]    
  div = (pos_max-pos_min)
  div[div==0] = 1
  gnrl_conf.slope_pos = (max_val-min_val)/ div
  gnrl_conf.intersec_pos = max_val - gnrl_conf.slope_pos * pos_max
  gnrl_conf.slope_wav = (gnrl_conf.rng_wav[1]-gnrl_conf.rng_wav[0]) / (audio_max-audio_min)
  gnrl_conf.intersec_wav = gnrl_conf.rng_wav[1] - gnrl_conf.slope_wav * audio_max
  if not os.path.exists(gnrl_conf.out_folder):
    os.makedirs(gnrl_conf.out_folder)

  # Here list all evalutions definitions
  eval_next_step(model, list_htr_files)
  return


if __name__ == '__main__':
  gnrl_conf=Configurations(1, args.data, args.exp, args.rot)
  gnrl_conf.correct_file = '{}/MOCAP/correction.txt'.format(args.data)
  gnrl_conf.snr_lst = ['Clean', 20, 10, 0]

  cuda.get_device().use()
  xp = cuda.cupy

  config_file = '{}/train_01.cfg'.format(args.folder)
  config.read(config_file)
  seq = config.getint('data', 'sequence')
  InitOpt= config.get('network', 'init_opt')
  InitOpt= [int(x) for x in InitOpt.split(';')]
  
  main()
