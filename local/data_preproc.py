#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, os, h5py, argparse, shutil
import numpy as np
from scipy.io import wavfile 
from motion_format import format_motion_audio, motionread, format_audio_noise, JOINTS, Configuration
from utillib.print_utils import print_info, print_warning, print_error

def calculate_minmax(fileslist):
  print_info('### MinMax file not found...')
  print_info('### Creating MinMax file...')
  if len(fileslist) < 1:
    print_error('No files were found in the folder ...')
    raise ValueError()
  for item in fileslist:
    print_info('   '+ item)
    rot_quats = motionread(item, 'htr', config['rot'] , JOINTS, True)
    rot_quats[:,0:3] /= config['scale'] # Scaled Position Translation to meters
    init_trans = np.mean(rot_quats[0:31,0:3], axis=0)
    rot_quats[:,0:3] -= init_trans
    tmp_minmax =  np.concatenate((np.amin(rot_quats, axis = 0)[:, None], np.amax(rot_quats, axis = 0)[:,None]), axis=1)# minimum
    if not 'pos_minmax' in locals():
      pos_minmax = np.zeros((tmp_minmax.shape), dtype=np.float32)
      pos_minmax[:,0] = tmp_minmax[:,1]
      pos_minmax[:,1] = tmp_minmax[:,0]
    pos_minmax[:,0]= np.amin([tmp_minmax[:,0],pos_minmax[:,0]], axis=0) # minimum
    pos_minmax[:,1]= np.amax([tmp_minmax[:,1],pos_minmax[:,1]], axis=0) # maximum
  with h5py.File(config['file_pos_minmax'], 'a') as f:
    ds = f.create_dataset('minmax', data=pos_minmax.T)
  return

def prepare_motion():
  with open(args.list) as f:
    files = f.readlines()
  files = [ x.split('\n')[0] for x in files ]
  list_htr_files= []
  align = []
  for x in files:
    _file, _align = x.split('\t')
    list_htr_files += [_file]
    align += [_align]

  if not os.path.exists(config['file_pos_minmax'] ):
    calculate_minmax(list_htr_files)
  with h5py.File(config['file_pos_minmax'], 'r') as f:
    pos_min = f['minmax'][0,:][None,:] 
    pos_max = f['minmax'][1,:][None,:] 

  print('### Preparing dataset...')

  div = pos_max-pos_min
  div[div==0] = 1
  config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0])/ div
  config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max

  audio_max = np.zeros((1, 129), dtype=np.float32)
  audio_min = np.ones((1, 129), dtype=np.float32) * -120.
  config['slope_wav'] = (config['rng_wav'][1]-config['rng_wav'][0]) / (audio_max-audio_min)
  config['intersec_wav'] = config['rng_wav'][1] - config['slope_wav'] * audio_max

  prefix = '{}/train_motion_'.format(config['out_folder'])
  try:
    for filename in glob.glob('{}*'.format(prefix)):
      os.remove(filename)
  except Exception as e:
    pass

  for j in range(len(list_htr_files)):
    for i in range(len(snr_lst)):
      audiodata, current_step = format_motion_audio(list_htr_files[j], config, snr=snr_lst[i], align=align[j])
      h5file = '{}f{:03d}.h5'.format(prefix, j*len(snr_lst)+i)
      with h5py.File(h5file, 'a') as f:
        ds = f.create_dataset('input', data=audiodata)
        ds = f.create_dataset('current', data=current_step)


def prepare_audio():
  with open(args.list) as f:
    files = f.readlines()
  files = [ x.split('\n')[0] for x in files ]
  list_htr_files= [x.split('\t')[0] for x in files ]

  audio_max = np.zeros((1, 129), dtype=np.float32)
  audio_min = np.ones((1, 129), dtype=np.float32) * -120.

  print('### Preparing dataset...')
  config['slope_wav'] = (config['rng_wav'][1]-config['rng_wav'][0]) / (audio_max-audio_min)
  config['intersec_wav'] = config['rng_wav'][1] - config['slope_wav'] * audio_max

  prefix = '{}/train_preaudio_'.format(config['out_folder'])
  try:
    for filename in glob.glob('{}*'.format(prefix)):
      os.remove(filename)
  except Exception as e:
    pass

  for j in range(len(list_htr_files)):
    audiodata = format_audio_noise(list_htr_files[j], config, snr_lst)
    h5file = '{}f{:03d}.h5'.format(prefix, j)
    with h5py.File(h5file, 'a') as f:
      for _key in audiodata:
        key = 'Clean' if _key == 'snr_None' else _key
        ds = f.create_dataset(key, data=audiodata[_key])

def main():
  if args.type == 'motion':
    prepare_motion()
  elif args.type == 'denoise':
    prepare_audio()
  else:
    print_error('Type of data has not been configured.')
    raise TypeError()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', '-e', type=str, help='Music Genre for the experiment')
  parser.add_argument('--list', '-l', type=str, help='Listfile')
  parser.add_argument('--save', '-v', type=str, help='Output Data')
  parser.add_argument('--rot', '-r', type=str, help='Type of Rotations')
  parser.add_argument('--type', '-t', type=str, help='Type of dataset preparation')
  parser.add_argument('--snr', '-i',  nargs='+', type=int, help='List of SNR')
  parser.add_argument('--silence', '-c', type=int, help='List of SNR', default=0)
  parser.add_argument('--scale', '-s', type=float, help='Position Scale', default=1)
  parser.add_argument('--wlen', '-w', type=int, help='STFT Window size', default=0)
  parser.add_argument('--hop', '-o', type=int, help='STFT Hop size', default=0)
  parser.add_argument('--fps', '-p', type=int, help='Motion file FPS', default=0)
  parser.add_argument('--freq', '-f', type=int, help='Audio frequency sampling', default=16000)
  args=parser.parse_args()

  config = Configuration(args)
  snr_lst = [None] + args.snr
  config['file_pos_minmax'] = '{}/minmax/{}_pos_minmax_{}.h5'.format(args.save, args.exp, args.rot)
  config['out_folder'] = '{}/data'.format(args.save) 
  main()
