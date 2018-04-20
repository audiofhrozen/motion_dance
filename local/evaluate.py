#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
import glob, os, h5py, imp, six, argparse, timeit, platform, colorama
import chainer
from chainer import cuda, serializers, Variable
from motion_format import format_motion_audio, calculate_rom, extract_beats, render_motion, Configuration
import numpy as np
import pandas
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, metrics)
from sklearn.cluster import KMeans
from sklearn.svm.classes import SVC
import soundfile
from time import localtime, strftime
from utillib.audio import single_spectrogram
import utillib.BTET.beat_evaluation_toolbox as be
from utillib.print_utils import print_info
from utillib.audio import add_noise, single_spectrogram
try:
  disp=os.environ['DISPLAY']
except Exception as e:
  import matplotlib as mpl
  mpl.use('Agg')
  pass
from matplotlib import pyplot as plt

def format_audio(audioname, noise, snr, freq_samp, wav_range):
  #TODO: Add these values as arguments
  audio_max = 5.
  audio_min = -120.
  slope = (config['rng_wav'][1]-config['rng_wav'][0]) / (audio_max-audio_min)
  intersec = config['rng_wav'][1] - slope * audio_max
  wavname = audioname.replace('MP3', 'WAVE')
  wavname = wavname.replace('.mp3', '.wav')
  if not os.path.exists(wavname):
    if platform == 'Windows':
      cmmd = 'ffmpeg -y -i {} -acodec pcm_s16le -ar {} -ac 1 {}'.format(audioname, freq_samp, wavname)
      subprocess.Popen(cmmd, shell=False).communicate() 
    elif platform == 'Linux':
      os.system('sox {} -c 1 -r {} {}'.format(audioname, freq_samp, wavname))
    else:
      raise TypeError('OS not supported')
  data_wav, _  = soundfile.read(wavname)
  data_wav /= np.amax(np.abs(data_wav))
  data_wav = add_noise(data_wav, noise, snr)
  idxs = np.linspace(0, freq_samp, 31, endpoint=True, dtype=np.int)
  audio_length=np.ceil(data_wav.shape[0]*args.fps/args.freq).astype(np.int)
  stft_data = np.ones((audio_length, 1, 129, 5), dtype=np.float32)*config['rng_wav'][0]
  for i in range(audio_length):
    prv = idxs[i%args.fps]+freq_samp*int(i/args.fps)
    loc = idxs[i%args.fps+1]+freq_samp*int(i/args.fps)
    stft = single_spectrogram(data_wav[prv:loc], freq_samp, args.wlen, args.hop)
    hops =stft.shape[0]
    stft_data[i,0,:,:hops] =  np.swapaxes(stft, 0,1)
  stft_data =  (stft_data * slope) + intersec
  return stft_data


def eval_rom(TIME, NOISE, SNR, TRUE, PREDICT, BEATS, FILE):
  print_info('Evaluating motion beat')
  text = '{}\t {}\t noise:{}\t snr:{}\t fscore:{:.02f}\t precission:{:.02f}\t recall:{:.02f}\t acc:{:.02f}\n'
  rom = ''
  for i in range(len(TIME)):
    fn = os.path.basename(FILE[i])
    fn = fn.split('.')[0]
    if os.path.exists(NOISE[i]):
      noise_name = os.path.basename(NOISE[i])
      noise_name = noise_name.split('.')[0]
    else:
      noise_name = NOISE[i]
    if NOISE[i] == 'Clean':
      motion_beat_frame = calculate_rom(TRUE[i][:,3:], args.alignframe)
      motion_beat_frame = extract_beats(BEATS[i]*args.fps, motion_beat_frame, args.alignframe)
      motion_beat = motion_beat_frame.astype(np.float)/float(args.fps)
      fs, p, r, a = be.fMeasure(BEATS[i] , motion_beat)
      np.savetxt('{}/evaluation/{}_{}_true_{}_{}.txt'.format(args.folder, args.stage, fn, noise_name, SNR[i]), motion_beat, fmt='%.09f')
      rom += text.format(fn, 'true', noise_name, SNR[i], fs, p, r, a)
    motion_beat_frame = calculate_rom(PREDICT[i][:,3:], args.alignframe)
    motion_beat_frame = extract_beats(BEATS[i]*args.fps, motion_beat_frame, args.alignframe)
    motion_beat = motion_beat_frame.astype(np.float)/float(args.fps)
    fs, p, r, a = be.fMeasure(BEATS[i] , motion_beat)
    np.savetxt('{}/evaluation/{}_{}_predicted_{}_{}dB.txt'.format(args.folder, args.stage, fn, noise_name, SNR[i]), motion_beat, fmt='%.09f')
    rom += text.format(fn, 'predicted', noise_name, SNR[i], fs, p, r, a)
  with open('{}/results/{}_rom.txt'.format(args.folder,args.stage), 'w') as f:
    f.write(rom)
  return


def main():
  with open(args.list) as f:
    trained_files = f.readlines()
    trained_files = [ x.split('\n')[0] for x in trained_files ]

  trained_list= []
  trained_align = []
  for x in trained_files:
    _file, _align = x.split('\t')
    trained_list += [_file]
    trained_align += [_align]

  with open(args.audio_list) as f:
    audio_list = f.readlines()
    audio_list = [ x.split('\n')[0] for x in audio_list ]
  untrain_list = []

  for i in range(len(audio_list)):
    audioname = os.path.basename(audio_list[i]).split('.')[0]
    istrained = False
    for j in trained_list:
      if audioname in j:
        istrained = True
        break
    if not istrained:
      untrain_list.append(audio_list[i])

  print_info('Evaluation training...')   
  print_info('Loading model definition from {}'.format(args.network))
  print_info('Using gpu {}'.format(args.gpu))

  net = imp.load_source('Network', args.network) 
  audionet = imp.load_source('Network', './models/audio_nets.py')
  model = net.Dancer(args.initOpt, getattr(audionet, args.encoder))
  serializers.load_hdf5(args.model, model)
  print_info('Loading pretrained model from {}'.format(args.model))
  model.to_gpu()

  minmaxfile = './exp/data/{}_{}/minmax/pos_minmax.h5'.format(args.exp, args.rot)
  with h5py.File(minmaxfile, 'r') as f:
    pos_min = f['minmax'][0:1,:]
    pos_max = f['minmax'][1:2,:] 


  config['out_folder']= '{}/evaluation'.format(args.folder)
  div = (pos_max-pos_min)
  div[div==0] = 1
  config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0] )/ div
  config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max

  if not os.path.exists(config['out_folder']):
    os.makedirs(config['out_folder'])

  print_info('Evaluating model for trained audio files')
  for j in range(len(trained_list)):
    mbfile = trained_list[j].replace('MOCAP{}HTR'.format(slash), 'Annotations{}corrected'.format(slash))
    mbfile = mbfile.replace('{}_'.format(args.exp), '')
    mbfile = mbfile.replace('.htr', '.txt')
    mbfile = mbfile.replace('test_', '')
    music_beat = np.unique(np.loadtxt(mbfile))
    for noise in list_noises:
      if noise == 'Clean':
        list_snr = [None]
      else:
        list_snr = snr_lst
      noise_mp3= noise.replace('WAVE/', '')
      noise_mp3= noise_mp3.replace('.wav', '.mp3')
      if os.path.exists(noise_mp3):
        if not os.path.exists(noise):
          print_warning('Wavefile not found in folder, converting from mp3 file.')
          if platform == 'Windows':
            cmmd = 'ffmpeg -y -i {} -acodec pcm_s16le -ar {} -ac 1 {}'.format(noise_mp3, args.freq, noise)
            subprocess.Popen(cmmd, shell=False).communicate() 
          elif platform == 'Linux':
            os.system('sox {} -c 1 -r {} {}'.format(noise_mp3, args.freq, noise))
          else:
            raise TypeError('OS not supported')
      for snr in list_snr:
        audiofile = trained_list[j].replace('MOCAP{}HTR'.format(slash), 'AUDIO{}WAVE'.format(slash))
        audiofile = audiofile.replace('{}_'.format(args.exp), '')
        audiofile = audiofile.replace('.htr', '.wav')
        audio = format_audio(audiofile, noise, snr, args.freq, config['rng_wav'])
        predicted_motion = np.zeros((audio.shape[0]-1, args.initOpt[2]), dtype=np.float32)
        feats = np.zeros((audio.shape[0]-1, args.initOpt[1]), dtype=np.float32)
        start = timeit.default_timer()
        state = model.state
        current_step=Variable(xp.asarray(np.zeros((1, args.initOpt[2]), dtype=np.float32)))
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
          for k in range(audio.shape[0]-1):
            audiofeat = model.audiofeat(Variable(xp.asarray(audio[k:k+1])))
            rnnAudio, state, out_step= model.forward(state, current_step, audiofeat, True)
            predicted_motion[k] = chainer.cuda.to_cpu(out_step.data)
            feats[k] = chainer.cuda.to_cpu(rnnAudio.data)
            current_step = out_step
        time = timeit.default_timer() - start
        predicted_motion = (predicted_motion - config['intersec_pos'])/config['slope_pos']
        predicted_motion = render_motion(predicted_motion, args.rot, scale=args.scale)
        print(predicted_motion.shape)
        # F-Score Eval
        motion_beat_frame = calculate_rom(predicted_motion[:,3:], args.alignframe)
        motion_beat_frame = extract_beats(music_beat*args.fps, motion_beat_frame, args.alignframe)
        motion_beat = motion_beat_frame.astype(np.float)/float(args.fps)
        fs, p, r, a = be.fMeasure(music_beat, motion_beat)
        print(time, fs, p, r, a)
        exit()
        rst_time += [time]
        rst_noise += [noise]
        rst_snr += [snr]
        rst_true += [true_motion]
        rst_predict += [predicted_motion]

        filename += [fileslist[j]]
        rst_feats +=[feats]

  print_info('Evaluating model for untrained audio files')
  # Here list all evalutions definitions
  eval_rom(rst_time, rst_noise, rst_snr, rst_true, rst_predict, beat_music, filename)
  return

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluation Motion')
  parser.add_argument('--alignframe', '-a', type=int, help='Frames allowed to align', default=0)
  parser.add_argument('--silence', '-c', type=int, help='List of SNR', default=0)
  parser.add_argument('--folder', '-d', type=str, help='Specify network folder')
  parser.add_argument('--exp', '-e', type=str, help='Experiment type')
  parser.add_argument('--encoder', '-E', type=str, help='Encoder type')
  parser.add_argument('--freq', '-f', type=int, help='Audio frequency sampling', default=16000)
  parser.add_argument('--gpu', '-g', type=int, help='GPU id', default=0)
  parser.add_argument('--initOpt', '-i',  nargs='+', type=int, help='Model initial options')
  parser.add_argument('--list', '-l', type=str, help='List Files to evaluate')
  parser.add_argument('--model', '-m', type=str, help='Trained Model in HDF5')
  parser.add_argument('--network', '-n', type=str, help='Network description in python format')  
  parser.add_argument('--hop', '-o', type=int, help='STFT Hop size', default=0)
  parser.add_argument('--fps', '-p', type=int, help='Motion file FPS', default=0)
  parser.add_argument('--snr', '-r',  nargs='+', type=int, help='List of SNR')
  parser.add_argument('--scale', '-s', type=float, help='Position Scale', default=1)
  parser.add_argument('--stage', '-S', type=str, help='Evaluation stage')
  parser.add_argument('--rot', '-t', type=str, help='Rotation type')
  parser.add_argument('--wlen', '-w', type=int, help='STFT Window size', default=0)
  parser.add_argument('--plot', type=int, help='Plot PCAs', default=0)
  parser.add_argument('--videos', type=int, help='Prepare video of PCAs', default=0)
  parser.add_argument('--audio_list', type=str, help='List Files used for training')
  args = parser.parse_args()
  config = Configuration(args)
  colorama.init()
  platform = platform.system()
  if platform == 'Windows':
    import subprocess
    slash='\\'
  elif platform == 'Linux':
    slash='/'
  else:
    raise OSError('OS not supported')  
  DATA_FOLDER=os.environ['DATA_EXTRACT']
  list_noises = ['Clean', 'white', \
      '{0}{1}AUDIO{1}WAVE{1}NOISE{1}claps.wav'.format(DATA_FOLDER, slash), \
      '{0}{1}AUDIO{1}WAVE{1}NOISE{1}crowd.wav'.format(DATA_FOLDER, slash)]
  audio_list=glob.glob('{0}{1}AUDIO{1}MP3{1}*.mp3'.format(DATA_FOLDER, slash)) 
  snr_lst = args.snr
  chainer.cuda.get_device_from_id(args.gpu).use()
  xp = cuda.cupy
  main()
