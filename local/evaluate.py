#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, os, h5py, imp, six, argparse, ConfigParser, timeit
from time import localtime, strftime
import numpy as np
import chainer
from chainer import cuda, serializers, Variable
from utillib.print_utils import print_info
from utillib.maths import rad2deg, deg2rad
from motion_format import format_motion_audio, motionread, calculate_rom, extract_beats, render_motion, Configuration
from sklearn.metrics import mean_squared_error
import BTET.beat_evaluation_toolbox as be
from mir_eval.separation import bss_eval_sources

try:
  disp=os.environ['DISPLAY']
except Exception as e:
  import matplotlib as mpl
  mpl.use('Agg')
  pass
from matplotlib import pyplot as plt

def eval_error_generations(TIME, NOISE, SNR, TRUE, PREDICT):
  print_info('Evaluating next step...')
  mse_error = ''
  smame = ''

  str_mse = '{} SNR:{} mse error:{:.06f} prediction time:{:.02f}\n'
  str_smame = '{} SNR:{} smame:{:.06f}\n'

  for i in range(len(TIME)):
    mse = mean_squared_error(TRUE[i], PREDICT[i]) 
    mse_error += str_mse.format(NOISE[i], SNR[i], mse, TIME[i])

    true_render = render_motion(TRUE[i], config, scale=args.scale)
    predicted_render = render_motion(PREDICT[i], config, scale=args.scale)
    sm = np.mean(np.abs(predicted_render-true_render)/(np.abs(true_render)+np.abs(predicted_render))) * 100.0
    smame += str_smame.format(NOISE[i], SNR[i], sm)
    if os.path.exists(NOISE[i]):
      noise_name = os.path.basename(NOISE[i])
      noise_name = noise_name.split('.')[0]
    else:
      noise_name = NOISE[i]
    with h5py.File('{}/evaluation/{}_{}_{}_motion.h5'.format(args.folder, args.stage, noise_name, SNR[i]), 'w') as f:
      ds = f.create_dataset('true', data=true_render)
      ds = f.create_dataset('predicted', data=predicted_render)

  with open('{}/results/{}_mse_error.txt'.format(args.folder, args.stage), 'w+') as f:
    f.write(mse_error)
  with open('{}/results/{}_smame.txt'.format(args.folder, args.stage), 'w') as f:
    f.write(smame)
  return

def eval_rom(TIME, NOISE, SNR, TRUE, PREDICT, BEATS, FILE):
  print_info('Evaluating motion beat...')
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
      #with h5py.File('prueba2.h5', 'a') as f:
      #  ds = f.create_dataset('rot', data=TRUE[i][:,3:])
      #  ds = f.create_dataset('prot', data=PREDICT[i][:,3:])
      #exit()
      motion_beat_frame = calculate_rom(TRUE[i][:,3:], args.alignframe)

      motion_beat_frame = extract_beats(BEATS[i]*args.fps, motion_beat_frame, args.alignframe)
      #plt.vlines(motion_beat_frame, [0.2], [0.3], 'r', label='TRUE')
      motion_beat = motion_beat_frame.astype(np.float)/float(args.fps)
      fs, p, r, a = be.fMeasure(BEATS[i] , motion_beat)
      np.savetxt('{}/evaluation/{}_{}_true_{}_{}.txt'.format(args.folder, args.stage, fn, noise_name, SNR[i]), motion_beat)
      rom += text.format(fn, 'true', NOISE[i], SNR[i], fs, p, r, a)
    #print(args.alignframe)
    motion_beat_frame = calculate_rom(PREDICT[i][:,3:], args.alignframe)
    #plt.vlines(motion_beat_frame, [0.1], [0.2], 'b', label='Predict')
    #plt.vlines(BEATS[i]*args.fps, [0], [0.1], 'r', label='Music')
    motion_beat_frame = extract_beats(BEATS[i]*args.fps, motion_beat_frame, args.alignframe)
    #plt.legend()
    #plt.show()
    #exit()
    motion_beat = motion_beat_frame.astype(np.float)/float(args.fps)
    fs, p, r, a = be.fMeasure(BEATS[i] , motion_beat)
    np.savetxt('{}/evaluation/{}_{}_predicted_{}_{}.txt'.format(args.folder, args.stage, fn, noise_name, SNR[i]), motion_beat)
    rom += text.format(fn, 'predicted', NOISE[i], SNR[i], fs, p, r, a)
  with open('{}/results/{}_rom.txt'.format(args.folder,args.stage), 'w') as f:
    f.write(rom)
  return

def eval_bss(NOISE, SNR, FEATS, FILE):
  print_info('Evaluating Features parameters...')

  j = [ x for x in range(len(NOISE)) if NOISE[x] == 'Clean'][0]
  reference_source = FEATS[j].reshape(1,-1)
  text = '{}\t noise:{}\t snr:{}\t sdr:{}\t sir:{}\t sar:{}\t\n'
  bss= ''
  for i in range(len(FEATS)):
    estimated_source = FEATS[i].reshape(1,-1)
    sdr, sir, sar, _ = bss_eval_sources(reference_source, estimated_source)
    fn = os.path.basename(FILE[i])
    fn = fn.split('.')[0]
    sir = [0] if sir[0] == np.inf else sir
    bss += text.format(fn, NOISE[i], SNR[i], sdr[0], sir[0], sar[0])
  with open('{}/results/{}_bss.txt'.format(args.folder, args.stage), 'w') as f:
    f.write(bss)
  return

def main():
  with open(args.list) as f:
    files = f.readlines()
    files = [ x.split('\n')[0] for x in files ]

  fileslist= []
  align = []
  for x in files:
    _file, _align = x.split('\t')
    fileslist += [_file]
    align += [_align]

  print_info('Evaluation training...')   
  print_info('Loading model from {}...'.format(args.network))

  net = imp.load_source('Network', args.network) 
  encoder = getattr(net, args.encoder)
  model = net.Dancer(args.initOpt, encoder)
  serializers.load_hdf5(args.model, model)
  model.to_gpu()

  minmaxfile = '{}/minmax/{}_pos_minmax_{}.h5'.format(args.folder, args.exp, args.rot)
  with h5py.File(minmaxfile, 'r') as f:
    pos_min = f['minmax'][0,:][None,:] 
    pos_max = f['minmax'][1,:][None,:] 
  audio_max = np.zeros((1, 129), dtype=np.float32)
  audio_min = np.ones((1, 129), dtype=np.float32) * -120.

  config['out_folder']= '{}/evaluation'.format(args.folder)
  div = (pos_max-pos_min)
  div[div==0] = 1
  config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0] )/ div
  config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max
  config['slope_wav'] = (config['rng_wav'][1]-config['rng_wav'][0]) / (audio_max-audio_min)
  config['intersec_wav'] = config['rng_wav'][1] - config['slope_wav'] * audio_max
  if not os.path.exists(config['out_folder']):
    os.makedirs(config['out_folder'])

  rst_time = []
  rst_noise = []
  rst_snr = []
  rst_true = []
  rst_predict = []
  beat_music = []
  filename =  []
  rst_feats = []
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
        os.system('sox {} -c 1 -r 16000 {}'.format(noise_mp3, noise))
    for snr in list_snr:
      for j in range(len(fileslist)):
        audio, true_motion = format_motion_audio(fileslist[j], config, snr, noise, align[j])
        feats = None
        predicted_motion = np.zeros((true_motion.shape[0], true_motion.shape[1]), dtype=np.float32)
        start = timeit.default_timer()
        predicted_motion[0,:]=true_motion[0,:]
        state = model.state
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
          for k in range(true_motion.shape[0]-1):
            current_step=predicted_motion[k:k+1,:]
            state, _output= model.forward(Variable(xp.asarray(audio[k:k+1])), 
              Variable(xp.asarray(current_step)), state)
            predicted_motion[k+1] = chainer.cuda.to_cpu(_output.data)

            _output = model.encode(Variable(xp.asarray(audio[k:k+1])))
            if feats is None:
              feats = chainer.cuda.to_cpu(_output.data)
            else:
              _output = chainer.cuda.to_cpu(_output.data)
              feats = np.concatenate((feats, _output), axis=0)
        time = timeit.default_timer() - start
        #predicted_motion = np.concatenate((true_motion[0:1], predicted_motion), axis=0)
        predicted_motion = (predicted_motion - config['intersec_pos'])/config['slope_pos']
        true_motion = (true_motion - config['intersec_pos'])/config['slope_pos']

        rst_time += [time]
        rst_noise += [noise]
        rst_snr += [snr]
        rst_true += [true_motion]
        rst_predict += [predicted_motion]
        mbfile = fileslist[j].replace('MOCAP/HTR', 'Annotations/corrected')
        mbfile = mbfile.replace('{}_'.format(args.exp), '')
        mbfile = mbfile.replace('.htr', '.txt')
        mbfile = mbfile.replace('test_', '')
        beat_music += [np.unique(np.loadtxt(mbfile))]
        filename += [fileslist[j]]
        rst_feats +=[feats]

  # Here list all evalutions definitions
  eval_error_generations(rst_time, rst_noise, rst_snr, rst_true, rst_predict)
  eval_rom(rst_time, rst_noise, rst_snr, rst_true, rst_predict, beat_music, filename)
  eval_bss(rst_noise, rst_snr, rst_feats, filename)

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
  parser.add_argument('--evals', '-v', type=str, help='Evaluation to execute')
  parser.add_argument('--wlen', '-w', type=int, help='STFT Window size', default=0)
  args = parser.parse_args()
  config = Configuration(args)
  list_noises = ['Clean', 'white', './data/extracted/AUDIO/WAVE/NOISE/claps.wav', './data/extracted/AUDIO/WAVE/NOISE/crowd.wav']
  snr_lst = args.snr

  chainer.cuda.get_device_from_id(args.gpu).use()
  xp = cuda.cupy
  main()
