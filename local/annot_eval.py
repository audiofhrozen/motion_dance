#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import os, sys, argparse, glob, platform, h5py

from madmom.features import beats
from motion_format import motionread, calculate_rom, extract_beats, JOINTS
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import signal, stats
import utillib.BTET.beat_evaluation_toolbox as be

try:
  disp=os.environ['DISPLAY']
except Exception as e:
  import matplotlib as mpl
  mpl.use('Agg')
  pass
from matplotlib import pyplot as plt

def convermp32wav(infile, outfile):
  folder = os.path.dirname(outfile)
  if not os.path.exists(folder):
    os.makedirs(folder)
  if platform == 'Windows':
    cmmd = 'ffmpeg -y -i {} -acodec pcm_s16le -ar 16000 -ac 1 {}'.format(infile, outfile)
    subprocess.Popen(cmmd, shell=False).communicate() 
  elif platform == 'Linux':
    os.system('sox {} -c 1 -r 16000 {}'.format(infile, outfile))
  else:
    raise TypeError('OS not supported')
  return

def processMarsyas(filename):
  wav_fn = filename.replace('MOCAP{}HTR'.format(slash), 'AUDIO{}WAVE'.format(slash))
  wav_fn = wav_fn.replace('{}_'.format(args.exp), '')
  wav_fn = wav_fn.replace('test_', '')
  wav_fn = wav_fn.replace('.htr', '.wav')
  if not os.path.exists(wav_fn):
    mp3_fn = wav_fn.replace('WAVE', 'MP3')
    mp3_fn = mp3_fn.replace('wav', 'mp3')
    print('Wavefile not found in folder, converting from mp3 file.')
    convermp32wav(mp3_fn, wav_fn)
  os.system('ibt -off -o "beats" {} tmp_marsyas.txt'.format(wav_fn))
  with open('tmp_marsyas.txt') as f:
    beats=f.readlines()
  beat=[float(x.split(' ')[0]) for x in beats]
  beat = np.asarray(beat)
  os.remove('tmp_marsyas.txt')
  return beat

def processmadmomRNN(proc, filename):
  wav_fn = filename.replace('MOCAP{}HTR'.format(slash), 'AUDIO{}WAVE'.format(slash))
  wav_fn = wav_fn.replace('{}_'.format(args.exp), '')
  wav_fn = wav_fn.replace('test_', '')
  wav_fn = wav_fn.replace('.htr', '.wav')
  if not os.path.exists(wav_fn):
    mp3_fn = wav_fn.replace('WAVE', 'MP3')
    mp3_fn = mp3_fn.replace('wav', 'mp3')
    print('Wavefile not found in folder, converting from mp3 file.')
    convermp32wav(mp3_fn, wav_fn)
  act = beats.RNNBeatProcessor()(wav_fn)
  bpm = proc(act)
  bpm = np.unique(bpm)
  return bpm

def plot_vals(labels, means, stds, comp, title, idx):
  ind = np.arange(len(labels))
  width = 0.25
  fig, ax = plt.subplots()
  rects = [None] *means.shape[1]
  colors = ['r', 'b', 'g']
  for i in range(means.shape[1]):
    rects[i] = ax.bar(ind + i*width, means[:,i], width, color=colors[i], yerr=[stds[:,i,0],stds[:,i,1]]) #
  rects2 = ax.bar(ind + width, means[:,1], width, color='b', yerr=[stds[:,1,0],stds[:,1,1]]) #
  ax.set_title(title)
  ax.set_xticks(ind + width*i / 2)
  ax.set_xticklabels(labels) #rotation=40
  ax.legend(rects, comp)
  fig.savefig('{}{}{}_init_bpm_results_{}.png'.format(args.output, slash, args.stage,idx))
  return

def readfromfile(filename, folder):
  filename = filename.replace('MOCAP{}HTR'.format(slash), folder) 
  filename = filename.replace('{}_'.format(args.exp), '')
  filename = filename.replace('test_', '')
  filename = filename.replace('.htr', '.txt')
  try:
    databeats = np.unique(np.loadtxt(filename))
  except Exception as e:
    databeats = None
    pass 
  return databeats

def calculate_precission(arguments):
  # This sequence takes too much time, so it's been parallelized
  # TODO: Need to improve sequence calculate_rom to translations instead of rotations 
  # and change the way to extract the beats from the motion
  idx, rotations, musicbf, musicb, mp_result, alignframe, fps = arguments
  rots = rotations[idx:]
  motion_beat_frame = calculate_rom(rots, alignframe)
  motion_beat_frame = extract_beats(musicbf, motion_beat_frame, alignframe)
  align_beat = motion_beat_frame.astype(np.float)/float(fps)
  _, precission, _, _ = be.fMeasure(musicb , align_beat)
  mp_result.append([idx, precission, align_beat])

def main():
  with open(args.list) as f:
    filelist = f.readlines()
    filelist = [x.split('\n')[0] for x in filelist]

  proc = beats.BeatTrackingProcessor(fps=100)
  music_beat = []
  marsyas_beat = []
  mad_beat = []
  for i in range(len(filelist)):
    fn = filelist[i]
    databeats = readfromfile(fn,'Annotations{}corrected'.format(slash))
    if databeats is None:
      raise ValueError('No music beat annotations found for exp {}, prepare first the beat annotations.'.format(args.exp))
    music_beat += [databeats]

    databeats = readfromfile(fn,'Annotations{}Marsyas_ibt'.format(slash))
    if databeats is None:
      databeats = processMarsyas(fn)
      mssfn = fn.replace('MOCAP{}HTR'.format(slash), 'Annotations{}Marsyas_ibt'.format(slash)) 
      mssfn = mssfn.replace('{}_'.format(args.exp), '')
      mssfn = mssfn.replace('.htr', '.txt')
      mssfolder = os.path.dirname(mssfn)
      if not os.path.exists(mssfolder):
        os.makedirs(mssfolder)
      np.savetxt(mssfn,databeats, delimiter='\n', fmt='%.09f')
    marsyas_beat += [databeats]

    databeats = readfromfile(fn,'Annotations{}madmom'.format(slash))
    if databeats is None:
      databeats = processmadmomRNN(proc, fn)
      mdmfn = fn.replace('MOCAP{}HTR'.format(slash), 'Annotations{}madmom'.format(slash)) 
      mdmfn = mdmfn.replace('{}_'.format(args.exp), '')
      mdmfn = mdmfn.replace('.htr', '.txt')
      np.savetxt(mdmfn,databeats, delimiter='\n', fmt='%.09f')
    mad_beat +=[databeats]

  evals_name = ['fMeasure']

  print('Aligning motion files with each music...')
  motion_beat = []
  align_idx = []
  for i in range(len(music_beat)):
    mp_result = manager.list([])
    rotations = motionread(filelist[i], 'htr', 'euler', JOINTS)
    music_beat_frame = np.asarray(music_beat[i]*float(args.fps), dtype=np.int)
    rotlist = ([x,rotations, music_beat_frame,  music_beat[i], mp_result, args.alignframe, args.fps] for x in range(args.motionrange))
    pool.map(calculate_precission, rotlist)
    precission =  np.asarray([x[1] for x in mp_result]) 
    max_prec = np.where(precission==np.amax(precission))[0][0]
    align_idx += [mp_result[max_prec][0]]
    motion_beat += [mp_result[max_prec][2]] 

  stepfiles = glob.glob('{}{}{}_*.txt'.format(args.steps_folder, slash, args.exp))
  if len(stepfiles) < 1:
    print('No steps annotations files found. Preparing dance steps with basic configuration')
    entropy_eval = np.ones((len(filelist), args.beats_skips,args.beats_range)) * 10.
    for i in range(len(filelist)):
      rotations = motionread(filelist[i], 'htr', 'euler', JOINTS, True)
      idx = align_idx[i]
      aligned_motion = rotations[idx:,]
      music_beat_frame = np.asarray(music_beat[i]*float(args.fps), dtype=np.int)
      for beat_start in range (args.beats_skips):
        for beat_length in range (1,args.beats_range):
          start_idx = beat_start
          basic_step = None
          entropy_test = []
          while start_idx+beat_length < music_beat[i].shape[0]:
            start = music_beat_frame[start_idx]
            stop = music_beat_frame[start_idx+beat_length]
            if basic_step is None:
              basic_step = aligned_motion[start:stop]
            else:
              eval_step = aligned_motion[start:stop]
              new_lenght = eval_step.shape[0]
              resample_step = signal.resample(basic_step, new_lenght)
              entropy_step= stats.entropy(eval_step,resample_step)*eval_step.shape[1]
              entropy_test += [entropy_step]
            start_idx+=beat_length
          entropy_test = np.asarray(entropy_test)
          entropy_test[entropy_test==np.inf]=0
          entropy_eval[i, beat_start, beat_length] = np.mean(entropy_test)
    best_beat = np.argwhere(np.amin(entropy_eval)==entropy_eval)
    print(best_beat)
    i = best_beat[0,0]
    start = music_beat_frame[best_beat[0,1]]
    stop = music_beat_frame[best_beat[0,1]+best_beat[0,2]]
    rotations = motionread(filelist[i], 'htr', 'euler', JOINTS, True)
    aligned_motion = rotations[align_idx[i]:,]
    basic_step = aligned_motion[start:stop]
    savefile = '{}{}{}_basic_step.h5'.format(args.output, slash, args.stage)
    with h5py.File(savefile, 'w') as F:
      ds = F.create_dataset('filename', data= filelist[i])
      ds = F.create_dataset('location', data=best_beat)
      ds = F.create_dataset('dance_step', data=basic_step)
  else:
    print('Loading dance steps data')

  print('Evaluating MADMOM bpm')
  R_mad = be.evaluate_db(music_beat,mad_beat,measures='fMeasure', doCI=True)
  print('Evaluating Marsyas-ibt bpm')
  R_mar = be.evaluate_db(music_beat,marsyas_beat,measures='fMeasure', doCI=True)
  print('Evaluating MOTION bpm')
  R_mot = be.evaluate_db(music_beat,motion_beat,measures='fMeasure', doCI=True)

  init_results={ 'comparison' : ['Music-Madmom', 'Music-Marsyas', 'Music-Motion'],
    'fscore' : [R_mad['scores_mean']['fMeasure'], 
                R_mar['scores_mean']['fMeasure'],
                R_mot['scores_mean']['fMeasure']]}
  df = pd.DataFrame(init_results, columns = ['comparison', 'fscore'])
  df.to_csv('{}{}{}_init_results.csv'.format(args.output, slash, args.stage), encoding='utf-8')
  results =[R_mad, R_mot, R_mar]
  evals_mean = np.zeros((len(evals_name), len(results)))
  evals_std = np.zeros((len(evals_name), len(results),2))
  for i in range(len(evals_name)):
    for j in range(len(results)):
      evals_mean[i,j] = results[j]['scores_mean'][evals_name[i]]
      evals_std[i, j, 0] = np.abs(results[j]['scores_conf'][evals_name[i]][0] - evals_mean[i,j])
      evals_std[i, j, 1] = np.abs(results[j]['scores_conf'][evals_name[i]][1] - evals_mean[i,j])
  res_label=['Madmom', 'Dancer', 'Marsyas-ibt']
  plot_vals(evals_name, evals_mean, evals_std, res_label, 'Bootstrapping 95% confidence interval w.r.t. music beat', 1)
  align_txt = [ '{}\t{}'.format(filelist[i], align_idx[i]) for i in range(len(music_beat))]
  align_txt = '\n'.join(align_txt)
  with open('{}{}{}_files_align.txt'.format(args.output, slash, args.stage), 'w+') as f:
    f.write(align_txt)
  print('\nDone')

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='BPM Evaluation')
  parser.add_argument('--list', '-l', type=str, help='File list')
  parser.add_argument('--exp', '-e', type=str, help='Experiment type')
  parser.add_argument('--output', '-o', type=str, help='Folder to save')
  parser.add_argument('--alignframe', '-a', type=int, help='Frames allowed to align', default=0)
  parser.add_argument('--motionrange', '-m', type=int, help='Range allowed between music and motion files', default=0)
  parser.add_argument('--fps', '-f', type=int, help='Motion file FPS', default=0)
  parser.add_argument('--stage', '-s', type=str, help='Train or Test')
  parser.add_argument('--workers', '-w', type=int, help='Jobs on Parallel', default=6)
  parser.add_argument('--bpm_mdm', type=int, help='Process and compare the beats vs MADMOM Lib', default=0)
  parser.add_argument('--bpm_marsyas', type=int, help='Process and compare the beats vs Marsyas', default=0)
  parser.add_argument('--steps_folder', type=str, help='Folder with annotations of dance steps')
  parser.add_argument('--basic_steps', type=int, help='Dance steps in the motion', default=1)
  parser.add_argument('--beats_range', type=int, help='Maximum value of music beats used for a dance step', default=0)
  parser.add_argument('--beats_skips', type=int, help='Maximum value of music beats skipped for a dance step', default=0)
  args = parser.parse_args()
  platform = platform.system()
  manager=mp.Manager()
  pool=mp.Pool(processes=args.workers)
  if platform == 'Windows':
    import subprocess
    slash='\\'
  elif platform == 'Linux':
    slash='/'
  else:
    raise OSError('OS not supported')
  main()

