#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, os, h5py, argparse, shutil
import numpy as np
from numpy import linalg as LA
from scipy.io import wavfile 
from utillib.audio import add_noise, single_spectrogram
from transforms3d.euler import euler2quat, quat2euler
from utillib.maths import angle_between, deg2rad, rad2deg

JOINTS = ['pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 
        'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'clavicle_r']

def Configuration(args):
  indexes = np.linspace(0,args.freq, num=31, dtype=np.int)
  frame_lenght = 0
  for fps in range(args.fps):
    frame_lenght = np.amax((frame_lenght, indexes[fps+1] - indexes[fps]))

  config = {'step' : 1, 'exp' : args.exp , 'rot' : args.rot ,
            'wlen' : args.wlen , 'hop' : args.hop , 'freq' : args.freq ,
            'frame_lenght' : frame_lenght, 'indexes' : indexes ,
            'rng_pos' : [-0.9, 0.9] , 'rng_wav' : [-0.9, 0.9] , 
            'silence' : args.silence, 'scale' : args.scale,
            'fps' : args.fps
          }
  return config

def format_motion_audio(filename, config, snr=None, noise='white', align=0):
  #TODO: Change configs and test differents
  wavefile = filename.replace('MOCAP/HTR', 'AUDIO/WAVE')
  wavefile = wavefile.replace('.htr', '.wav')
  wavefile = wavefile.replace('{}_'.format(config['exp']), '')
  wavefile = wavefile.replace('test_', '')
  position_data = motionread(filename, 'htr', config['rot'] , JOINTS, True)
  position_data[:,0:3] /= config['scale'] # Scaled Position Translation to meters
  init_trans = np.mean(position_data[0:31,0:3], axis=0)
  position_data[:,0:3] -= init_trans  
  position_data = position_data[int(align):,] 
  position_data = position_data * config['slope_pos'] + config['intersec_pos']
  silence_pos = np.ones((config['silence']*config['fps'], position_data.shape[1]), dtype=np.float32) * position_data[0,:]
  position_data = np.concatenate((silence_pos, position_data, silence_pos), axis=0)            
  #current_step = position_data[:-1*config.next_step]
  #next_step = position_data[config.next_step:]

  _, data_wav = wavfile.read(wavefile)
  silence_wav = np.random.rand(config['silence']*config['freq']).astype(np.float32)*(10**-5)     
  data_max = (np.amax(np.absolute(data_wav.astype(np.float32)))) 
  data_max = data_max if data_max != 0 else 1 
  data_wav = data_wav.astype(np.float32) / data_max
  data_wav = np.concatenate((silence_wav,data_wav,silence_wav))
  data_wav = add_noise(data_wav, noise, snr)

  for frame in range(position_data.shape[0]):
    index_1 = int(int(frame/config['fps']) * config['freq'] + config['indexes'][int(frame%config['fps'])])
    index_2 = int(int(frame/config['fps']) * config['freq'] + config['indexes'][int(frame%config['fps'])+1])
    _tmp = np.zeros((config['frame_lenght'],), dtype=np.float32)
    len2 = data_wav[index_1: index_2].shape[0] 
    _tmp[0:len2] = data_wav[index_1: index_2]
    stft_data = single_spectrogram(_tmp, config['freq'], config['wlen'], config['hop']) 
    stft_data =  stft_data * config['slope_wav'] + config['intersec_wav']
    if frame ==0:
      audiodata = np.zeros((position_data.shape[0], 1, stft_data.shape[1], stft_data.shape[0]), dtype = np.float32) 
    audiodata[frame, 0] = np.swapaxes(stft_data, 0,1)

  return audiodata, position_data

def format_audio_noise(filename, config, snr_list, noise='white'):
  wavefile = filename.replace('MOCAP/HTR', 'AUDIO/WAVE')
  wavefile = wavefile.replace('.htr', '.wav')
  wavefile = wavefile.replace('{}_'.format(config['exp']), '')
  wavefile = wavefile.replace('test_', '')
    
  _, data_wav = wavfile.read(wavefile)
  silence_wav = np.random.rand(config['silence']*config['freq']).astype(np.float32)*(10**-5)     
  data_max = (np.amax(np.absolute(data_wav.astype(np.float32)))) 
  data_max = data_max if data_max != 0 else 1 
  data_wav = data_wav.astype(np.float32) / data_max
  _data_wav = np.concatenate((silence_wav,data_wav,silence_wav))

  position_data = motionread(filename, 'htr', config['rot'], JOINTS)
  silence_pos = np.ones((config['silence']*config['fps'], position_data.shape[1]), dtype=np.float32)
  position_data = np.concatenate((silence_pos, position_data, silence_pos), axis=0)      
  data = dict()
  for snr in snr_list: 
    data_wav = add_noise(_data_wav, noise, snr)
    for frame in range(position_data.shape[0]):
      index_1 = int(int(frame/config['fps']) * config['freq'] + config['indexes'][int(frame%config['fps'])])
      index_2 = int(int(frame/config['fps']) * config['freq'] + config['indexes'][int(frame%config['fps'])+1])
      _tmp = np.zeros((config['frame_lenght'],), dtype=np.float32)
      len2 = data_wav[index_1: index_2].shape[0] 
      _tmp[0:len2] = data_wav[index_1: index_2]
      stft_data = single_spectrogram(_tmp, config['freq'], config['wlen'], config['hop']) 
      stft_data =  stft_data * config['slope_wav'] + config['intersec_wav']
      if frame ==0:
        audiodata = np.zeros((position_data.shape[0], 1, stft_data.shape[1], stft_data.shape[0]), dtype = np.float32) 
      audiodata[frame, 0] = np.swapaxes(stft_data, 0,1)
    data['snr_{}'.format(snr)] =  audiodata
  return data


def peak_detect(signal):
    # signal in 1D
    gradient = np.gradient(signal)
    zero_cross= np.where(np.diff(np.signbit(gradient)))[0]
    peak = []
    for i in range(0,len(zero_cross)-2):
        xss1,_ , xss2  = zero_cross[i:i+3]
        portion = signal[xss1:xss2]
        amax =  np.amax(np.abs(portion))
        idx = np.where(np.abs(portion) == amax)[0]
        peak += [(xss1+x) for x in idx]
    peak = np.sort(np.unique(np.asarray(peak)))
    return peak
       
def closezero_detect(signal):
    # signal in 1D
    gradient = np.gradient(signal)
    zero_cross= np.where(np.diff(np.signbit(gradient)))[0]
    closzero = []
    for i in range(len(zero_cross)-2):
        xss1,_ , xss2  = zero_cross[i:i+3]
        portion = signal[xss1:xss2]
        amin =  np.amin(np.abs(portion))
        idx = np.where(np.abs(portion) == amin)[0]
        closzero += [(xss1+x) for x in idx]
    return np.asarray(closzero)

def calculate_rom(rot_quats, align=0, fps=30):
  """
  The code was implemented based on the following papers:
      [1] Chieh Ho, Wei-Tze Tsai, Keng-Sheng Lin, and Homer H Chen, National Taiwan University, 
          Extraction and Alignment Evaluation of Motion Beats for Street Dance, ICASSP 2013
      [2] Wei-Ta Chu, Member, IEEE, and Shang-Yin Tsai, National Chung Cheng University, 
          Rhythm of Motion Extraction and Rhythm-Based Cross-Media Alignment for Dance Videos, 
          IEEE Transactions on Multimedia ( Volume: 14, Issue: 1, Feb. 2012 )
  
  [1]'s algorithm was modified for rotations 
  """
  
  num_frames, num_quats = rot_quats.shape
  joints = int(num_quats/4)

  # Reduce noise by setting all frames of the axis to the same value
  for j in range(num_quats):
      if (np.std(rot_quats[:,j], ddof=1)< 0.021):
          rot_quats[:,j] = np.mean(rot_quats[:,j])

  # Calculate the speed of each frame and the angle between
  speed = np.zeros((num_frames, num_quats))
  angle = np.zeros((num_frames, joints))
  sprot_norm = np.zeros((num_frames, joints))
  for i in range(1, num_frames):
      speed[i] = rot_quats[i]- rot_quats[i-1] 
      for j in range(joints):
          v1 = speed[i-1, j*4:j*4+4]
          v2 = speed[i, j*4:j*4+4]
          if not (np.sum(v1) == 0 and np.sum(v2) == 0):
              angle[i, j] = angle_between(v1, v2) 
          sprot_norm[i,j] = LA.norm(v2)
  _beats = []
  active = []
  jnts_beat = []

  # Calculate a candidate beat by matching angle's peak and a 
  # local minimum of the speed of each joint.  
  for j in range(joints):
      if np.std(angle[:,j]) > 0:
          angle[:,j] /= np.amax(np.abs(angle[:,j]))
          sprot_norm[:,j] /= np.amax(np.abs(sprot_norm[:,j]))
          peak_angle = peak_detect(angle[:,j])
          zero_vel = closezero_detect(sprot_norm[:,j])
          joint_beat = []
          init_frame = 0
          for zero in zero_vel:
              for idx in range(init_frame, init_frame+20):
                  if (zero >= peak_angle[idx] -align) or (zero <= peak_angle[idx] + align):
                      joint_beat += [zero]
                      init_frame = idx 
                      break
          _beats += [joint_beat]
          jnts_beat += joint_beat
          active += [j]
  vel_drop = np.zeros((num_frames, len(active)))
  jnts_beat = np.array(jnts_beat)

  # Calculate the speed drop on each beat
  for j in range(len(active)):
      peak_vel = peak_detect(sprot_norm[:,active[j]])
      for vdp in _beats[j]:
          vpk = np.where(peak_vel < vdp)[0]
          if len(vpk) > 0:
             vpk =  peak_vel[vpk[-1]]
             vel_drop[vdp,j] = sprot_norm[vpk,active[j]] - sprot_norm[vdp,active[j]]
  vel_drop = np.sum(vel_drop, axis=1)

  # Process Velocity drops
  min_drop = np.where(vel_drop < np.std(vel_drop))[0]
  vel_drop[min_drop] = -0.1
  drop_cross = np.where(np.diff(np.signbit(vel_drop)))[0]
  for i in range(0,len(drop_cross),2):
      xi = drop_cross[i]-align
      xj = drop_cross[i+1] + align +1
      segment = vel_drop[xi:xj]
      maxs = np.where(segment > 0)[0]
      if len(maxs)>1:
          max_id = np.where(segment == np.amax(segment))[0][0]
          #xj = np.amin(xj, num_frames).astype(np.int)
          for j in range(xi,xj):
              if not(j-xi == max_id):
                  vel_drop[j] = -0.1 
  drops = np.asarray(np.where(vel_drop > 0)[0])

  # Match the speed drop with the candidate beats. 
  candidate_beat = [] 
  for dp in drops:
      candidate = np.where(jnts_beat==dp)[0]
      if len(candidate) > 0:
          candidate_beat += [dp]
  candidate_beat = np.sort(np.unique(np.asarray(candidate_beat)))
  return candidate_beat

def extract_beats(musicbeat, motionbeat, align=0):
  beats = []
  for beat in musicbeat:
      extracted = np.where(motionbeat >= beat-align)[0]
      if len(extracted) > 0 and (extracted[0] <= beat+align):
          beats +=[motionbeat[extracted[0]]]
  return np.asarray(beats)

def motionread(filename, extension, rottype='euler', joints=None, translation=False):
  if extension == 'htr':
    with open(filename, 'rb') as f:
      datafile = f.read().split('\r\n')
      frames = int(datafile[5].split(' ')[1])
    if rottype == 'euler':
      axis = 3
    elif rottype == 'quat':
      axis = 4
    else:
      raise ValueError('The given rotation type is incorrect or is not implemented')
    dims = axis * len(joints)
    positions = np.zeros((frames, dims), dtype=np.float32)
    for j in range(len(joints)):
        k = datafile.index('[{}]'.format(joints[j]))
        array_data = datafile[k+1 : k+frames+1]
        array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
        if translation and (j==0):
            translations = array_data[:,1:4] #np.asarray([float(x) for x in datafile[k+i+1].split('\t')[1:4]])
        euler_rot = array_data[:,4:7]
        euler_rot = deg2rad(euler_rot) 
        for i in range(frames):         
          eX, eY, eZ = euler_rot[i]
          current_rotations = euler2quat(eX, eY, eZ)
          if rottype == 'euler':
            current_rotations = np.asarray(quat2euler(current_rotations))
          positions[i, j*axis:(j+1)*axis] = current_rotations
  else:
    raise NotImplementedError('The {} extension file is not implemented, Please, format the file in HTR'.format(extension))
  if translation:
    positions = np.concatenate((translations, positions), axis=1)
  return positions

def render_motion(motion, config, Translation=True, scale=1):
  num_frames, _ = motion.shape
  if Translation:
    dims = 3
  else:
    dims = 0
  axis_motion = np.zeros((num_frames, len(JOINTS)*3+dims), dtype=np.float32)
  for i in range(len(JOINTS)):
    if i ==0 and Translation:
      axis_motion[:,i*3:(i+1)*3] = scale*motion[:,0:3] 
    for j in range(num_frames):
      if config['rot']=='euler':
        axis_motion[j,i*3+dims:(i+1)*3+dims] = rad2deg(motion[j,i*3+dims:(i+1)*3+dims]) 
      elif config['rot']=='quat':
        axis_motion[j,i*3+dims:(i+1)*3+dims] = rad2deg(np.asarray([quat2euler(motion[j,i*4+dims:(i+1)*4+dims])])) 
      else:
        print_error('Incorrect type of rotation')
        raise TypeError()
  return axis_motion