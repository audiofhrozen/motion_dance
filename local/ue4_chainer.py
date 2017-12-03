#!/usr/bin/python

import OSC, argparse, colorama, h5py #install pyOSC
from time import sleep
import numpy as np
from sys import stdout
from transforms3d.euler import euler2quat, quat2euler
 
parser = argparse.ArgumentParser(description='UE4-Python OSC connection')
parser.add_argument('--hdf5', '-d', type=str, help='File for motion generation in HDF5 format')
parser.add_argument('--htr', '-t', type=str, help='File for motion generation in HTR format')
parser.add_argument('--minmax', '-m', type=str, help='File with the minmax in HDF5 format')
parser.add_argument('--rot', '-r', type=str, help='Type of Rotation')
args = parser.parse_args()


list_address = ['PosBody', 'RotPelvis', 'RotHead', 'RotNeck','RotSpine1','RotSpine2','RotLeftUp','RotRightUp', #last indx: 7
  'RotLeftLow', 'RotRightLow','RotLeftThigh','RotRightThigh', 'RotLeftCalf', 'RotRightCalf', 'RotLeftFoot',  #last indx: 14
  'RotRightFoot','RotLeftClav','RotRightClav']

_PARTS = ['pelvis', 'pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 
        'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'clavicle_r']

characters = ['DNN']
_SERVER_IP = '192.168.170.112'
_SERVER_PORT = 6060
_HEIGHT_CH = 100.0

def rad2deg(radians):
    degrees = (radians * 180.0) /np.pi 
    return degrees

# TODO: Need to get configuration from files

def from_nn_file(filename, slope_pos, intersec_pos):
  with h5py.File(filename, 'r') as f:
    motion = np.zeros(f['predicted'].shape, dtype=np.float32)
    np.copyto(motion, f['predicted'])

  motion = (motion - intersec_pos)/slope_pos
  axis_motion = np.zeros(( len(_PARTS),motion.shape[0],3), dtype=np.float32)
  for i in range(len(_PARTS)):
    if i ==0:
      axis_motion[i,:] = 100*motion[:,0:3] 
    else:
      for j in range(motion.shape[0]):
        axis_motion[i, j,:] = rad2deg(np.asarray([quat2euler(motion[j,i*4-1:i*4+3])]))
  return axis_motion

def from_htr_file(filename):
  with open(filename, 'rb') as f:
    _DATA_FILE = f.read().split('\r\n')
  num_frames = int(_DATA_FILE[5].split(' ')[1])
  _ROTATIONS = np.zeros((len(_PARTS),num_frames,3), dtype=np.float32)
  
  for i in range(len(_PARTS)):
    k = _DATA_FILE.index('[{}]'.format(_PARTS[i]))
    array_data = _DATA_FILE[k+1:k+num_frames+1]
    array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
    if i == 0:
      _ROTATIONS[i] = array_data[:,1:4] # Position Translation
    else:
      _ROTATIONS[i] = array_data[:,4:7] % 360.0 # Rotations
  init_trans = np.mean(_ROTATIONS[0,0:31], axis=0)
  _ROTATIONS[0] -= init_trans
  return _ROTATIONS 

def tx_osc(rotations):
  num_frames = rotations.shape[1]
  c = OSC.OSCClient()
  c.connect((_SERVER_IP, _SERVER_PORT))  

  for frame in range(num_frames):
    for addr in range(len(list_address)):
      oscmsg = OSC.OSCMessage()
      oscmsg.setAddress(characters[0])
      oscmsg.append(list_address[addr]) 
      if addr == 0:  
        msg =  [rotations[addr,frame,0]/10.0, rotations[addr,frame,1]/-10.0, rotations[addr,frame,2]/10.0+_HEIGHT_CH]
      else:
        msg =  [rotations[addr,frame,0], rotations[addr,frame,1]*-1, rotations[addr,frame,2]*-1]
      oscmsg += msg
      c.send(oscmsg)
      if frame==0:
        print(oscmsg)
    sleep(0.03)
    stdout.write('Frame: {:06d}/{:06d}\r'.format(frame, num_frames))
    stdout.flush()
  return

def main():
  if args.htr is not None:
    rots = from_htr_file(args.htr)
  elif args.hdf5 is not None:


      try:
        with h5py.File(args.minmax, 'r') as f:
          pos_max = np.zeros((1,f['minmax'].shape[1]), dtype=np.float32) # ToDo:Need to define
          pos_min = np.zeros((1,f['minmax'].shape[1]), dtype=np.float32)
          np.copyto(pos_min, f['minmax'][0,:])
          np.copyto(pos_max, f['minmax'][1,:])
      except Exception as e:
        raise e

      min_val = np.ones((pos_max.shape), dtype=np.float32)* -0.9 # ToDo:Need to define
      max_val = np.ones((pos_max.shape), dtype=np.float32)* 0.9
      div = (pos_max-pos_min)
      div[div==0] = 1
      slope_pos = (max_val-min_val)/ div
      intersec_pos = max_val - slope_pos * pos_max
      rots = from_nn_file(args.hdf5, slope_pos, intersec_pos)
  else:
    raise TypeError('Add the file name in the execution')
  tx_osc(rots)
  print(colorama.Fore.YELLOW+'Motion Finished')  

if __name__ == '__main__':
  colorama.init()
  main()

