import h5py, OSC #install pyOSC
from time import sleep
import numpy as np
import types, argparse
from sys import stdout
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy import signal
import colorama 
from transforms3d.euler import euler2quat, quat2euler

parser = argparse.ArgumentParser(description='UE4-Python OSC connection')
parser.add_argument('F', metavar='file', type=str, help='File for motion generation in HTR format')
args = parser.parse_args()

list_address = ['PosBody', 'RotPelvis', 'RotHead', 'RotNeck','RotSpine1','RotSpine2','RotLeftUp','RotRightUp', #last indx: 7
  'RotLeftLow', 'RotRightLow','RotLeftThigh','RotRightThigh', 'RotLeftCalf', 'RotRightCalf', 'RotLeftFoot',  #last indx: 14
  'RotRightFoot','RotLeftClav','RotRightClav']

_PARTS = ['pelvis', 'pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 
        'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'clavicle_r']

characters = ['True']
_SERVER_IP = '192.168.170.112'
_SERVER_PORT = 6060
_HEIGHT_CH = 100.0

def deg2rad(degrees):
    radians = (degrees * np.pi) /180.0 
    return radians

def rad2deg(radians):
    degrees = (radians * 180.0) /np.pi 
    return degrees

def tx_osc():
  FileName =  args.F

  with open(FileName, 'rb') as f:
    _DATA_FILE = f.read().split('\r\n')
  num_frames = int(_DATA_FILE[5].split(' ')[1])
  _ROTATIONS = np.zeros((len(_PARTS),num_frames,3), dtype=np.float32)
  
  for i in range(len(_PARTS)):
    k = _DATA_FILE.index('[{}]'.format(_PARTS[i]))
    array_data = _DATA_FILE[k+1:k+num_frames+1]
    array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
    if i == 6:
      _ROTATIONS[i] = array_data[:,1:4] # Position Translation
    else:
      _ROTATIONS[i] = array_data[:,4:7] % 360.0  #deg2rad(array_data) # Rotations

  with h5py.File('prueba1', 'w')as f:
    ds = f.create_dataset('1', data=_ROTATIONS)

  for i in range(1, _ROTATIONS.shape[0]):
    array_data = deg2rad(_ROTATIONS[i])
    for j in range(_ROTATIONS.shape[1]):
      a,b,c = array_data[j]
      quat = euler2quat(a,b,c)
      array_data[j] = quat2euler(quat)
    _ROTATIONS[i] = rad2deg(array_data)

  with h5py.File('prueba2', 'w')as f:
    ds = f.create_dataset('1', data=_ROTATIONS)

  init_trans = np.mean(_ROTATIONS[0,0:31], axis=0)
  _ROTATIONS[0] -= init_trans

  mpl.rcParams['legend.fontsize'] = 10
  fig =plt.figure()
  ax = fig.gca(projection='3d')
  part = 10
  ax.plot(_ROTATIONS[part,1800:1890,0],_ROTATIONS[part,1800:1890,1],_ROTATIONS[part,1800:1890,2], label='position')
  ax.legend()
  plt.show()
  color = ['b', 'r', 'g', 'y', 'k']
  idx = [0,7,8,13] #,7,8,9,10
  plt.figure()
  for x in range(len(idx)): #0,7,8,11,12 
    i = idx[x]
    b, a = signal.butter(3, 0.05)
    new_signal = signal.filtfilt(b, a, np.gradient(_ROTATIONS[i,:,0]))
    zero_crossings = np.where(np.diff(np.signbit(new_signal)))[0]

    bpm_move = np.zeros((_ROTATIONS[i,:,0].shape))
    bpm_move[zero_crossings] = 10.0
    #print(bpm_move.shape, new_signal.shape, zero_crossings.astype(np.float32)/30.0)
    plt.plot(_ROTATIONS[i,:1500,0], color[x], label=_PARTS[i])
    plt.plot(bpm_move[0:1500], color[x], label='cross')
    plt.plot(new_signal[0:1500], color[x],label='gradient')
  plt.legend()
  plt.grid()
  plt.show()
  exit()
  
  c = OSC.OSCClient()
  c.connect((_SERVER_IP, _SERVER_PORT))  

  for frame in range(num_frames):
    for addr in range(len(list_address)):
      oscmsg = OSC.OSCMessage()
      oscmsg.setAddress(characters[0])
      oscmsg.append(list_address[addr]) 
      if addr == 0:  
        msg =  [_ROTATIONS[addr,frame,0]/10.0, _ROTATIONS[addr,frame,1]/-10.0, _ROTATIONS[addr,frame,2]/10.0+_HEIGHT_CH]
      else:
        msg =  [_ROTATIONS[addr,frame,0], _ROTATIONS[addr,frame,1]*-1, _ROTATIONS[addr,frame,2]*-1]
      oscmsg += msg
      c.send(oscmsg)
      if frame==0:
        print(oscmsg)
    sleep(0.03)
    stdout.write('Frame: {:06d}/{:06d}\r'.format(frame, num_frames))
    stdout.flush()

  print(colorama.Fore.YELLOW+'\nMotion Finished')


if __name__ == '__main__':
  colorama.init()
  tx_osc()


  """
  mpl.rcParams['legend.fontsize'] = 10
  fig =plt.figure()
  ax = fig.gca(projection='3d')
  part = 10
  ax.plot(_ROTATIONS[part,1800:1890,0],_ROTATIONS[part,1800:1890,1],_ROTATIONS[part,1800:1890,2], label='position')
  ax.legend()
  plt.show()
  color = ['b', 'r', 'g', 'y', 'k']
  idx = [0,7,8,9,10] #,7,8,9,10
  plt.figure()
  for x in range(len(idx)): #0,7,8,11,12 
    i = idx[x]
    b, a = signal.butter(3, 0.05)
    new_signal = signal.filtfilt(b, a, np.gradient(_ROTATIONS[i,:,0]))
    zero_crossings = np.where(np.diff(np.signbit(new_signal)))[0]

    bpm_move = np.zeros((_ROTATIONS[i,:,0].shape))
    bpm_move[zero_crossings] = 10.0
    print(bpm_move.shape, new_signal.shape, zero_crossings.astype(np.float32)/30.0)
    plt.plot(_ROTATIONS[i,:500,0], color[x], label=_PARTS[i])
    plt.plot(bpm_move[0:500], color[x], label='cross')
    #plt.plot(new_signal[0:2000], 'g',label='gradient')
  plt.legend()
  plt.grid()
  plt.show()
  """