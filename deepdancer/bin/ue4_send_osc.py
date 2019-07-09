#!/usr/bin/python

import warnings
try:
    warnings.filterwarnings('ignore')
except Exception as e:
    pass


import argparse
import h5py
import logging
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
import numpy as np
import OSC  # install pyOSC
# from scipy import signal
from sys import stdout
from time import sleep
from transforms3d.euler import euler2quat
from transforms3d.euler import quat2euler


list_address = ['PosBody', 'RotPelvis', 'RotHead', 'RotNeck',
                'RotSpine1', 'RotSpine2', 'RotLeftUp', 'RotRightUp',  # last indx: 7
                'RotLeftLow', 'RotRightLow', 'RotLeftThigh', 'RotRightThigh',
                'RotLeftCalf', 'RotRightCalf', 'RotLeftFoot',  # last indx: 14
                'RotRightFoot', 'RotLeftClav', 'RotRightClav']

_PARTS = ['pelvis', 'pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l',
          'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'clavicle_r']

characters = ['True']
_SERVER_IP = '192.168.170.112'
_SERVER_PORT = 6060
_HEIGHT_CH = 100.0


def tx_osc():
    FileName = args.F

    with open(FileName, 'rb') as f:
        _DATA_FILE = f.read().split('\r\n')
    num_frames = int(_DATA_FILE[5].split(' ')[1])
    _ROTATIONS = np.zeros((len(_PARTS), num_frames, 3), dtype=np.float32)

    for i in range(len(_PARTS)):
        k = _DATA_FILE.index('[{}]'.format(_PARTS[i]))
        array_data = _DATA_FILE[k + 1:k + num_frames + 1]
        array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
        if i == 0:
            _ROTATIONS[i] = array_data[:, 1:4]  # Position Translation
        else:
            _ROTATIONS[i] = array_data[:, 4:7] % 360.0  # deg2rad(array_data) # Rotations

    # with h5py.File('prueba1', 'w')as f:
    #    f.create_dataset('1', data=_ROTATIONS)

    for i in range(1, _ROTATIONS.shape[0]):
        array_data = np.radians(_ROTATIONS[i])
        for j in range(_ROTATIONS.shape[1]):
            a, b, c = array_data[j]
            quat = euler2quat(a, b, c)
            array_data[j] = quat2euler(quat)
        _ROTATIONS[i] = np.degrees(array_data)

    with h5py.File('prueba2', 'w')as f:
        f.create_dataset('1', data=_ROTATIONS)

    init_trans = np.mean(_ROTATIONS[0, 0:31], axis=0)
    _ROTATIONS[0] -= init_trans

    c = OSC.OSCClient()
    c.connect((_SERVER_IP, _SERVER_PORT))

    for frame in range(num_frames):
        for addr in range(len(list_address)):
            oscmsg = OSC.OSCMessage()
            oscmsg.setAddress(characters[0])
            oscmsg.append(list_address[addr])
            if addr == 0:
                msg = [_ROTATIONS[addr, frame, 0] / 10.0, _ROTATIONS[addr, frame, 1] / -10.0,
                       _ROTATIONS[addr, frame, 2] / 10.0 + _HEIGHT_CH]
            else:
                msg = [_ROTATIONS[addr, frame, 0], _ROTATIONS[addr, frame, 1] * -1,
                       _ROTATIONS[addr, frame, 2] * -1]
            oscmsg += msg
            c.send(oscmsg)
            if frame == 0:
                logging.info(oscmsg)
        sleep(0.03)
        stdout.write('Frame: {:06d}/{:06d}\r'.format(frame, num_frames))
        stdout.flush()

    logging.info('Motion Finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UE4-Python OSC connection')
    parser.add_argument('F', metavar='file', type=str,
                        help='File for motion generation in HTR format')
    args = parser.parse_args()

    tx_osc()
