#!/usr/bin/env python

import warnings
try:
    warnings.filterwarnings('ignore')
except Exception as e:
    pass


import argparse
import h5py
import logging
import numpy as np
import signal
import sys
from sys import stdout
from time import sleep

if sys.version_info[0] < 3:
    import OSC  # install python-osc
    pyver3 = False
else:
    from pythonosc import osc_message_builder
    from pythonosc import udp_client
    pyver3 = True


list_address = ['PosBody', 'RotPelvis', 'RotHead', 'RotNeck',
                'RotSpine1', 'RotSpine2', 'RotLeftUp', 'RotRightUp',  # last indx: 7
                'RotLeftLow', 'RotRightLow', 'RotLeftThigh', 'RotRightThigh',
                'RotLeftCalf', 'RotRightCalf', 'RotLeftFoot',  # last indx: 14
                'RotRightFoot', 'RotLeftClav', 'RotRightClav']

_PARTS = ['pelvis', 'pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l',
          'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'clavicle_r']

characters = ['DNN']  # DNN
_SERVER_IP = '192.168.170.202'
_HEIGHT_CH = 100.0


# TODO(nelson): Need to get configuration from files
def signal_handler(signal, frame):
    logging.info('Bye...  ')
    sys.exit(0)


def from_nn_file(filename):
    with h5py.File(filename, 'r') as f:
        _motion = np.zeros(f['motion'].shape, dtype=np.float32)
        np.copyto(_motion, f['motion'])  # predicted
    num_frames, _ = _motion.shape
    _ROTATIONS = np.zeros((len(_PARTS), num_frames, 3), dtype=np.float32)
    for i in range(len(list_address)):
        _ROTATIONS[i, :] = _motion[:, i * 3:i * 3 + 3]

    return _ROTATIONS


def from_htr_file(filename):
    with open(filename, 'rb') as f:
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
            _ROTATIONS[i] = array_data[:, 4:7] % 360.0  # Rotations
    init_trans = np.mean(_ROTATIONS[0, 0:31], axis=0)
    _ROTATIONS[0] -= init_trans
    return _ROTATIONS


def tx_osc(rotations):
    num_frames = rotations.shape[1]
    if pyver3:
        c = udp_client.SimpleUDPClient(_SERVER_IP, args.port)
    else:
        c = OSC.OSCClient()
        c.connect((_SERVER_IP, args.port))

    for frame in range(num_frames):
        for addr in range(len(list_address)):
            if pyver3:
                oscmsg = osc_message_builder.OscMessageBuilder(address=characters[0])
                oscmsg.add_arg(list_address[addr])
                if addr == 0:
                    oscmsg.add_arg(float(rotations[addr, frame, 0] / 10.0))
                    oscmsg.add_arg(float(rotations[addr, frame, 1] / -10.0))
                    oscmsg.add_arg(float(rotations[addr, frame, 2] / 10.0 + _HEIGHT_CH))
                else:
                    oscmsg.add_arg(float(rotations[addr, frame, 0]))
                    oscmsg.add_arg(float(rotations[addr, frame, 1] * -1.))
                    oscmsg.add_arg(float(rotations[addr, frame, 2] * -1.))
                c.send(oscmsg.build())
            else:
                oscmsg = OSC.OSCMessage()
                oscmsg.setAddress(characters[0])
                oscmsg.append(list_address[addr])
                if addr == 0:
                    msg = [rotations[addr, frame, 0] / 10.0,
                           rotations[addr, frame, 1] / -10.0,
                           rotations[addr, frame, 2] / 10.0 + _HEIGHT_CH]
                else:
                    msg = [rotations[addr, frame, 0],
                           rotations[addr, frame, 1] * -1,
                           rotations[addr, frame, 2] * -1]
                oscmsg += msg
                c.send(oscmsg)
            if frame == 0:
                logging.info(oscmsg)
        sleep(0.032)
        stdout.write('Frame: {:06d}/{:06d}\r'.format(frame, num_frames))
        stdout.flush()
    return


def main():
    if args.htr is not None:
        rots = from_htr_file(args.htr)
    elif args.hdf5 is not None:
        rots = from_nn_file(args.hdf5)
    else:
        raise TypeError('Add the file name in the execution')
    tx_osc(rots)
    logging.info('Motion Finished')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='UE4-Python OSC connection')
    parser.add_argument('--hdf5', '-d', type=str,
                        help='File for motion generation in HDF5 format')
    parser.add_argument('--htr', '-t', type=str,
                        help='File for motion generation in HTR format')
    parser.add_argument('--port', '-p', type=int,
                        help='File for motion generation in HTR format', default=6060)
    args = parser.parse_args()
    main()
