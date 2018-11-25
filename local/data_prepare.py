#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
try:
    warnings.filterwarnings('ignore')
except Exception:
    pass

import argparse
import glob
import h5py
import logging
from motion_format import Configuration
from motion_format import format_motion_audio
from motion_format import JOINTS
from motion_format import motionread
import numpy as np
import os
import platform
import six


def calculate_minmax(fileslist):
    logging.info('MinMax file not found...')
    logging.info('Creating MinMax file...')
    if len(fileslist) < 1:
        logging.error('No files were found in the folder ...')
        raise ValueError()
    for item in fileslist:
        logging.info('Reading at {}'.format(item))
        rot_quats = motionread(item, 'htr', config['rot'], JOINTS, True)
        rot_quats[:, 0:3] /= config['scale']  # Scaled Position Translation to meters
        init_trans = np.mean(rot_quats[0:31, 0:3], axis=0)
        rot_quats[:, 0:3] -= init_trans
        tmp_minmax = np.concatenate((np.amin(rot_quats, axis=0)[:, None],
                                    np.amax(rot_quats, axis=0)[:, None]), axis=1)  # minimum
        if 'pos_minmax' not in locals():
            pos_minmax = np.zeros((tmp_minmax.shape), dtype=np.float32)
            pos_minmax[:, 0] = tmp_minmax[:, 1]
            pos_minmax[:, 1] = tmp_minmax[:, 0]
        pos_minmax[:, 0] = np.amin([tmp_minmax[:, 0], pos_minmax[:, 0]], axis=0)  # minimum
        pos_minmax[:, 1] = np.amax([tmp_minmax[:, 1], pos_minmax[:, 1]], axis=0)  # maximum
    with h5py.File(config['file_pos_minmax'], 'a') as f:
        f.create_dataset('minmax', data=pos_minmax.T)
    return


def prepare_motion():
    with open(args.list) as f:
        files = f.readlines()
    files = [x.split('\n')[0] for x in files]
    list_htr_files = []
    align = []
    for x in files:
        _file, _align = x.split('\t')
        list_htr_files += [_file]
        align += [_align]

    if not os.path.exists(config['file_pos_minmax']):
        calculate_minmax(list_htr_files)
    with h5py.File(config['file_pos_minmax'], 'r') as f:
        pos_min = f['minmax'][0, :][None, :]
        pos_max = f['minmax'][1, :][None, :]

    logging.info('Preparing dataset...')

    div = pos_max - pos_min
    div[div == 0] = 1
    config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0]) / div
    config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max

    audio_max = 5.
    audio_min = -230.
    config['slope_wav'] = (config['rng_wav'][1] - config['rng_wav'][0]) / (audio_max - audio_min)
    config['intersec_wav'] = config['rng_wav'][1] - config['slope_wav'] * audio_max

    prefix = os.path.join(config['out_folder'], '{}_motion_'.format(args.set))
    try:
        for filename in glob.glob('{}*'.format(prefix)):
            os.remove(filename)
    except Exception as e:
        pass

    for j in six.moves.range(len(list_htr_files)):
        for i in six.moves.range(len(snr_lst)):
            audiodata, current_step = format_motion_audio(list_htr_files[j], config, snr=snr_lst[i], align=align[j])
            h5file = '{}f{:03d}.h5'.format(prefix, j * len(snr_lst) + i)
            with h5py.File(h5file, 'a') as f:
                f.create_dataset('input', data=audiodata)
                f.create_dataset('current', data=current_step)


def main():
    if args.type == 'motion':
        prepare_motion()
    else:
        logging.error('Type of data has not been configured.')
        raise TypeError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str,
                        help='Music Genre for the experiment')
    parser.add_argument('--list', '-l', type=str,
                        help='Listfile')
    parser.add_argument('--save', '-v', type=str,
                        help='Output Data')
    parser.add_argument('--rot', '-r', type=str,
                        help='Type of Rotations')
    parser.add_argument('--set', type=str,
                        help='Type of dataset')
    parser.add_argument('--type', '-t', type=str,
                        help='Type of dataset preparation')
    parser.add_argument('--snr', '-i', nargs='+', type=int,
                        help='List of SNR')
    parser.add_argument('--silence', '-c', type=int,
                        help='List of SNR', default=0)
    parser.add_argument('--scale', '-s', type=float,
                        help='Position Scale', default=1)
    parser.add_argument('--wlen', '-w', type=int,
                        help='STFT Window size', default=0)
    parser.add_argument('--hop', '-o', type=int,
                        help='STFT Hop size', default=0)
    parser.add_argument('--fps', '-p', type=int,
                        help='Motion file FPS', default=0)
    parser.add_argument('--freq', '-f', type=int,
                        help='Audio frequency sampling', default=16000)
    parser.add_argument('--verbose', type=int,
                        help='Logging Verbose', default=1)
    args = parser.parse_args()
    config = Configuration(args)
    snr_lst = [None] + args.snr
    platform = platform.system()
    format = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    if args.verbose < 1:
        logging.basicConfig(
            level=logging.WARN, format=format)
        logging.warning('Skip DEBUG/INFO messages')
    else:
        logging.basicConfig(
            level=logging.INFO, format=format)

    config['file_pos_minmax'] = os.path.join(args.save, 'minmax', 'pos_minmax.h5')
    config['out_folder'] = os.path.join(args.save, 'data')
    main()
