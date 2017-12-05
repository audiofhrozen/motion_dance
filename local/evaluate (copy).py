#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, os, h5py, imp, six, argparse, ConfigParser, timeit
import numpy as np
from python_utils.audio import STFT, single_spectrogram
import scipy.io.wavfile as wavfile
import chainer
from chainer import cuda, serializers, Variable
from python_utils.audio import *
from data_preproc import format_data
from motion_config import _WLEN, _FPS, _INDEXES, _PARTS, _RANG_POS, _RANG_WAV, _ADD_SILENCE, _FREQ_SAMP, _HOP 
import colorama 

parser = argparse.ArgumentParser(description='Evaluation Motion')
parser.add_argument('--folder', '-f', type=str, help='Specify network folder')
parser.add_argument('--data', '-d', type=str, help='Data folder')
parser.add_argument('--exp', '-e', type=str, help='Experiment type')
args = parser.parse_args()
config = ConfigParser.ConfigParser()

_NEXT_STEP = 1
_DATA_FOLDER = args.data
_CORRECTION_FILE = '{}/MOCAP/correction.txt'.format(_DATA_FOLDER)
_SNR_LIST = ['Clean', 20, 10]
_EXP = args.exp
frame_lenght = 0
for fps in range(_FPS):
    frame_lenght = np.amax((frame_lenght, _INDEXES[fps+1] - _INDEXES[fps])) 

def eval_next_step(_model, fileslist):
    print(colorama.Fore.YELLOW+'Evaluating next step...')
    div = (pos_max-pos_min)
    div[div==0] = 1
    slope_pos = (_RANG_POS[1]-_RANG_POS[0])/ div
    intersec_pos = _RANG_POS[1] - slope_pos * pos_max
    slope_wav = (_RANG_WAV[1]-_RANG_WAV[0]) / (audio_max-audio_min)
    intersec_wav = _RANG_WAV[1] - slope_wav * audio_max
    glvars = [_EXP, _DATA_FOLDER, corrections_name, corrections_frame, _NEXT_STEP, frame_lenght]
    for i in range(len(_SNR_LIST)):
        for j in range(len(fileslist)):
            _audio, _motion, _,  _name = format_data(fileslist[j],_SNR_LIST[i], slope_pos, intersec_pos, slope_wav, intersec_wav, glvars)
            _model.reset_state()
            _output = np.zeros((_motion.shape[0]-1, _motion.shape[1]), dtype=np.float32)
            _start = timeit.default_timer()
            for k in range(_motion.shape[0]-1):
                if (k % 150 ==0):
                    _model.reset_state()
                if k == 0:
                    _current_step=_motion[0:1,:]
                else:
                    _current_step=_output[k-1:k,:]
                _output[k] = _model.forward(Variable(xp.asarray(_audio[k:k+1])), 
                    Variable(xp.asarray(_current_step))).data.get()
            _time = timeit.default_timer() -_start
            print(colorama.Fore.WHITE+str(_time))
            _out_file = '{}/{}_{}_next_step.h5'.format(_OUTPUT_FOLDER,_name,_SNR_LIST[i])
            if os.path.exists(_out_file):
                os.remove(_out_file)
            with h5py.File(_out_file, 'a') as f:
                ds = f.create_dataset('true', data=_motion[1:])
                ds = f.create_dataset('predicted', data=_output)
    return


def main():
    global pos_max, pos_min, audio_max, audio_min, _OUTPUT_FOLDER, corrections_name, corrections_frame
    list_htr_files = glob.glob('{}/MOCAP/HTR/test_{}_*.htr'.format(_DATA_FOLDER, _EXP))
    net_folder = args.folder
    
    print(colorama.Fore.GREEN+'Evaluation training...')   
    print(colorama.Fore.GREEN+'Loading model from {}...'.format(net_folder))

    with open(_CORRECTION_FILE) as f:
        corrections = f.read().split('\r\n')[:-1]
    corrections = [x.split('\t') for x in corrections]
    corrections_name = [x[0] for x in corrections]
    corrections_frame = [int(x[1]) for x in corrections]

    net = imp.load_source('Network', '{}/network.py'.format(net_folder))         
    ls_model = glob.glob('{}/train/*.model'.format(net_folder))
    model_file = ls_model[0]
    model = net.Network(InitOpt)
    serializers.load_hdf5(model_file, model)
    model.to_gpu()
    model.train = False

    pos_max = np.zeros((1,len(_PARTS)*3), dtype=np.float32)
    pos_min = np.zeros((1,len(_PARTS)*3), dtype=np.float32)
    _FILE_POSMINMAX = '{}/{}_pos_minmax.h5'.format(_DATA_FOLDER, _EXP)
    with h5py.File(_FILE_POSMINMAX, 'r') as f:
        np.copyto(pos_min, f['minmax'][0,:])
        np.copyto(pos_max, f['minmax'][1,:])
    audio_max = np.zeros((1, 129), dtype=np.float32)
    audio_min = np.ones((1, 129), dtype=np.float32) * -120.

    _OUTPUT_FOLDER = '{}/evaluation'.format(net_folder)
    if not os.path.exists(_OUTPUT_FOLDER):
        os.makedirs(_OUTPUT_FOLDER)

    eval_next_step(model, list_htr_files)
    return


if __name__ == '__main__':
    cuda.get_device().use()
    xp = cuda.cupy
    config_file = '{}/train_01.cfg'.format(args.folder)
    config.read(config_file)
    seq = config.getint('data', 'sequence')
    InitOpt= config.get('network', 'init_opt')
    InitOpt= [int(x) for x in InitOpt.split(';')]
    main()
