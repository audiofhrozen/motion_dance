#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, os, h5py, argparse
import numpy as np
from scipy.io import wavfile 
from python_utils.audio import *
from motion_config import _WLEN, _FPS, _INDEXES, _PARTS, _RANG_POS, _RANG_WAV, _ADD_SILENCE, _FREQ_SAMP, _HOP 
from matplotlib import pyplot as plt

def convert_to_rad(degrees):
    degrees = degrees % 360.0
    radians = (degrees/180) * np.pi 
    #cos = np.cos(radians)
    return radians

def calculate_minmax(fileslist):
    print('### MinMax file not found...')
    print('### Creating MinMax file...')
    for item in fileslist:
        print(item)
        with open(item, 'rb') as f:
            data_file = f.read().split('\r\n')
        num_frames = int(data_file[5].split(' ')[1])
        tmp_minmax = np.zeros((len(_PARTS)*3,2), dtype=np.float32)
        for i in range(len(_PARTS)):
            j = data_file.index('[{}]'.format(_PARTS[i]))
            array_data = data_file[j+1:j+num_frames+1]
            array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
            if i == 0:
                array_data = array_data[:,1:4] #Position Translation
                init_trans = np.mean(array_data[0:31], axis=0)
                array_data -= init_trans
            else:
                array_data = array_data[:,4:7] #Rotations
                array_data = convert_to_cos(array_data)
            tmp_minmax[i*3:(i+1)*3, 0] =  np.amin(array_data, axis = 0) # minimum
            tmp_minmax[i*3:(i+1)*3, 1] =  np.amax(array_data, axis = 0) # maximum
        if not 'pos_minmax' in locals():
            pos_minmax = np.zeros((len(_PARTS)*3,2), dtype=np.float32)
            pos_minmax[:,0] = tmp_minmax[:,1]
            pos_minmax[:,1] = tmp_minmax[:,0]
        pos_minmax[:,0]= np.amin([tmp_minmax[:,0],pos_minmax[:,0]], axis=0) # minimum
        pos_minmax[:,1]= np.amax([tmp_minmax[:,1],pos_minmax[:,1]], axis=0) # maximum
    with h5py.File(_FILE_POSMINMAX, 'a') as f:
        ds = f.create_dataset('minmax', data=pos_minmax.T)
    return

def format_data(filename, snr, slope_pos, intersec_pos, slope_wav, intersec_wav, global_vars=None):
    if not global_vars is None:
        _EXP, _DATA_FOLDER, corrections_name, corrections_frame, _NEXT_STEP, frame_lenght = global_vars
    else:
        global _EXP, _DATA_FOLDER, corrections_name, corrections_frame, _NEXT_STEP, frame_lenght 
    wavefile = filename.replace('MOCAP/HTR', 'WAVE/SOX')
    wavefile = wavefile.replace('.htr', '.wav')
    wavefile = wavefile.replace('{}_'.format(_EXP), '')
    wavefile = wavefile.replace('test_', '')
    correction_name = filename.replace('{}/MOCAP/HTR/'.format(_DATA_FOLDER), '')
    correction_name = correction_name.replace('.htr', '')
    correction_name = correction_name.replace('test_', '')
    correct_idx = corrections_name.index(correction_name)
    if not os.path.exists(wavefile):
        print('### Wave file not found in SOX folder... converting...')
        tmpwav = wavefile.replace('WAVE/SOX', 'WAVE')
        dirfolder = os.path.dirname(wavefile)
        if not os.path.exists(dirfolder):
            os.makedirs(dirfolder)
        os.system('sox {} -c 1 -r 16000 {}'.format(tmpwav, wavefile))
    with open(filename, 'rb') as f:
        data_file = f.read().split('\r\n')
    num_frames = int(data_file[5].split(' ')[1])
    position_data = np.zeros((num_frames, len(_PARTS)*3), dtype=np.float32)



    for idx in range(len(_PARTS)):
        j = data_file.index('[{}]'.format(_PARTS[idx]))
        array_data = data_file[j+1:j+num_frames+1]
        array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
        if idx == 0:
            array_data = array_data[:,1:4] / 1000.0 #Position Translation in m
            init_trans = np.mean(array_data[0:31], axis=0)
            array_data -= init_trans
        else:
            array_data = array_data[:,4:7] #Rotations
            array_data = convert_to_rad(array_data)
        position_data[:, idx*3:(idx+1)*3] = array_data



    position_data = position_data[corrections_frame[correct_idx]:,] 
    position_data = position_data * slope_pos + intersec_pos
    silence_pos = np.ones((_ADD_SILENCE*_FPS, position_data.shape[1]), dtype=np.float32) * position_data[0,:]
    position_data = np.concatenate((silence_pos, position_data, silence_pos), axis=0)            

    #print(position_data.shape)
    #for i in range(51):
    #    f = plt.figure()
    #    plt.plot(position_data[:,i])
    #    f.savefig('echo_{:02d}.png'.format(i), dpi=f.dpi)
    #    plt.close(f)
    #plt.show()
    #exit()

    current_step = position_data[:-1*_NEXT_STEP]
    next_step = position_data[_NEXT_STEP:]
    _, data_wav = wavfile.read(wavefile)
    silence_wav = np.zeros((_ADD_SILENCE*_FREQ_SAMP,),dtype=np.float32)     
    data_max = (np.amax(np.absolute(data_wav.astype(np.float32)))) 
    data_max = data_max if data_max != 0 else 1 
    data_wav = data_wav.astype(np.float32) / data_max
    data_wav = np.concatenate((silence_wav,data_wav,silence_wav))
    data_wav = add_noise(data_wav, _SNR=snr)

    for frame in range(current_step.shape[0]):
        index_1 = int(int(frame/_FPS) * _FREQ_SAMP + _INDEXES[int(frame%_FPS)])
        index_2 = int(int(frame/_FPS) * _FREQ_SAMP + _INDEXES[int(frame%_FPS)+1])
        _tmp = np.zeros((frame_lenght,), dtype=np.float32)
        len2 = data_wav[index_1: index_2].shape[0] 
        _tmp[0:len2] = data_wav[index_1: index_2]
        stft_data = single_spectrogram(_tmp, _FREQ_SAMP, _WLEN, _HOP) 
        stft_data =  stft_data * slope_wav + intersec_wav
        if frame ==0:
            audiodata = np.zeros((current_step.shape[0], 1, stft_data.shape[1], stft_data.shape[0]), dtype = np.float32) 
        audiodata[frame, 0] = np.swapaxes(stft_data, 0,1)

    return audiodata, current_step, next_step, correction_name

def training_data(fileslist):
    global pos_max, pos_min, audio_max, audio_min, _OUTPUT_FOLDER, corrections_name, corrections_frame
    print('### Preparing dataset...')
    div = (pos_max-pos_min)
    div[div==0] = 1
    slope_pos = (_RANG_POS[1]-_RANG_POS[0])/ div
    intersec_pos = _RANG_POS[1] - slope_pos * pos_max
    slope_wav = (_RANG_WAV[1]-_RANG_WAV[0]) / (audio_max-audio_min)
    intersec_wav = _RANG_WAV[1] - slope_wav * audio_max
    #_audio, _motion, _ = format_data(fileslist[j],_SNR_LIST[i], slope_pos, intersec_pos, slope_wav, intersec_wav)
    with open(_CORRECTION_FILE) as f:
        corrections = f.read().split('\r\n')[:-1]
    corrections = [x.split('\t') for x in corrections]
    corrections_name = [x[0] for x in corrections]
    corrections_frame = [int(x[1]) for x in corrections]
    for i in range(len(_SNR_LIST)):
        for j in range(len(fileslist)):
            audiodata, current_step, next_step, wavefile = format_data(fileslist[j],_SNR_LIST[i], slope_pos, intersec_pos, slope_wav, intersec_wav)
            print(wavefile)
            h5file = '{}/motion_file_D{:03d}.h5'.format(_OUTPUT_FOLDER, i*len(fileslist)+j)
            with h5py.File(h5file, 'a') as f:
                ds = f.create_dataset('input', data=audiodata)
                ds = f.create_dataset('current', data=current_step)
                ds = f.create_dataset('next', data=next_step)
    return



def main():
    global pos_max, pos_min, audio_max, audio_min
    list_htr_files = glob.glob('{}/MOCAP/HTR/{}_*.htr'.format(_DATA_FOLDER, _EXP))
    if not os.path.exists(_FILE_POSMINMAX):
        calculate_minmax(list_htr_files)
    pos_max = np.zeros((1,len(_PARTS)*3), dtype=np.float32)
    pos_min = np.zeros((1,len(_PARTS)*3), dtype=np.float32)
    with h5py.File(_FILE_POSMINMAX, 'r') as f:
        np.copyto(pos_min, f['minmax'][0,:])
        np.copyto(pos_max, f['minmax'][1,:])
    audio_max = np.zeros((1, 129), dtype=np.float32)
    audio_min = np.ones((1, 129), dtype=np.float32) * -120.
    if not os.path.exists(_OUTPUT_FOLDER):
        os.makedirs(_OUTPUT_FOLDER)
    training_data(list_htr_files)

def parsers():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp', '-e', type=str, help='Music Genre for the experiment')
    argparser.add_argument('--data', '-d', type=str, help='Data Folder')
    argparser.add_argument('--output', '-o', type=str, help='Output Data')
    args=argparser.parse_args()
    return args

if __name__ == '__main__':

    args = parsers()
    _NEXT_STEP = 1
    _EXP = args.exp
    _DATA_FOLDER = args.data
    _OUTPUT_FOLDER = '{}/UE_{}_NEXT_{}'.format(args.output, _EXP,_NEXT_STEP)
    _FILE_POSMINMAX = '{}/{}_pos_minmax.h5'.format(_DATA_FOLDER, _EXP)
    _CORRECTION_FILE = '{}/MOCAP/correction.txt'.format(_DATA_FOLDER)
    _SNR_LIST = ['Clean']

    frame_lenght = 0
    for fps in range(_FPS):
        frame_lenght = np.amax((frame_lenght, _INDEXES[fps+1] - _INDEXES[fps])) 
    main()
