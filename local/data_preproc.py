#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, os, h5py, argparse, shutil
import numpy as np
from scipy.io import wavfile 
from python_utils.audio import *
from motion_config import Configurations
from transforms3d.euler import euler2quat, quat2euler
from python_utils.print_utils import print_info, print_warning, print_error
from python_utils.maths import deg2rad, rad2deg

def calculate_minmax(fileslist):
    print_info('### MinMax file not found...')
    print_info('### Creating MinMax file...')
    if len(fileslist) < 1:
        raise ValueError('No files were found in the folder ...')
    for item in fileslist:
        #print(item)
        with open(item, 'rb') as f:
            data_file = f.read().split('\r\n')
        num_frames = int(data_file[5].split(' ')[1])
        tmp_minmax = np.zeros((gnrl_conf.pos_dim,2), dtype=np.float32)
        for i in range(len(gnrl_conf.parts)):
            j = data_file.index('[{}]'.format(gnrl_conf.parts[i]))
            array_data = data_file[j+1:j+num_frames+1]
            array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
            if i == 0:
                array_data = array_data[:,1:4] / 100.0 # Scaled Position Translation
                init_trans = np.mean(array_data[0:31], axis=0)
                array_data -= init_trans
                tmp_minmax[0:3, 0] =  np.amin(array_data, axis = 0) # minimum
                tmp_minmax[0:3, 1] =  np.amax(array_data, axis = 0) # maximum
            else:
                array_data = array_data[:,4:7] #Rotations
                array_data = deg2rad(array_data)
                if gnrl_conf.rot_type=='euler':
                    new_array = (array_data % (2*np.pi))#* np.sign(array_data) 
                    tmp_minmax[i*3:i*3+3, 0] =  np.amin(new_array, axis = 0) # minimum
                    tmp_minmax[i*3:i*3+3, 1] =  np.amax(new_array, axis = 0) # maximum                    
                elif gnrl_conf.rot_type=='quat':
                    new_array = np.zeros((num_frames, 4),dtype=np.float32)
                    for k in range(num_frames):
                        new_array[k] = euler2quat(array_data[k,0],array_data[k,1],array_data[k,2])                
                    tmp_minmax[i*4-1:(i+1)*4-1, 0] =  np.amin(new_array, axis = 0) # minimum
                    tmp_minmax[i*4-1:(i+1)*4-1, 1] =  np.amax(new_array, axis = 0) # maximum
                else:
                    print_error('Incorrect type of rotation')
                    raise TypeError('.')
        if not 'pos_minmax' in locals():
            pos_minmax = np.zeros((gnrl_conf.pos_dim,2), dtype=np.float32)
            pos_minmax[:,0] = tmp_minmax[:,1]
            pos_minmax[:,1] = tmp_minmax[:,0]
        pos_minmax[:,0]= np.amin([tmp_minmax[:,0],pos_minmax[:,0]], axis=0) # minimum
        pos_minmax[:,1]= np.amax([tmp_minmax[:,1],pos_minmax[:,1]], axis=0) # maximum
    with h5py.File(gnrl_conf.file_pos_minmax, 'a') as f:
        ds = f.create_dataset('minmax', data=pos_minmax.T)
    return

def format_data(filename, snr, Configs):
    wavefile = filename.replace('MOCAP/HTR', 'WAVE/SOX')
    wavefile = wavefile.replace('.htr', '.wav')
    wavefile = wavefile.replace('{}_'.format(Configs.exp), '')
    wavefile = wavefile.replace('test_', '')
    correction_name = filename.replace('{}/MOCAP/HTR/'.format(Configs.data_folder), '')
    correction_name = correction_name.replace('.htr', '')
    correction_name = correction_name.replace('test_', '')
    correct_idx = Configs.correct_name.index(correction_name)
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
    position_data = np.zeros((num_frames, Configs.pos_dim), dtype=np.float32)

    for idx in range(len(Configs.parts)):
        j = data_file.index('[{}]'.format(Configs.parts[idx]))
        array_data = data_file[j+1:j+num_frames+1]
        array_data = np.loadtxt(array_data, delimiter='\t', dtype=np.float32)
        if idx == 0:
            array_data = array_data[:,1:4] / 100.0 # Scaled Position Translation 
            init_trans = np.mean(array_data[0:31], axis=0)
            array_data -= init_trans
            position_data[:, 0:3] = array_data
        else:
            array_data = array_data[:,4:7] #Rotations
            array_data = deg2rad(array_data)
            if Configs.rot_type=='euler':
                position_data[:, idx*3:idx*3+3] =  (array_data % (2*np.pi))#* np.sign(array_data)              
            elif Configs.rot_type=='quat':
                new_array = np.zeros((num_frames, 4),dtype=np.float32)
                for k in range(num_frames):
                    new_array[k] = euler2quat(array_data[k,0],array_data[k,1],array_data[k,2])   
                position_data[:, idx*4-1:idx*4+3] = new_array
            else:
                raise TypeError('Incorrect type of rotation')


    position_data = position_data[Configs.correct_frm[correct_idx]:,] 
    position_data = position_data * Configs.slope_pos + Configs.intersec_pos
    silence_pos = np.ones((Configs.add_silence*Configs.fps, position_data.shape[1]), dtype=np.float32) * position_data[0,:]
    position_data = np.concatenate((silence_pos, position_data, silence_pos), axis=0)            

    #print(position_data.shape)
    #for i in range(51):
    #    f = plt.figure()
    #    plt.plot(position_data[:,i])
    #    f.savefig('echo_{:02d}.png'.format(i), dpi=f.dpi)
    #    plt.close(f)
    #plt.show()
    #exit()

    current_step = position_data[:-1*Configs.next_step]
    next_step = position_data[Configs.next_step:]
    _, data_wav = wavfile.read(wavefile)
    silence_wav = np.random.rand(Configs.add_silence*Configs.frq_smp).astype(np.float32)*(10**-5)     
    data_max = (np.amax(np.absolute(data_wav.astype(np.float32)))) 
    data_max = data_max if data_max != 0 else 1 
    data_wav = data_wav.astype(np.float32) / data_max
    data_wav = np.concatenate((silence_wav,data_wav,silence_wav))
    data_wav = add_noise(data_wav, _SNR=snr)

    for frame in range(current_step.shape[0]):
        index_1 = int(int(frame/Configs.fps) * Configs.frq_smp + Configs.indexes[int(frame%Configs.fps)])
        index_2 = int(int(frame/Configs.fps) * Configs.frq_smp + Configs.indexes[int(frame%Configs.fps)+1])
        _tmp = np.zeros((Configs.frame_lenght,), dtype=np.float32)
        len2 = data_wav[index_1: index_2].shape[0] 
        _tmp[0:len2] = data_wav[index_1: index_2]
        stft_data = single_spectrogram(_tmp, Configs.frq_smp, Configs.wlen, Configs.hop) 
        stft_data =  stft_data * Configs.slope_wav + Configs.intersec_wav
        if frame ==0:
            audiodata = np.zeros((current_step.shape[0], 1, stft_data.shape[1], stft_data.shape[0]), dtype = np.float32) 
        audiodata[frame, 0] = np.swapaxes(stft_data, 0,1)

    return audiodata, current_step, next_step, correction_name

def main():
    list_htr_files = glob.glob('{}/MOCAP/HTR/{}_*.htr'.format(gnrl_conf.data_folder, gnrl_conf.exp))
    if not os.path.exists(gnrl_conf.file_pos_minmax ):
        calculate_minmax(list_htr_files)
    pos_max = np.zeros((1,gnrl_conf.pos_dim), dtype=np.float32)
    pos_min = np.zeros((1,gnrl_conf.pos_dim), dtype=np.float32)
    with h5py.File(gnrl_conf.file_pos_minmax, 'r') as f:
        np.copyto(pos_min, f['minmax'][0,:])
        np.copyto(pos_max, f['minmax'][1,:])
    audio_max = np.zeros((1, 129), dtype=np.float32)
    audio_min = np.ones((1, 129), dtype=np.float32) * -120.
    if os.path.exists(gnrl_conf.out_folder):
        shutil.rmtree(gnrl_conf.out_folder)
    os.makedirs(gnrl_conf.out_folder)

    print('### Preparing dataset...')
    min_val = np.ones((pos_max.shape), dtype=np.float32)* gnrl_conf.rng_pos[0]
    max_val = np.ones((pos_max.shape), dtype=np.float32)* gnrl_conf.rng_pos[1]    
    #for i in range(pos_max.shape[1]):
    #    if abs(pos_max[0,i]-pos_min[0,i]) < 0.2: #Reduce noise in axis with less than 20 degrees only for training
    #        min_val[0,i] *= 0.05
    #        max_val[0,i] *= 0.05
    div = (pos_max-pos_min)
    div[div==0] = 1
    gnrl_conf.slope_pos = (max_val-min_val)/ div
    gnrl_conf.intersec_pos = max_val - gnrl_conf.slope_pos * pos_max
    gnrl_conf.slope_wav = (gnrl_conf.rng_wav[1]-gnrl_conf.rng_wav[0]) / (audio_max-audio_min)
    gnrl_conf.intersec_wav = gnrl_conf.rng_wav[1] - gnrl_conf.slope_wav * audio_max
    with open(gnrl_conf.correct_file) as f:
        corrections = f.read().split('\r\n')[:-1]
    corrections = [x.split('\t') for x in corrections]
    gnrl_conf.correct_name = [x[0] for x in corrections]
    gnrl_conf.correct_frm = [int(x[1]) for x in corrections]
    for j in range(len(list_htr_files)):
        for i in range(len(gnrl_conf.snr_lst)):
            audiodata, current_step, next_step, wavefile = format_data(list_htr_files[j],gnrl_conf.snr_lst[i], gnrl_conf)
            print(wavefile)
            h5file = '{}/motion_file_D{:03d}.h5'.format(gnrl_conf.out_folder, j*len(gnrl_conf.snr_lst)+i)
            with h5py.File(h5file, 'a') as f:
                ds = f.create_dataset('input', data=audiodata)
                ds = f.create_dataset('current', data=current_step)
                ds = f.create_dataset('next', data=next_step)

def arg_parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp', '-e', type=str, help='Music Genre for the experiment')
    argparser.add_argument('--data', '-d', type=str, help='Data Folder')
    argparser.add_argument('--out', '-o', type=str, help='Output Data')
    argparser.add_argument('--rot', '-r', type=str, help='Type of Rotations')
    args=argparser.parse_args()
    return args

if __name__ == '__main__':

    args = arg_parse()
    _next=1
    gnrl_conf=Configurations(_next, args.data, args.exp, args.rot)
    gnrl_conf.correct_file = '{}/MOCAP/correction.txt'.format(args.data)
    gnrl_conf.snr_lst = ['Clean', 20, 10]
    gnrl_conf.file_pos_minmax = '{}/{}_pos_minmax_{}.h5'.format(args.data, args.exp, args.rot)
    gnrl_conf.out_folder = '{}/UE_{}_NEXT_{}_ROT_{}'.format(args.out, args.exp,_next, args.rot) 

    main()
