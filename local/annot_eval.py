#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, h5py, argparse, glob
from scipy import signal
import numpy as np
import BTET.beat_evaluation_toolbox as be
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='BPM Evaluation')
parser.add_argument('--folder', '-f', type=str, help='Motion data folder')
parser.add_argument('--exp', '-e', type=str, help='Experiment type')
parser.add_argument('--display', '-d', type=str, help='Display Plot')
parser.add_argument('--output', '-o', type=str, help='Save Image')
args = parser.parse_args()

if not bool(strtobool(args.display)):
    import matplotlib as mpl
    mpl.use('Agg')
from matplotlib import pyplot as plt

_NUM, _DEN = signal.butter(3, 0.05)


def plot_vals(labels, means, stds, comp, title, idx):
    ind = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots()
    rects = [None] *means.shape[1]
    colors = ['r', 'b', 'g']
    for i in range(means.shape[1]):
        rects[i] = ax.bar(ind + i*width, means[:,i], width, color=colors[i], yerr=[stds[:,i,0],stds[:,i,1]]) #
    #rects2 = ax.bar(ind + width, means[:,1], width, color='b', yerr=[stds[:,1,0],stds[:,1,1]]) #
    ax.set_title(title)
    ax.set_xticks(ind + width*i / 2)
    ax.set_xticklabels(labels, rotation=40)
    ax.legend(rects, comp)
    if args.display:
        plt.show()
    fig.savefig('{}/init_bpm_results_{}.png'.format(args.output, idx))
    return

def catch_hark_bpm(filename):
    with open(filename) as f:
        readdata = f.read().split('\n')
    beats=[]
    for i in range(len(readdata)):
        if readdata[i].startswith('Beat detected'):
            beat=float(readdata[i].replace('Beat detected at time: ', ''))
            beats+=[(160*beat)/16000]
    beats = np.asarray(beats)
    return beats

def catch_from_htr(filename):
    with open(filename, 'rb') as f:
        _DATA_FILE = f.read().split('\r\n')
        num_frames = int(_DATA_FILE[5].split(' ')[1])

    _PARTS = ['pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 
        'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'pelvis']
    _ROTATIONS = np.zeros((len(_PARTS),num_frames,3), dtype=np.float32)

    for i in range(len(_PARTS)):
        k = _DATA_FILE.index('[{}]'.format(_PARTS[i]))
        for j in range(num_frames):
            _tmp_data = np.asarray([float(x) for x in _DATA_FILE[k+j+1].split('\t')])
            _ROTATIONS[i,j] = _tmp_data[1:4] if i == 0 else _tmp_data[4:7]  #Position Translation 1,4 ; Rotations 4,7#_tmp_data 

    for i in [0]: #0,7,8,11,12 
        new_signal = signal.filtfilt(_NUM, _DEN, np.gradient(_ROTATIONS[i,:,0]))
        zero_crossings = np.where(np.diff(np.signbit(new_signal)))[0].astype(np.float64)
    return (zero_crossings/30.0)


def catch_from_h5(filename):
    with open(filename, 'rb') as f:
        _DATA_FILE = f.read().split('\r\n')
        num_frames = int(_DATA_FILE[5].split(' ')[1])

    _PARTS = ['pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 
        'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'pelvis']
    _ROTATIONS = np.zeros((len(_PARTS),num_frames,3), dtype=np.float32)

    for i in range(len(_PARTS)):
        k = _DATA_FILE.index('[{}]'.format(_PARTS[i]))
        for j in range(num_frames):
            _tmp_data = np.asarray([float(x) for x in _DATA_FILE[k+j+1].split('\t')])
            _ROTATIONS[i,j] = _tmp_data[1:4] if i == 0 else _tmp_data[4:7]  #Position Translation 1,4 ; Rotations 4,7#_tmp_data 

    for i in [0]: #0,7,8,11,12 
        new_signal = signal.filtfilt(_NUM, _DEN, np.gradient(_ROTATIONS[i,:,0]))
        zero_crossings = np.where(np.diff(np.signbit(new_signal)))[0].astype(np.float64)
    return (zero_crossings/30.0)


def main():
    filelist = glob.glob('{}/{}_*'.format(args.folder, args.exp)) 
    per_bpm = []
    hark_bpm = []
    mad_bpm = []   
    mot_bpm = []
    for fn in filelist:
        print(fn)
        mot_bpm += [catch_from_htr(fn)]  

        per_fn = fn.replace('MOCAP/HTR/{}_'.format(args.exp), 'Annotations/PER/')
        per_fn = per_fn.replace('.htr', '_user_1.h5')
        if per_bpm is not None:
            try:
                with h5py.File(per_fn) as f:
                    beats = np.zeros(f['beats'].shape)
                    np.copyto(beats, f['beats'])
                    beats = np.unique(beats)
                per_bpm += [beats]
            except Exception as e:
                print('No annotation files from user found, skipping evaluations from user')
                per_bpm = None

        hark_fn = per_fn.replace('PER', 'HARK')
        hark_fn = hark_fn.replace('_user_1.h5', '.txt')
        hark_bpm += [catch_hark_bpm(hark_fn)]      

        mad_fn = hark_fn.replace('HARK', 'MADMOM')
        mad_fn = mad_fn.replace('txt', 'h5')
        with h5py.File(mad_fn) as f:
            bpm = np.zeros(f['bpm'].shape)
            np.copyto(bpm, f['bpm'])
            bpm = np.unique(bpm)
        mad_bpm +=[bpm]
    evals_name = ['fMeasure', 'cemgilAcc', 'gotoAcc', 'pScore', 'cmlC', 'cmlT', 'amlC', 'amlT']#R_hark['scores'].keys()

    if per_bpm is not None:
        print('Evaluating HARK bpm')
        R_hark = be.evaluate_db(per_bpm,hark_bpm,measures='all', doCI=True)

        print('Evaluating MADMOM bpm')
        R_mad = be.evaluate_db(per_bpm,mad_bpm,measures='all', doCI=True)

        print('Evaluating MOTION bpm')
        R_mot = be.evaluate_db(per_bpm,mot_bpm,measures='all', doCI=True)

        results =[R_hark, R_mad, R_mot]
        
        evals_mean = np.zeros((len(evals_name), len(results)))
        evals_std = np.zeros((len(evals_name), len(results),2))
        for i in range(len(evals_name)):
            for j in range(len(results)):
                evals_mean[i,j] = results[j]['scores_mean'][evals_name[i]]
                evals_std[i, j, 0] = np.abs(results[j]['scores_conf'][evals_name[i]][0] - evals_mean[i,j])
                evals_std[i, j, 1] = np.abs(results[j]['scores_conf'][evals_name[i]][1] - evals_mean[i,j])
        res_label=['HARK', 'Madmom', 'Dancer']
        plot_vals(evals_name, evals_mean, evals_std, res_label, 'Bootstrapping 95% confidence interval w.r.t. annotator', 1)
    
    print('Evaluating HARK-MOTION bpm')
    R_hark = be.evaluate_db(hark_bpm, mot_bpm,measures='all', doCI=True)

    print('Evaluating MADMOM-MOTION bpm')
    R_mad = be.evaluate_db(mad_bpm,mot_bpm,measures='all', doCI=True)

    results =[R_hark, R_mad]
    if per_bpm is not None:
        results = [R_mot] +results
    evals_mean = np.zeros((len(evals_name), len(results)))
    evals_std = np.zeros((len(evals_name), len(results),2))
    for i in range(len(evals_name)):
        for j in range(len(results)):
            evals_mean[i,j] = results[j]['scores_mean'][evals_name[i]]
            evals_std[i, j, 0] = np.abs(results[j]['scores_conf'][evals_name[i]][0] - evals_mean[i,j])
            evals_std[i, j, 1] = np.abs(results[j]['scores_conf'][evals_name[i]][1] - evals_mean[i,j])

    res_label=['HARK', 'Madmom']
    if per_bpm is not None:
        res_label= ['Dancer'] +res_label
    plot_vals(evals_name, evals_mean, evals_std, res_label, 'Bootstrapping 95% confidence interval of dancer w.r.t. BPM Trackers',2)

if __name__=='__main__':
    main()

