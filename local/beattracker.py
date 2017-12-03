#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, h5py, argparse, glob
import numpy as np
from madmom.features import beats

parser = argparse.ArgumentParser(description='BPM with madmom')
parser.add_argument('--infile', '-i', type=str, help='Wave file to process')
parser.add_argument('--outfile', '-o', type=str, help='File to store the bpm (HDF5)')
args = parser.parse_args()

def bpm_at_madmom(filename):
    proc = beats.BeatTrackingProcessor(fps=100)
    act = beats.RNNBeatProcessor()(filename)
    bpm = proc(act)
    return bpm

def main():
    beat_mtx = bpm_at_madmom(args.infile)
    if os.path.exists(args.outfile):
        os.remove(args.outfile)
    with h5py.File(args.outfile, 'a') as f:
        ds = f.create_dataset('bpm', data=beat_mtx) 
    return

if __name__=='__main__':
    main()

