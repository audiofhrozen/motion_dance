#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, h5py, argparse, glob
import numpy as np

parser = argparse.ArgumentParser(description='BPM alignment for hark files')
parser.add_argument('--folder', '-f', type=str, help='Folder to read the Annotation files')
#parser.add_argument('--h5folder', '-d', type=str, help='Folder to save the bpm info in array (HDF5)')
args = parser.parse_args()

def catch_hark_bpm(filename):
    with open(filename) as f:
        readdata = f.read().split('\n')
    print(filename)
    beats=[]
    for i in range(len(readdata)):
        if readdata[i].startswith('Beat detected'):
            beat=float(readdata[i].replace('Beat detected at time: ', ''))
            beats+=[(160*beat)/16000]
    print(beats)
    exit()
    return


def main():
    files = glob.glob('{}/*.txt'.format(args.harkfolder))
    sorted(files)
    for fn in files:
        bpm_info= catch_bpm(fn)
        savefile= fn.replace('txt', 'h5')
        if os.path.exists(savefile):
            os.remove(savefile)
        with h5py.File(savefile,'a') as f:
            ds = f.create_dataset('bpm', data=bpm_info)
    pass

if __name__=='__main__':
    main()
