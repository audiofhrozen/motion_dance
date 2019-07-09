#!/usr/bin/python
"""
    How to use:
    python beattap.py --out ./Annotations --pfx user_1 --wav ./train/bachata_03.mp3
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import signal
import sys
from threading import Thread
import time
import vlc

if sys.version_info[0] < 3:
    import Tkinter as tk
    this_input = raw_input
else:
    import tkinter as tk
    this_input = input


def signal_handler(signal, frame):
    # CTRL+C Finish
    sys.exit(0)


def optimize_beat(wavfile, bpm_user, margin):
    from madmom.features import beats
    proc = beats.BeatTrackingProcessor(fps=100)
    act = beats.RNNBeatProcessor()(wavfile)
    bpm = proc(act)
    bpm_mdm = np.unique(bpm)
    for i in range(bpm_user.shape[0]):
        idx = bpm_mdm[(bpm_mdm >= bpm_user[i] - margin) & (bpm_mdm <= bpm_user[i] + margin)]
        if len(idx) > 0:
            bpm_user[i] = idx[0]
    # fileout = 'tmp_madmom_beat.txt'
    # np.savetxt(fileout, bpm_mdm, fmt='%.09f', newline='\n')
    return bpm_user


class BPMApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry('640x480')
        self.configure(background='black')
        self.bpm_count = []
        self.delay = 0  # in seconds

        self.label_init = tk.Label(self, text='Press any key to record the beat...\n ', bg='black', fg='white',
                                   font=('Comic Sans MS', 12))
        self.label_init.pack()

        self.v_bpm = tk.StringVar()
        self.label_bpm = tk.Label(self, textvariable=self.v_bpm, bg='black', fg='white', font=('Comic Sans MS', 50))
        self.label_bpm.pack()
        self.v_bpm.set('0 bpm')

        self.v_cnt = tk.StringVar()
        self.count = tk.Label(self, textvariable=self.v_cnt, bg='black', fg='white', font=('Comic Sans MS', 12))
        self.count.pack()
        self.cnt = ''
        self.v_cnt.set(self.cnt)

        self.bind('<KeyPress>', self.onKeyPress)
        self.focus_force()
        self.start = 0

    def onKeyPress(self, event):
        t = time.time()
        self.bpm_count += [t - self.start - self.delay]
        if len(self.bpm_count) > 1:
            bpm = 60.0 / (self.bpm_count[-1] - self.bpm_count[-2])
            self.v_bpm.set('{:2.0f} bpm'.format(bpm))
        self.cnt += '.'
        if len(self.cnt) > 100:
            self.cnt = '.'
        self.v_cnt.set(self.cnt)

    def PlaySound(self, audiosample, delay):
        self.delay = delay
        # self.player = pyglet.media.Player()
        # self.player = vlc.MediaPlayer#('{}'.format(audiosample))
        # self.music_file = pyglet.media.load(audiosample)
        self.player = vlc.MediaPlayer('{}'.format(audiosample))
        sound_thread = Thread(target=self.startPlaying)
        sound_thread.start()

    def startPlaying(self):
        # self.player.queue(self.music_file)
        self.player.play()
        self.start = time.time()
        # pyglet.app.run()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out', '-o', type=str,
                           help='Directory to save the file')
    argparser.add_argument('--pfx', '-p', type=str,
                           help='Prefix save file')
    argparser.add_argument('--wav', '-w', type=str,
                           help='Wavefile')
    argparser.add_argument('--delay', '-d', type=float,
                           help='Delay of the Tap', default=0.12)
    argparser.add_argument('--optimize', '-z', type=bool,
                           help='Optimize of the beats with MADMOM as reference', default=True)
    argparser.add_argument('--margin', '-m', type=float,
                           help='Error Margin of the bpm between lib and user', default=0.5)
    argparser.add_argument('--annotation', '-a', type=str,
                           help='Annotation for optimize if there is one')
    args = argparser.parse_args()
    if args.annotation is None:
        print("Beginning BPM Record")
        print("Press Enter key to begin or Ctrl+C to Exit...")
        this_input()
        root = BPMApp()
        root.PlaySound(args.wav, args.delay)
        root.mainloop()
        # pyglet.app.exit()

        user_bpm = np.asarray(root.bpm_count)
        fileout = os.path.basename(args.wav)
        fileout = os.path.splitext(fileout)[0]
        fileout = '{}/user/{}_{}.txt'.format(args.out, fileout, args.pfx)
        np.savetxt(fileout, user_bpm, fmt='%.09f')
    else:
        print("Loading file to optimize")
        user_bpm = np.loadtxt(args.annotation)

    if args.optimize:
        user_bpm = optimize_beat(args.wav, user_bpm, args.margin)
        fileout = os.path.basename(args.wav)
        fileout = os.path.splitext(fileout)[0]
        fileout = '{}/optimized/{}.txt'.format(args.out, fileout)
        np.savetxt(fileout, user_bpm, fmt='%.09f', newline='\n')

    print('Record Finished')
    return


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
