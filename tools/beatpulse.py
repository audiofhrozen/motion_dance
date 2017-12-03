#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os, h5py, time, pyglet, signal, sys
from threading import Thread
import numpy as np
import Tkinter as tk

#pyglet.lib.load_library('avbin')
#pyglet.have_avbin=True

"""CTRL+C Finish"""
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

class BPMApp(tk.Tk):
	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		self.geometry('640x480')
		self.configure(background='black')
		self.bpm_count = []

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
		self.bpm_count +=[t-self.start]
		if len(self.bpm_count)>1:
			bpm = 60.0/(self.bpm_count[-1]-self.bpm_count[-2])
			self.v_bpm.set('{:2.0f} bpm'.format(bpm))
		self.cnt +='.'
		if len(self.cnt)>100:
			self.cnt='.'
		self.v_cnt.set(self.cnt)

	def PlaySound(self, audiosample):
		self.player = pyglet.media.Player()
		self.music_file = pyglet.media.load(audiosample)
		sound_thread = Thread(target=self.startPlaying)
		sound_thread.start()

	def startPlaying(self):
		self.player.queue(self.music_file)
		self.player.play()
		self.start = time.time()
		pyglet.app.run()

argparser = argparse.ArgumentParser()
argparser.add_argument('--out', '-o', type=str, help='Directory to save the file')
argparser.add_argument('--pfx', '-p', type=str, help='Prefix save file')
argparser.add_argument('--wav', '-w', type=str, help='Wavefile')
args=argparser.parse_args()

def main():
	print("Beginning BPM Record")
	print("Press Enter key to begin or Ctrl+C to Exit...")
	raw_input()
	root = BPMApp()
	root.PlaySound(args.wav)
	root.mainloop()
	pyglet.app.exit()

	fileout = os.path.basename(args.wav)
	fileout = fileout.replace('.wav', '')
	fileout = '{}/{}_{}.h5'.format(args.out,fileout, args.pfx)

	if os.path.exists(fileout):
		os.remove(fileout)

	with h5py.File(fileout,'a') as f:
		dset = f.create_dataset('beats', data= root.bpm_count)

	print('Record Finished')
	return

if __name__ == '__main__':
	main()
