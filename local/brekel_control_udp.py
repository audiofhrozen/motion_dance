#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import datetime
import os
import signal
import socket
import sys
from sys import stdout
from time import sleep
import vlc


UDP_IP = "192.168.170.110"
UDP_PORT = 8880
SLEEP = 15
PID = os.getpid()
Computer_Name = socket.gethostname()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP


def signal_handler(signal, frame):
    """CTRL+C"""
    print('previous stop')
    MESSAGE = 'Brekel_recording_stop\t{}\t{}'.format(Computer_Name, PID)
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def main():
    print('Network trigger for Brekel')

    audio_name = os.path.basename(args.audiofile)
    audio_name = audio_name.split('.')[0]
    filename = 'Motion_{}_{}'.format(audio_name, datetime.datetime.now().strftime('%Y-%m-%d_%H%M'))
    MESSAGE = 'Brekel_recording_start\t{}\t{}\t{}\t'.format(Computer_Name, PID, filename)

    print("Server IP:{}\t Port:{}".format(UDP_IP, UDP_PORT))
    print("Filename: {}".format(filename))

    for i in range(SLEEP):
        stdout.write('The program will start recording in : {} sec(s)\r'.format(SLEEP - i))
        stdout.flush()
        sleep(1)

    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    print('Recording...', ' ' * 35)
    p = vlc.MediaPlayer('{}'.format(args.audiofile))
    p.play()

    sleep(0.1)
    # p.audio_set_volume(100)
    length = p.get_length() / 1000 - 2  # convert to seconds
    sleep(length)

    MESSAGE = 'Brekel_recording_stop\t{}\t{}'.format(Computer_Name, PID)
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Program based on Chainer Framework')
    parser.add_argument('--audiofile', '-w', type=str, help='Audio File')
    args = parser.parse_args()
    main()
