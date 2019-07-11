#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings

try:
    # TODO(nelson): Delete this part when H5PY message is fixed.
    warnings.filterwarnings('ignore')
except Exception as e:
    pass

import beat_evaluation_toolbox as be
import chainer
import glob
import h5py
import imp
import logging
import os
import platform
import six
import timeit

from chainer import cuda
from chainer import serializers
from chainer import Variable

from deepdancer.motion import calculate_rom
from deepdancer.motion import Configuration
from deepdancer.motion import extract_beats
from deepdancer.motion import render_motion

import numpy as np
import pandas
from scipy import signal
from scipy import stats
import soundfile

from deepdancer.utils import add_noise
from python_speech_features.sigproc import framesig
from python_speech_features.sigproc import logpowspec


def format_audio(audioname, noise, snr, freq_samp, wav_range):
    # TODO(nelson): Add these values as arguments
    audio_max = 5.
    audio_min = -230.
    slope = (config['rng_wav'][1] - config['rng_wav'][0]) / (audio_max - audio_min)
    intersec = config['rng_wav'][1] - slope * audio_max
    wavname = audioname.replace('MP3', 'WAVE')
    wavname = wavname.replace('.mp3', '.wav')

    if not os.path.exists(wavname):
        if platform == 'Windows':
            cmmd = 'ffmpeg -loglevel panic -y -i {} -acodec pcm_s16le -ar {} -ac 1 {}'.format(
                audioname, freq_samp, wavname)
            logging.info(cmmd)
            subprocess.Popen(cmmd, shell=False).communicate()
        elif platform == 'Linux':
            os.system('sox {} -c 1 -r {} {}'.format(audioname, freq_samp, wavname))
        else:
            logging.error('OS not supported')
            raise TypeError('OS not supported')
    data_wav, _ = soundfile.read(wavname)
    data_wav /= np.amax(np.abs(data_wav))
    data_wav = add_noise(data_wav, noise, snr)
    idxs = np.linspace(0, freq_samp, 31, endpoint=True, dtype=np.int)
    audio_length = np.ceil(data_wav.shape[0] * args.fps / args.freq).astype(np.int)
    stft_data = np.ones((audio_length, 1, 129, 5), dtype=np.float32) * config['rng_wav'][0]

    NFFT = int(2**(np.ceil(np.log2(config['wlen']))))

    for i in six.moves.range(audio_length):
        prv = idxs[i % args.fps] + freq_samp * int(i / args.fps)
        loc = idxs[i % args.fps + 1] + freq_samp * int(i / args.fps)
        _tmp = np.zeros((config['frame_lenght'],), dtype=np.float32)
        len2 = data_wav[prv:loc].shape[0]
        _tmp[0:len2] = data_wav[prv:loc]
        frames = framesig(_tmp, config['wlen'], config['hop'], winfunc=lambda x: np.hamming(x))
        stft = logpowspec(frames, NFFT)
        # stft = single_spectrogram(data_wav[prv:loc], freq_samp, args.wlen, args.hop)
        # hops = stft.shape[0]
        stft_data[i, 0] = np.swapaxes(stft, 0, 1)
    del data_wav
    stft_data = (stft_data * slope) + intersec
    return stft_data


def metrics(predicted_motion, music_beat, motion_beat_idx, dance_step):
    music_beat_frame = music_beat * args.fps
    music_beat_frame = music_beat_frame.astype(np.int)

    # TODO(nelson): Change evaluation to position rather than rotations
    # F-Score Eval
    motion_beat_frame = calculate_rom(predicted_motion[:, 3:], args.alignframe)
    motion_beat_frame = extract_beats(music_beat_frame, motion_beat_frame, args.alignframe)
    motion_beat = motion_beat_frame.astype(np.float) / float(args.fps)
    fscore, prec, recall, acc = be.fMeasure(music_beat, motion_beat)

    # Entropy
    entropy_eval = np.zeros((args.beats_skips))
    beat_length = motion_beat_idx[0, 2]

    # Entropy w.r.t. the initial dance step & delay
    for i in six.moves.range(16):
        entropy_test = list()
        for k in six.moves.range(i, music_beat.shape[0] - beat_length, beat_length):
            start = music_beat_frame[k]
            stop = music_beat_frame[k + beat_length]
            if stop >= predicted_motion.shape[0]:
                break
            eval_step = predicted_motion[start:stop]
            new_lenght = eval_step.shape[0]
            resample_step = signal.resample(dance_step, new_lenght)
            entropy_step = stats.entropy(eval_step, resample_step)  # *eval_step.shape[1]
            entropy_step[entropy_step == np.inf] = 0
            entropy_step = np.sum(entropy_step)
            entropy_test.append(entropy_step)
        entropy_eval[i] = np.mean(entropy_test)

    best_init_beat = np.argwhere(np.amin(entropy_eval) == entropy_eval)[0, 0]
    best_beat_entropy = entropy_eval[best_init_beat]

    # Entropy w.r.t. the trained dance step
    start = music_beat_frame[best_init_beat]
    stop = music_beat_frame[best_init_beat + beat_length]
    best_step = predicted_motion[start:stop]
    new_lenght = best_step.shape[0]
    resample_step = signal.resample(dance_step, new_lenght)
    entropy_step = stats.entropy(best_step, resample_step)
    entropy_step[entropy_step == np.inf] = 0
    best_step_entropy = np.sum(entropy_step)
    return fscore, prec, recall, acc, best_init_beat, best_beat_entropy, best_step_entropy


def main():
    with open(args.list) as f:
        trained_files = f.readlines()
        trained_files = [x.split('\n')[0] for x in trained_files]

    trained_list = []
    trained_align = []
    for x in trained_files:
        _file, _align = x.split('\t')
        trained_list += [_file]
        trained_align += [_align]

    if args.untrained == 1:
        with open(args.audio_list) as f:
            audio_list = f.readlines()
            audio_list = [x.split('\n')[0] for x in audio_list]
        untrain_list = []

        for i in six.moves.range(len(audio_list)):
            audioname = os.path.basename(audio_list[i]).split('.')[0]
            istrained = False
            for j in trained_list:
                if audioname in j:
                    istrained = True
                    break
            if not istrained:
                untrain_list.append(audio_list[i])

    with h5py.File(args.step_file) as F:
        motion_beat_idx = np.array(F['location'], copy=True)
        dance_step = np.array(F['dance_step'], copy=True)
    logging.info('Evaluation training...')
    logging.info('Loading model definition from {}'.format(args.network))
    logging.info('Using gpu {}'.format(args.gpu))

    net = imp.load_source('Network', args.network)
    audionet = imp.load_source('Network', os.path.join('./deepdancer/models', 'audio_nets.py'))
    model = net.Dancer(args.initOpt, getattr(audionet, args.encoder))
    serializers.load_hdf5(args.model, model)
    logging.info('Loading pretrained model from {}'.format(args.model))
    if args.gpu >= 0:
        model.to_gpu()

    minmaxfile = os.path.join('exp', 'data', '{}_{}'.format(args.exp, args.rot), 'minmax', 'pos_minmax.h5')
    with h5py.File(minmaxfile, 'r') as f:
        pos_min = f['minmax'][0:1, :]
        pos_max = f['minmax'][1:2, :]

    config['out_folder'] = os.path.join(args.folder, 'evaluation')
    div = (pos_max - pos_min)
    div[div == 0] = 1
    config['slope_pos'] = (config['rng_pos'][1] - config['rng_pos'][0]) / div
    config['intersec_pos'] = config['rng_pos'][1] - config['slope_pos'] * pos_max

    if not os.path.exists(config['out_folder']):
        os.makedirs(config['out_folder'])

    for i in six.moves.range(2):
        if i == 0:
            stage = 'trained'
            filelist = trained_list
        else:
            if args.untrained == 0:
                logging.info('Only trained data. Exit')
                exit()
            stage = 'untrained'
            filelist = untrain_list

        logging.info('Evaluating model for {} audio files'.format(stage))
        results = dict()
        results_keys = ['filename', 'noise', 'snr', 'fscore', 'precission', 'recall', 'acc',
                        'forward_time', 'init_beat', 'entropy_beat', 'entropy_step']
        for key in results_keys:
            results[key] = list()
        for j in six.moves.range(len(filelist)):
            if i == 0:
                mbfile = filelist[j].replace(os.path.join('MOCAP', 'HTR'), os.path.join('Annotations', 'corrected'))
                mbfile = mbfile.replace('{}_'.format(args.exp), '')
                mbfile = mbfile.replace('.htr', '.txt')

                audiofile = filelist[j].replace(os.path.join('MOCAP', 'HTR'), os.path.join('AUDIO', 'MP3'))
                audiofile = audiofile.replace('{}_'.format(args.exp), '')
                audiofile = audiofile.replace('.htr', '.mp3')
            else:
                mbfile = filelist[j].replace(os.path.join('AUDIO', 'MP3'), os.path.join('Annotations', 'corrected'))
                mbfile = mbfile.replace('.mp3', '.txt')

                audiofile = filelist[j]
            music_beat = np.unique(np.loadtxt(mbfile))
            filename = os.path.basename(mbfile).split('.')[0]

            for noise in list_noises:
                if noise == 'Clean':
                    list_snr = [None]
                else:
                    list_snr = snr_lst
                noise_mp3 = noise.replace(os.path.join('WAVE', ''), '')
                noise_mp3 = noise_mp3.replace('.wav', '.mp3')
                noise_name = noise
                if os.path.exists(noise_mp3):
                    noise_name = os.path.basename(noise_mp3).split('.')[0]
                    if not os.path.exists(noise):
                        logging.warning('Wavefile not found in folder, converting from mp3 file.')
                        noise_dir = os.path.dirname(noise)
                        if not os.path.exists(noise_dir):
                            os.makedirs(noise_dir)
                        if platform == 'Windows':
                            cmmd = 'ffmpeg -loglevel panic -y -i {} -acodec pcm_s16le -ar {} -ac 1 {}'.format(
                                noise_mp3, args.freq, noise)
                            subprocess.Popen(cmmd, shell=False).communicate()
                        elif platform == 'Linux':
                            os.system('sox {} -c 1 -r {} {}'.format(noise_mp3, args.freq, noise))
                        else:
                            raise TypeError('OS not supported')
                for snr in list_snr:
                    logging.info('Forwarding file: {} with noise: {} at snr: {}'.format(
                        audiofile, os.path.basename(noise), snr))
                    audio = format_audio(audiofile, noise, snr, args.freq, config['rng_wav'])
                    predicted_motion = np.zeros((audio.shape[0], args.initOpt[2]), dtype=np.float32)
                    feats = np.zeros((audio.shape[0] - 1, args.initOpt[1]), dtype=np.float32)
                    start = timeit.default_timer()
                    state = model.state
                    current_step = Variable(xp.asarray(np.zeros((1, args.initOpt[2]), dtype=np.float32)))
                    with chainer.no_backprop_mode(), chainer.using_config('train', False):
                        for k in six.moves.range(audio.shape[0] - 1):
                            audiofeat = model.audiofeat(Variable(xp.asarray(audio[k:k + 1])))
                            rnnAudio, state, out_step = model.forward(state, current_step, audiofeat, True)
                            if args.gpu >= 0:
                                predicted_motion[k + 1] = chainer.cuda.to_cpu(out_step.data)
                                feats[k] = chainer.cuda.to_cpu(rnnAudio.data)
                            else:
                                predicted_motion[k + 1] = out_step.data
                                feats[k] = rnnAudio.data
                            current_step = out_step

                    del audio
                    time = timeit.default_timer() - start
                    predicted_motion = (predicted_motion - config['intersec_pos']) / config['slope_pos']
                    predicted_motion = render_motion(predicted_motion, args.rot, scale=args.scale)
                    logging.info('Forwarding time was: {:.02f}s'.format(time))
                    # Evaluations
                    fscore, prec, recall, acc, best_init_beat, best_beat_entropy, best_step_entropy = metrics(
                        predicted_motion, music_beat, motion_beat_idx, dance_step)
                    results['filename'].append(filename)  # TODO(nelson): Separation by genre
                    results['noise'].append(noise_name)
                    results['snr'].append(str(snr))
                    results['fscore'].append(fscore)
                    results['precission'].append(prec)
                    results['recall'].append(recall)
                    results['acc'].append(acc)
                    results['forward_time'].append(time)
                    results['init_beat'].append(best_init_beat)
                    results['entropy_beat'].append(best_beat_entropy)
                    results['entropy_step'].append(best_step_entropy)

                    # Save extra data
                    save_file = os.path.join(args.folder, 'evaluation', '{}_{}_{}_{}_snr{}_ep{}_output.h5'.format(
                        args.stage, stage, filename, noise_name, str(snr), args.epoch))
                    if os.path.exists(save_file):
                        os.remove(save_file)
                    with h5py.File(save_file, 'w') as f:
                        f.create_dataset('motion', data=predicted_motion)
                        f.create_dataset('audiofeats', data=feats)

                    del feats
                    del predicted_motion

            # Save evaluations results
            df = pandas.DataFrame(results, columns=results_keys)
            result_file = os.path.join(args.folder, 'results',
                                       'result_{}_{}_ep{}.csv'.format(args.stage, stage, args.epoch))
            df.to_csv(result_file, encoding='utf-8')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion evaluation')
    parser.add_argument('--alignframe', '-a', type=int,
                        help='Frames allowed to align', default=0)
    parser.add_argument('--silence', '-c', type=int,
                        help='List of SNR', default=0)
    parser.add_argument('--epoch', type=int,
                        help='Trained epochs', default=0)
    parser.add_argument('--folder', '-d', type=str,
                        help='Specify network folder')
    parser.add_argument('--exp', '-e', type=str,
                        help='Experiment type')
    parser.add_argument('--encoder', '-E', type=str,
                        help='Encoder type')
    parser.add_argument('--freq', '-f', type=int,
                        help='Audio frequency sampling', default=16000)
    parser.add_argument('--gpu', '-g', type=int,
                        help='GPU id', default=0)
    parser.add_argument('--initOpt', '-i', nargs='+', type=int,
                        help='Model initial options')
    parser.add_argument('--list', '-l', type=str,
                        help='List Files to evaluate')
    parser.add_argument('--model', '-m', type=str,
                        help='Trained Model in HDF5')
    parser.add_argument('--network', '-n', type=str,
                        help='Network description in python format')
    parser.add_argument('--hop', '-o', type=int,
                        help='STFT Hop size', default=0)
    parser.add_argument('--fps', '-p', type=int,
                        help='Motion file FPS', default=0)
    parser.add_argument('--snr', '-r', nargs='+', type=int,
                        help='List of SNR')
    parser.add_argument('--scale', '-s', type=float,
                        help='Position Scale', default=1)
    parser.add_argument('--stage', '-S', type=str,
                        help='Evaluation stage')
    parser.add_argument('--rot', '-t', type=str,
                        help='Rotation type')
    parser.add_argument('--wlen', '-w', type=int,
                        help='STFT Window size', default=0)
    parser.add_argument('--audio_list', type=str,
                        help='List Files used for training')
    parser.add_argument('--step_file', type=str,
                        help='File with dance steps')
    parser.add_argument('--beats_skips', type=int,
                        help='Maximum value of music beats skipped for a dance step', default=0)
    parser.add_argument('--untrained', type=int,
                        help='Do untrained data if 1', default=0)
    parser.add_argument('--verbose', type=int,
                        help='Logging Verbose', default=0)
    args = parser.parse_args()
    config = Configuration(args)
    platform = platform.system()
    if platform == 'Windows':
        import subprocess
    elif platform == 'Linux':
        pass
    else:
        raise OSError('OS not supported')
    DATA_FOLDER = os.environ['DATA_EXTRACT']
    list_noises = ['Clean', 'white',
                   os.path.join(DATA_FOLDER, 'AUDIO', 'WAVE', 'NOISE', 'claps.wav'),
                   os.path.join(DATA_FOLDER, 'AUDIO', 'WAVE', 'NOISE', 'crowd.wav')]
    audio_list = glob.glob(os.path.join(DATA_FOLDER, 'AUDIO', 'MP3', '*.mp3'))
    snr_lst = args.snr
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np
    format = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    if args.verbose < 1:
        logging.basicConfig(
            level=logging.WARN, format=format)
        logging.warning('Skip DEBUG/INFO messages')
    else:
        logging.basicConfig(
            level=logging.INFO, format=format)
    main()
