# Sequential Learning for Dance generation

[![Build Status](https://travis-ci.com/Fhrozen/motion_dance.svg?branch=master)](https://travis-ci.com/Fhrozen/motion_dance)

Generating dance using deep learning techniques.

The proposed model is shown in the following image:

![Proposed Model](images/seq2seq_mc.png?raw=true "model")

The joints of the skeleton employed in the experiment are shown in the following image:

![Skeleton](images/skeleton.png?raw=true "skeleton")

## Unreal Engine 4 Visualization

You can obtain the UE4 environment from [here](https://github.com/audiofhrozen/dance-engine)
For demostration from evaluation files or for testing training files use (`local/ue4_send_osc.py`).
For realtime emulation execute (`run_realtime.sh`). 

## Installation

- If you intend to fully use this package, then see [Installation]()

## Execution

The main routine is executed by:
```sh
$ ./run.sh --net $net --exp $exp --sequence $sequence --epoch $epochs --stage $stage
```
Being possible to train different type of datasets (`$exp`)

To run into a docker container use the file (`run_in_docker.sh`) instead of (`run.sh`)

### Use of GPU
If you use GPU in your experiment, set `--gpu` option in `run.sh` appropriately, e.g., 
```sh
$ ./run.sh --gpu 0
```
Default setup uses GPU 0 (`--gpu 0`). For CPU execution set gpu to -1

## Requirements

For training and evaluating the following python libraries are required:
- [chainer=>3.1.0](https://github.com/chainer/chainer) 
- [chainerui](https://github.com/chainer/chainerui)
- [cupy=>2.1.0](https://github.com/cupy/cupy)
- [madmom](https://github.com/CPJKU/madmom/)
- [Beat Tracking Evaluation toolbox](https://github.com/Fhrozen/Beat-Tracking-Evaluation-Toolbox). The original code is found [here](https://github.com/adamstark/Beat-Tracking-Evaluation-Toolbox)
- [mir_eval](https://github.com/craffel/mir_eval)
- [transforms3d](https://github.com/matthew-brett/transforms3d)
- h5py, numpy, soundfile, scipy, scikit-learn, pandas

Install the following music libraries to convert the audio files:
```sh
$ sudo apt-get install libsox-fmt-mp3
```

Additionally, you may require [Marsyas](http://marsyas.info/doc/manual/marsyas-user/Building-latest-Marsyas-on-Debian_002fUbuntu.html#Building-latest-Marsyas-on-Debian_002fUbuntu) to extract the bet reference information.  

For real-time emulation:
- pyOSC (for python v2)
- python-osc (for python v3)
- vlc (optional)

## References

[1] Nelson Yalta, Shinji Watanabe, Kazuhiro Nakadai, Tetsuya Ogata, "Weakly Supervised Deep Recurrent Neural Networks for Basic Dance Step Generation", [arXiv](https://arxiv.org/abs/1807.01126)

[2] Nelson Yalta, Kazuhiro Nakadai, Tetsuya Ogata, "Sequential Deep Learning for Dancing Motion Generation", [SIG-Challenge 2016](http://www.osaka-kyoiku.ac.jp/~challeng/SIG-Challenge-046/SIG-Challenge-046-08.pdf)
