#!/bin/bash

# Copyright 2017 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration

net="s2s"
rot="quat"
exp="sequence"
stage=0

train=1 # type experiment 

epoch=10
batch=50
display_log=1000
gpu=0
workers=4
sequence=150
init_step=0
feats="CNN"
dance_steps=1
untrained=1
verbose=0

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] gpu model rotation type"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] gpu model rotation type"
              exit 0;;
        --*) ext=${1#--}
              frombreak=true
              for i in _ {a..z} {A..Z}; do
                for var in `eval echo "\\${!$i@}"`; do
                  if [ "${var}" == "${ext}" ]; then
                    eval $ext=$2
                    frombreak=false
                    break 2
                  fi 
                done 
              done
              if $frombreak ; then
                echo "bad option $1" 
                exit 1
              fi
              ;;
        *) echo "argument $1 does not exit"
            exit 1;;
    esac
    shift
    shift
done
frame_align=3
motion_align=35
fps=30
wlen=256
hop=80
frqsmp=16000
silence=10
scale=100.0
featextract=${feats}Feat 

LSTM_units=500
CNN_outs=65
network="./models/net_${net}.py"

if [ rot=='quat' ]; then
  Net_out=71
elif [ rot=='euler' ]; then
  Net_out=54
fi

echo "============================================================"
echo "                        DeepDancer"
echo "============================================================"


. ./local/run_end${train}.sh || exit 1

echo "`basename $0` Done."