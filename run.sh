#!/bin/bash

# Copyright 2017 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration

net="lstm"
rot="euler"
exp="bpm"
graphs=false
stage=0
exp_name=$(date +"%Y-%m-%d_%H%M")
epochs=5
batch=50
display_log=1000

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
                  if [ "$var" == "$ext" ]; then
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

if [ $stage -le -1 ]; then
  echo "============================================================"
  echo "                       Data Download                        "
  echo "============================================================"
  local/getdata.sh 
fi

if [ $stage -le 0 ]; then
  echo "============================================================"
  echo "                       Data Preparation                     "
  echo "============================================================"
  out=$DATA_ROOT/training
  local/prepare_data.sh $out $exp $rot $graphs ./exp/training_files/$exp_name
fi

if [ $stage -le 1 ]; then
  echo "============================================================"
  echo "                       Training Network                     "
  echo "============================================================"
  bin/chainer_train.py t -D $exp_name -C ./conf/train_"$net"_"$rot"_"$exp".cfg || exit 1
fi

if [ $stage -le 2 ]; then
  echo "============================================================"
  echo "                       Evaluating Network                   "
  echo "============================================================"
  echo $exp_name
  local/evaluate.py --folder ./exp/training_files/$exp_name --data $DATA_ROOT/extracted --exp $exp --rot $rot
fi

echo "`basename $0` Done."