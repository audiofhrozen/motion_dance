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
wlen=160
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

exp_name=${net}_${rot}_${feats}_initstep_${init_step}
exp_folder=./exp/${exp}/$exp_name
exp_data=./exp/data/${exp}_${rot}

echo "----- Exp: ${exp_name}"
if [ ${stage} -le -1 ]; then
  echo "Data Download"
  local/getdata.sh 
fi

trn_lst=${exp_data}/annots/train.lst
tst_lst=${exp_data}/annots/test.lst
if [ ${stage} -le 0 ]; then 
  mkdir -p ${exp_data}/annots 
  find ${DATA_EXTRAC}T/MOCAP/HTR/ -name ${exp}'_*.htr' | sort -u > ${trn_lst}
  steps_folder=${DATA_EXTRACT}/Annotations/steps

  echo "----- Preparing training annotations..."
  local/annot_eval.py -l ${trn_lst} \
                      -e ${exp} \
                      -o ${exp_data}/annots \
                      -m ${motion_align} \
                      -a ${frame_align} \
                      -f ${fps} \
                      -s "train"  \
                      --steps_folder ${steps_folder} \
                      --basic_steps ${dance_steps} \
                      --beats_range 8 \
                      --beats_skips 5 \
                      --verbose ${verbose}|| exit 1
fi
echo "----- End-to-End stage"

if [ ${stage} -le 1 ]; then 
  echo "----- Preparing training data for motion ..."
  mkdir -p ${exp_data}/data ${exp_data}/minmax
  local/data_prepare.py --type motion \
                        --exp ${exp} \
                        --list ${exp_data}/annots/train_files_align.txt \
                        --save ${exp_data} \
                        --rot ${rot} \
                        --snr 5 \
                        --silence ${silence} \
                        --fps ${fps} \
                        --hop ${hop} \
                        --wlen ${wlen} \
                        --scale ${scale} || exit 1
  #TODO: Add preparation for testing/validation during training (Need Larger dataset or to split in parts the whole sequence)
fi

if [ ${stage} -le 2 ]; then
  echo "Training Network "
  local/train_dance_rnn.py --folder ${exp_data}/data \
                        --sequence ${sequence}  \
                        --batch ${batch} \
                        --gpu ${gpu} \
                        --epoch ${epoch} \
                        --workers ${workers} \
                        --save ${exp_folder}/trained/endtoend \
                        --network ${network} \
                        --encoder ${featextract} \
                        --dataset "DanceSeqHDF5" \
                        --init_step ${init_step} \
                        --initOpt ${LSTM_units} ${CNN_outs} ${Net_out} \
                        --frequency 5 || exit 1
fi

if [ ${stage} -le 3 ]; then
  echo "Evaluating Network"
  find ${DATA_EXTRACT}/AUDIO/MP3 -name '*.mp3' | sort -u > ${tst_lst}
  mkdir -p ${exp_folder}/evaluation ${exp_folder}/results
  local/evaluate.py --folder ${exp_folder} \
                    --list ${exp_data}/annots/train_files_align.txt \
                    --beats_skips 16 \
                    --epoch ${epoch} \
                    --exp ${exp} \
                    --rot ${rot} \
                    --gpu ${gpu} \
                    --audio_list ${tst_lst} \
                    --network ${network} \
                    --initOpt ${LSTM_units} ${CNN_outs} ${Net_out} \
                    --fps ${fps} \
                    --scale ${scale} \
                    --model ${exp_folder}/trained/endtoend/trained_${epoch}.model \
                    --snr 20 10 0 \
                    --freq ${frqsmp} \
                    --hop ${hop} \
                    --wlen ${wlen} \
                    --encoder ${featextract} \
                    --stage "end2end" \
                    --alignframe ${frame_align} \
                    --step_file ${exp_data}/annots/train_basic_step.h5 \
                    --untrained ${untrained} || exit 1
fi

echo "`basename $0` Done."
