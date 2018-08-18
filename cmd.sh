#!/bin/bash

# Copyright 2017 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

if ! [ -f local/run.pl ]; then
    # run.pl from kaldi src to execute programs
    wget https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/wsj/s5/utils/parallel/run.pl -O local/run.pl
fi

export train_cmd="run.pl --mem 2G"
export decode_cmd="run.pl --mem 2G"
export cuda_cmd="run.pl --gpu 1"