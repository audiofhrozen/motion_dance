#!/bin/bash

# Copyright 2017 Waseda University (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

MAIN_ROOT=$PWD

if [ -e $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $MAIN_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $MAIN_ROOT/tools/venv/bin/activate
fi

export DATA_ROOT=$PWD/data
export DATA_EXTRACT=$DATA_ROOT

export PATH=${MAIN_ROOT}/local:${MAIN_ROOT}/deepdancer/bin:$PATH