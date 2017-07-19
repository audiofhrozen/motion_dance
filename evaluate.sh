#!/bin/bash
FOLDER=2017-07-13_1904

docker build -f docker.devel -t chainer:2.0.1 .
"NV_GPU=0 nvidia-docker run -d -P --name eval1 chainer:2.0.1" 

