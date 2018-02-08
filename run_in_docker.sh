#!/bin/bash

helptext="gpu model rotation type -- execute all programs inside docker container"
gpu=0
net="s2s"
rot="quat"
exp="sequence"
stage=0
name="ch320_u1604_c90_cdnn7_nossh" 
image="chainer/nossh:3.2.0-cuda9.0-cudnn7-16.04" 

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] $helptext"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] $helptext"
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

docker_image=$( docker images -q $image ) 
if ! [[ -n $docker_image  ]]; then
  echo "Building docker image..."
  (docker build -t $image -< ./docker/"$name".devel) || exit 1
fi

vol1="$PWD/../../:/Tasks" # <=Folder to save data
#vol2="/home/nelson/Documents/libs_python:/libs_python"

cmd1="cd /Tasks/egs/motion_dance_v3"
cmd2="./run.sh --net $net --rot $rot --exp $exp --stage $stage"
cmd3="chmod -R 777 ./exp"

echo "Executing application in Docker container"
cmd="NV_GPU=$gpu nvidia-docker run -i --rm --name task3_$gpu -v $vol1  $image /bin/bash -c '$cmd1; $cmd2; $cmd3'" #<--rm erase the container at finishing the training, deleting all data (including training), should be remove to keep the container but for this configuration is recommended to use ssh
echo $cmd
#exit 0
eval $cmd

echo "`basename $0` Done."
