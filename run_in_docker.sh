#!/bin/bash

helptext="gpu model rotation type -- execute all programs inside docker container"
gpu=0
net="lstm"
rot="euler"
exp="bpm"
graphs=false
stage=0
name="ch121_u1404_c75_cdnn5_nossh" #"ch300_u1604_c90_cdnn7_nossh" #chainer_cudnn7.5_nossh
image="chainer/nossh:1.21.0-cuda7.5-cudnn5-14.04" #"chainer/nossh:3.0.0-cuda9.0-cudnn7-16.04"

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
vol2="/home/nelson/Documents/libs_python:/libs_python"

cmd1="cd /Tasks/egs/motion_dance"
cmd2="./run.sh --net $net --rot $rot --exp $exp --graphs $graphs --stage $stage"
cmd3="chmod -R 777 ./exp/training_files"

echo "Executing evaluation"
cmd="NV_GPU=$gpu nvidia-docker run -i --rm --name task3_$gpu -v $vol1 -v $vol2 $image /bin/bash -c '$cmd1; $cmd2; $cmd3'" #<--rm erase the container at finishing the training, deleting all data (including training), should be remove to keep the container but for this configuration is recommended to use ssh
#echo $cmd
eval $cmd

echo "`basename $0` Done."
