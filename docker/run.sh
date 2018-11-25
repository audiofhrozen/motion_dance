#!/bin/bash

docker_gpu=0
docker_cuda=9.2
docker_cudnn=7

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h] docker_gpu docker_cuda docker_cudnn options"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h] ] docker_gpu docker_cuda docker_cudnn options"
              exit 0;;
        --docker*) ext=${1#--}
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
        --*) break
              ;;
    esac
    shift
    shift
done

if [ "${docker_gpu}" == "-1" ]; then
  from_image="ubuntu:16.04"
  image_label="dancer:ubuntu16.04"
else
  from_image="nvidia/cuda:${docker_cuda}-cudnn${docker_cudnn}-devel-ubuntu16.04"
  image_label="dancer:cuda${docker_cuda}-cudnn${docker_cudnn}-ubuntu16.04"
fi

image_label="${image_label}-user-${HOME##*/}"
cd ..
docker_image=$( docker images -q ${image_label} ) 
if ! [[ -n $docker_image  ]]; then
  echo "Building docker image..."
  
  build_args="--build-arg FROM_IMAGE=${from_image}"
  build_args="${build_args} --build-arg THIS_USER=${HOME##*/}"
  build_args="${build_args} --build-arg THIS_UID=${UID}"
  if [ ! -z "${HTTP_PROXY}" ]; then
    echo "Building with proxy ${HTTP_PROXY}"
    build_args="${build_args} --build-arg WITH_PROXY=${HTTP_PROXY}"
  fi 
  (docker build ${build_args} -f docker/Dockerfile -t ${image_label} .) || exit 1
fi

echo "Using image ${from_image}."
vol="-v $PWD:/motion_dance " # <=Folder to save data

cmd1="cd /motion_dance"
cmd2="./run.sh $@"

cmd="${cmd1}; ${cmd2}"
if [ "${docker_gpu}" == "-1" ]; then
  cmd="docker run -i --rm --name dancer_cpu ${vol} ${image_label} /bin/bash -c '${cmd}'"
else
  cmd="NV_GPU='${docker_gpu}' nvidia-docker run -i --rm --name dancer_gpu${docker_gpu} ${vol} ${image_label} /bin/bash -c '${cmd}'"
fi
echo "Executing application in Docker"
echo ${cmd}
eval ${cmd}

echo "`basename $0` done."
