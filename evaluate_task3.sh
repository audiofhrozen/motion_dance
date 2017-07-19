#!/bin/bash
FOLDER=
GPU=
image="chainer_nossh:2.0.1"

docker_image=$( docker images -q $image ) 
if ! [[ -n $docker_image  ]]; then
  echo "Building docker image..."
  docker build -f chainer_nossh.devel -t $image .
fi

vol1="/home/nelson/$FOLDER:/$FOLDER"
vol2="/home/nelson/DATA:/DATA"
vol3="/home/nelson/python_utils:/customlibs/python_utils"

cmd1="python /$FOLDER/evaluate_server.py $FOLDER"
cmd2="chmod -R 777 /$FOLDER/evaluation"

echo "Executing evaluation"
cmd="NV_GPU=$GPU nvidia-docker run -i --rm --name eval$GPU -v $vol1 -v $vol2 -v $vol3 $image /bin/bash -c '$cmd1; $cmd2'"
eval $cmd


