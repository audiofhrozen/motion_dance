#!/bin/bash

OPT=$1
GPU=$2
SERVER=$3
FOLDER=$4
PASS=$5
EXP=genre_salsa

SERVER_CMD="sshpass -p $PASS ssh -t nelson@$SERVER" 

if [ "$OPT" == "train" ]
then
  echo WIP
elif [ "$OPT" == "eval_3" ]
then
  if [ -d "../Task0003/training_files/$FOLDER/evaluation" ]
  then
    exit 1
  fi
  echo "Evaluation of Task0003"
  echo "Copying data into server"
  #sshpass -p $PASS rsync -rz /export/db01/Tasks/Task0003/DATA nelson@$SERVER:/home/nelson/
  #sshpass -p $PASS rsync -rz /home/nelson/Documents/python_utils nelson@$SERVER:/home/nelson/

  sshpass -p $PASS rsync ../Task0003/minmax/"$EXP"_pos_minmax.h5 nelson@$SERVER:/home/nelson/DATA/ 
  sshpass -p $PASS rsync -rz ../Task0003/training_files/$FOLDER nelson@$SERVER:/home/nelson/

  sshpass -p $PASS rsync ../Task0003/evaluate_server.py nelson@$SERVER:/home/nelson/$FOLDER/
  sshpass -p $PASS rsync ./chainer_nossh.devel nelson@$SERVER:/home/nelson/

  echo "Executing Docker File"
  sed "2s/.*/FOLDER=$FOLDER/" evaluate_task3.sh > evaluate_task3_"$GPU".sh
  sed -i "3s|.*|GPU=$GPU|" evaluate_task3_"$GPU".sh
  sshpass -p $PASS ssh -T nelson@$SERVER "bash -s" < evaluate_task3_"$GPU".sh
  
  echo "Copying files in localhost"
  sshpass -p $PASS rsync -rz nelson@$SERVER:/home/nelson/$FOLDER/evaluation ../Task0003/training_files/$FOLDER/
  sshpass -p $PASS ssh -T nelson@$SERVER "rm -r /home/nelson/$FOLDER"
  echo "Finish"
  exit 0
else
  echo Wrong option... Finishing Program
  exit 1
fi   

