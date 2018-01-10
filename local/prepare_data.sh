#!/bin/bash
exp="bpm"
rot="quat"
net="lstm"
frame_align=3
motion_align=15
fps=30

stage=2

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: `basename $0` [-h]"
            exit 0;;
        --help) echo "Usage: `basename $0` [-h]"
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

exp_name="$exp"_"$net"_"$rot"
exp_folder=./exp/$exp_name

if [ $stage -le 1 ]; then 
  mkdir -p $exp_folder/annots $exp_folder/data $exp_folder/minmax
  trn_lst=$exp_folder/annots/train.lst
  tst_lst=$exp_folder/annots/test.lst
  find $DATA_EXTRACT/MOCAP/HTR/ -name $exp'*.htr' | sort -u > $trn_lst
  find $DATA_EXTRACT/MOCAP/HTR/ -name 'test_'$exp'*.htr' | sort -u > $tst_lst
  echo "--- Preparing training annotations..."
  local/annot_eval.py -l $trn_lst -e $exp -o $exp_folder/annots -m $motion_align -a $frame_align -f $fps -s "train" || exit 1
  echo "--- Preparing test annotations..."
  local/annot_eval.py -l $tst_lst -e $exp -o $exp_folder/annots -m $motion_align -a $frame_align -f $fps -s "test" || exit 1
fi

if [ $stage -le 2 ]; then 
  echo "--- Preparing training data for motion ..."
  local/data_preproc.py --type motion --exp $exp --list $exp_folder/annots/train_files_align.txt \
                        --out $exp_folder --rot $rot --snr 10 || exit 1
  #TODO: Add preparation for testing/validation during training (Need Larger dataset or to split the parts of whole sequence)
fi



echo "`basename $0` Done."

