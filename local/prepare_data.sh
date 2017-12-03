#!/bin/bash
out=$1
exp=$2
rot=$3
display=$4
exp_folder=$5
stage=1
N=4

annot(){
  oe=${1/$2/$3}
  oe=${oe/wav/txt}
  tools/beattracker.n 512 160 512 $1 > $oe
  oe=${1/$2/$4}
  oe=${oe/wav/h5}
  local/beattracker.py -i $1 -o $oe 
}

wav_folder=$DATA_EXTRACT/WAVE/SOX
out_folder=$DATA_EXTRACT/Annotations

mkdir -p $exp_folder

if [ $stage -le 0 ]; then 
  echo "--- Preparing data annotations HARK/MADMOM ..." 
  mkdir -p $out_folder/HARK $out_folder/MADMOM 2>/dev/null
  (
  for wavfile in $wav_folder/*.wav; do
    ((i=i%N)); ((i++==0)) && wait
    annot $wavfile $wav_folder $out_folder/HARK $out_folder/MADMOM &
  done
  wait
  )
fi

if [ $stage -le 1 ]; then 
  echo "--- Evaluating Annotations ..."
  local/annot_eval.py -f $DATA_EXTRACT/MOCAP/HTR -e $exp -d $display -o $exp_folder|| exit 1
fi

if [ $stage -le 2 ]; then 
  echo "--- Preparing training data ..."
  local/data_preproc.py --exp $exp --data $DATA_EXTRACT --out $out --rot $rot || exit 1
fi

echo "done"

