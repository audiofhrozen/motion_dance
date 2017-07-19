#!/bin/bash

folders=$( ls ../Task0003/training_files )
for i in $folders; do
  echo "$i"
  ./server.sh eval_3 0 133.9.8.196 $i n3ls0n21
done

#./server.sh eval_3 1 133.9.8.196 2017-07-13_1904 n3ls0n21
