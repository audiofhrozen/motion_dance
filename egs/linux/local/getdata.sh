#!/bin/bash
stage=0
DATA_SRC="https://onedrive.live.com/download?cid=076F0D2E8DDC9958&resid=76F0D2E8DDC9958%2112594&authkey=ANMZUSswyGhaLwo" 

command -v wget >/dev/null 2>&1 ||\
    { echo "\"wget\" is needed but not found"'!'; exit 1; }

DATA_TGZ=$DATA_ROOT/tgz

mkdir -p $DATA_TGZ $DATA_EXTRACT 2>/dev/null

if [ $stage -le 0 ]; then 
echo "--- Starting data download (may take some time) ..."
wget -O $DATA_TGZ/data.tar.gz -nc --no-check-certificate $DATA_SRC || \
    { echo "WGET error"'!' ; exit 1 ; }
fi

echo "--- Starting archives extraction ..."
tar  -C $DATA_EXTRACT -xzf $DATA_TGZ/data.tar.gz  || exit 1
echo "`basename $0` Done."
