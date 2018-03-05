. ./path.sh

exp=genre_salsa
net=s2s
feats="RES"
#lej_summer_2014 hiphop_mashup hiphop_01 hiphop_02 hiphop_instrumental_01
#salsa_01 bachata_03 lej_summer_2016
wave="$DATA_EXTRACT/AUDIO/MP3/salsa_01.mp3"
init_step=1
model="exp/"$exp"_"$net"_quat_"$feats"_initstep_$init_step/trained/endtoend/trained.model" #15_epochs/
#model="./exp/"$exp"_"$net"_quat/trained/endtoend/model_10120"
featextract="$feats"Feat
python local/ue4_net.py  \
    -w $wave  \
    -m "models/net_$net.py"  \
    -t $model \
    -i 500 65 71  \
    -e $featextract \
    -x "exp/data/"$exp"_quat/minmax/pos_minmax.h5" \
    -g 1 \
    -c "DNN" \
    --host "192.168.170.110" \
    -r 1
