. ./path.sh

exp=bounce #sequence
net=btscs2s
wave="$DATA_EXTRACT/AUDIO/MP3/lej_summer_2014.mp3"
model="exp/"$exp"_"$net"_quat/trained/endtoend/trained.model" #15_epochs/
#model="./exp/"$exp"_"$net"_quat/trained/endtoend/model_10120"
feats=CNNFeat
python local/ue4_net.py  \
    -w $wave  \
    -m "models/net_$net.py"  \
    -t $model \
    -i 500 65 71  \
    -e $feats \
    -x "exp/data/"$exp"_quat/minmax/pos_minmax.h5" \
    -g 1 \
    -c "DNN"