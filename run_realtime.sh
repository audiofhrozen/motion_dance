. ./path.sh

exp=bounce
net=s2s
wave="$DATA_EXTRACT/AUDIO/MP3/hiphop_mashup.mp3"
model="exp/"$exp"_"$net"_quat/trained/endtoend/trained.model" #

feats=CNNFeat
python local/ue4_net.py  \
    -w $wave  \
    -m "models/net_$net.py"  \
    -t $model \
    -i 500 65 71  \
    -e $feats \
    -x "exp/"$exp"_"$net"_quat/minmax/"$exp"_pos_minmax_quat.h5" \
    -g 1 \
    -c "DNN"