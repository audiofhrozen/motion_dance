. ./path.sh

exp=salsa
net=s2s
feats=RES
init_step=1

declare -a hiphop_feats=("hiphop_drums"
                        "hiphop_drums_2"
                        "rock_drum_65"
                        "rock_drum_90"
                        "salsa_180_instr"
                        "salsa_200_instr")

declare -a salsa_feats=("salsa_150_congas"
                        "salsa_150_keys"
                        "salsa_150_var_instr"
                        "salsa_150_var_instr2"
                        "salsa_180_instr"
                        "salsa_200_instr")

declare -a seq_feats=("bachata_loop"
                        "bossa_nova_drum"
                        "hiphop_drums"
                        "hiphop_drums_2"
                        "rock_drum_65"
                        "rock_drum_90"
                        "salsa_150_congas"
                        "salsa_150_keys"
                        "salsa_150_var_instr"
                        "salsa_150_var_instr2"
                        "salsa_180_instr"
                        "salsa_200_instr")

for i in "${salsa_feats[@]}"
do
    wave="$DATA_EXTRACT/AUDIO/MP3/$i.mp3"
    epoch=5
    model="exp/"$exp"/"$net"_quat_"$feats"_initstep_$init_step/trained/endtoend/trained_$epoch.model" #15_epochs/trained_$epoch.model
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
        -s "exp/"$exp"/"$net"_quat_"$feats"_initstep_$init_step/untrained" \
        -r 1
    sleep 2
done
# For Bounce and Bounce2
# Trained:
#   hiphop_mashup
#   hiphop_instrumental_01 (bounce2 only)
# Untrained:
#   hiphop_01
#   fuma_el_barco
#   lej_summer_2015

# For Sequence:
# Trained:
#   bachata_02
#   hiphop_01
#   salsa_01
# Untrained:
#   lej_summer_2016
#   vivir_mi_vida
#   hiphop_instrumental_01
#   gotas_de_lluvia_live
#   por_retenerte_live

# For Salsa:
# Trained:
#   por_retenerte_live
#   un_monton_de_estrellas
# Untrained:
#   lej_summer_2016
#   fuma_el_barco
#   vivir_mi_vida

#lej_summer_2014 hiphop_01 hiphop_02 hiphop_instrumental_01
#salsa_01 bachata_03 lej_summer_2016 fuma_el_barco  nuestro_suenyo
#un_monton_de_estrellas devorame_otra_vez