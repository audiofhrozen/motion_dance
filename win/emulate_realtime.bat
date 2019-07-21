@ECHO off 
call "path.bat"

SET exp=salsa
SET net=s2smc
SET feats=RES
REM SET wave=%DATA_EXTRACT%/AUDIO/WAVE/salsa_03.wav
SET wave=%DATA_EXTRACT%/AUDIO/MP3/salsa_150_var_instr.mp3
SET epochs=5
SET pretrained=exp/%net%_quat_%feats%_initstep_0/trained/endtoend/trained_%epochs%.model


cd ..
python deepdancer/bin/ue4_net.py ^
                        --track %wave% ^
                        --model deepdancer/models/net_%net%.py ^
                        --pretrained %pretrained% ^
                        --initOpt 500 65 71 ^
                        --encoder %feats%Feat ^
                        --minmax exp/data/%exp%_quat/minmax/pos_minmax.h5 ^
                        --gpu 0 ^
                        --character DNN ^
                        --host localhost
cd win
