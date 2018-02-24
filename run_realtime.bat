@ECHO off 
call "path.bat"

SET exp=bounce
SET net=btscs2s
SET wave=%DATA_EXTRACT%/AUDIO/MP3/lej_summer_2014.mp3
SET pretrained=exp/%exp%_%net%_quat/trained/endtoend/trained.model
SET feats=CNNFeat

python local/ue4_net.py ^
-w %wave% ^
-m models/net_%net%.py ^
-t %pretrained% ^
-i 500 65 71 ^
-e %feats% ^
-x exp/data/%exp%_quat/minmax/pos_minmax.h5 ^
-g 0
-c DNN
