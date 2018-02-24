@echo off
setlocal enableextensions enabledelayedexpansion
call "path.bat"

set net=s2s
set rot=quat
set exp=bounce
set stage=3

set epoch=10
set batch=50
set display_log=1000
set gpu=0
set workers=4
set sequence=150

set frame_align=3
set motion_align=35
set fps=30
set wlen=160
set hop=80
set frqsmp=16000
set silence=10
set scale=100.0
set featextract=CNNFeat
set init_step=0

set LSTM_units=500
set CNN_outs=65
set network=models\net_%net%.py

if "%rot%" == "quat" set Net_out=71
if "%rot%" == "euler" set Net_out=54

echo ============================================================
echo                         DeepDancer
echo ============================================================

set exp_name=%exp%_%net%_%rot%_initstep_%init_step%
set exp_folder=exp\%exp_name%
set exp_data=exp\data\%exp%_%rot%

echo ----- Exp: %exp_name%
if %stage% leq -1 (
  echo Data Download
  call "local\getdata.bat" 
)

set trn_lst=%exp_data%\annots\train.lst
set tst_lst=%exp_data%\annots\test.lst

if %stage% leq 0 ( 
    if not exist %exp_data%\annots md %exp_data%\annots 
    ( dir %DATA_EXTRACT%\MOCAP\HTR\%exp%*.htr /s/b /a-d )>%trn_lst%
    ( dir %DATA_EXTRACT%\MOCAP\HTR\test_%exp%*.htr /s/b /a-d )>%tst_lst%
    echo ----- Preparing training annotations.
    python local/annot_eval.py -l %trn_lst% -e %exp% -o %exp_data%\annots -m %motion_align% -a %frame_align% -f %fps% -s train 
    if %errorlevel% neq 0 exit /b %errorlevel%
    echo ----- Preparing test annotations.
    python local/annot_eval.py -l %tst_lst% -e %exp% -o %exp_data%\annots -m %motion_align% -a %frame_align% -f %fps% -s test
    if %errorlevel% neq 0 exit /b %errorlevel%
)

if %stage% leq 1 ( 
    echo ----- Preparing training data for motion ...
    if not exist %exp_data%\data md %exp_data%\data 
    if not exist %exp_data%\minmax md %exp_data%\minmax
    python local/data_preproc.py --type motion --exp %exp% --list %exp_data%\annots\train_files_align.txt ^
                        --save %exp_data% --rot %rot% --snr 0 --silence %silence% --fps %fps% ^
                        --hop %hop% --wlen %wlen% --scale %scale% 
    if %errorlevel% neq 0 exit /b %errorlevel%
    rem TODO: Add preparation for testing/validation during training (Need Larger dataset or to split in parts the whole sequence)
)

if %stage% leq 2 (
    echo Training Network
    python local/train_dance_rnn.py --folder %exp_data%\data --sequence %sequence%  ^
                        --batch %batch% --gpu %gpu% --epoch %epoch% --workers %workers% ^
                        --save %exp_folder%\trained\endtoend --network %network% ^
                        --encoder %featextract% --dataset DanceSeqHDF5 ^
                        --init_step %init_step% ^
                        --initOpt %LSTM_units% %CNN_outs% %Net_out% --frequency 5
    if %errorlevel% neq 0 exit /b %errorlevel%
)

if %stage% leq 3 (
    echo Evaluating Network
    if not exist %exp_folder%\evaluation mkdir %exp_folder%\evaluation 
    if not exist %exp_folder%\results mkdir %exp_folder%\results
    python local/evaluate.py --folder %exp_folder% --list %exp_data%\annots\test_files_align.txt ^
                --exp %exp% --rot %rot% --gpu %gpu% ^
                --network %network% --initOpt %LSTM_units% %CNN_outs% %Net_out% ^
                --fps %fps% --scale %scale% --model %exp_folder%\trained\endtoend\trained.model ^
                --snr 20 --freq %frqsmp% --hop %hop% --wlen %wlen% --encoder %featextract% ^
                --stage end2end --alignframe %frame_align%
    if %errorlevel% neq 0 exit /b %errorlevel%                
)

echo %~nx0 Done.

endlocal