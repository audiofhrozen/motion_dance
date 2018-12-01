@echo off
setlocal enableextensions enabledelayedexpansion
call "path.bat"

set net=s2smc
set rot=quat
set exp=sequence
set stage=0
set feats=CNN
set init_step=1

set epoch=5
set batch=10
set display_log=1000
set gpu=0
set workers=4
set sequence=150
set basic_steps=1

set frame_align=3
set motion_align=35
set fps=30
set wlen=256
set hop=80
set frqsmp=16000
set silence=10
set scale=100.0
set featextract=%feats%Feat 
set verbose=1
set untrained=0

set LSTM_units=500
set CNN_outs=65
set network=models\net_%net%.py

rem Argument parser routine
:loop
if not "%1"=="" (
    set "test=%1"
    if /i "!test:~0,2!"=="--" (
        set var=!test:~2!
        if defined !var! (
            set "!var!=%2"
            shift
        )
    )
    shift 
    goto loop
)
rem end routine

if "%rot%" == "quat" set Net_out=71
if "%rot%" == "euler" set Net_out=54

echo ============================================================
echo                         DeepDancer
echo ============================================================

set exp_name=%net%_%rot%_%feats%_initstep_%init_step%
set exp_folder=exp\%exp%\%exp_name%
set exp_data=exp\data\%exp%_%rot%

cd ..
echo ----- Exp: %exp_name%
if %stage% leq -1 (
  echo stage -1: Data Download
  call "win\getdata.bat" 
)

set trn_lst=%exp_data%\annots\train.lst
set tst_lst=%exp_data%\annots\test.lst

if %stage% leq 0 ( 
    set steps_folder=%DATA_EXTRACT&%\Annotations\steps
    if not exist %exp_data%\annots md %exp_data%\annots 
    ( dir %DATA_EXTRACT%\MOCAP\HTR\%exp%_*.htr /s/b /a-d )>%trn_lst%
    echo stage 0: Preparing training annotations.
    python local/annot_eval.py -l %trn_lst% ^
                                -e %exp% ^
                                -o %exp_data%\annots ^
                                -m %motion_align% ^
                                -a %frame_align% ^
                                -f %fps% ^
                                -s train ^
                                --steps_folder $steps_folder ^
                                --basic_steps %basic_steps% ^
                                --beats_range 8 ^
                                --beats_skips 5 ^
                                --verbose %verbose%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

if %stage% leq 1 ( 
    echo stage 1: Preparing training data for motion ...
    if not exist %exp_data%\data md %exp_data%\data 
    if not exist %exp_data%\minmax md %exp_data%\minmax
    python local/data_prepare.py --type motion ^
                                --exp %exp% ^
                                --fps %fps% ^
                                --hop %hop% ^
                                --list %exp_data%\annots\train_files_align.txt ^
                                --save %exp_data% ^
                                --rot %rot% ^
                                --scale %scale% ^
                                --silence %silence% ^
                                --snr 0 ^
                                --wlen %wlen%  
    if !errorlevel! neq 0 exit /b !errorlevel!
    echo Done local/data_prepare.py
    rem TODO: Add preparation for testing/validation during training (Need Larger dataset or to split in parts the whole sequence)
)

if %stage% leq 2 (
    echo stage 2: Training Network
    python local/train_rnn.py --folder %exp_data%\data ^
                                    --sequence %sequence%  ^
                                    --batch %batch% ^
                                    --gpu %gpu% ^
                                    --epoch %epoch% ^
                                    --workers %workers% ^
                                    --save %exp_folder%\trained\endtoend ^
                                    --network %network% ^
                                    --encoder %featextract% ^
                                    --dataset DanceSeqHDF5 ^
                                    --init_step %init_step% ^
                                    --initOpt %LSTM_units% %CNN_outs% %Net_out% ^
                                    --frequency 1 ^
                                    --verbose %verbose%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

if %stage% leq 3 (
    echo stage 3: Evaluating Network
    ( dir %DATA_EXTRACT%\AUDIO\MP3\*.mp3 /s/b /a-d )>%tst_lst%
    if not exist %exp_folder%\evaluation md %exp_folder%\evaluation 
    if not exist %exp_folder%\results md %exp_folder%\results

    python local/evaluate.py --folder %exp_folder% ^
                            --list %exp_data%\annots\train_files_align.txt ^
                            --epoch %epoch% ^
                            --exp %exp% ^
                            --rot %rot% ^
                            --gpu %gpu% ^
                            --beats_skips 16 ^
                            --network %network% ^
                            --initOpt %LSTM_units% %CNN_outs% %Net_out% ^
                            --fps %fps% ^
                            --scale %scale% ^
                            --model %exp_folder%\trained\endtoend\trained_%epoch%.model ^
                            --snr 20 ^
                            --freq %frqsmp% ^
                            --hop %hop% ^
                            --wlen %wlen% ^
                            --encoder %featextract% ^
                            --stage end2end ^
                            --alignframe %frame_align% ^
                            --audio_list %tst_lst% ^
                            --step_file %exp_data%\annots\train_basic_step.h5 ^
                            --untrained %untrained% ^
                            --verbose %verbose%
    if !errorlevel! neq 0 exit /b !errorlevel!               
)

:eof
cd win
echo %~nx0 Done.
endlocal