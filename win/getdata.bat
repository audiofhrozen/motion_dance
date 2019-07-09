@echo off

set DATA_TGZ=%DATA_ROOT%\tgz
set DATA_URL="'https://onedrive.live.com/download?cid=076F0D2E8DDC9958&resid=76F0D2E8DDC9958%2112594&authkey=ANMZUSswyGhaLwo'"
               
if not exist %DATA_EXTRACT% md %DATA_EXTRACT%
if not exist %DATA_TGZ% md %DATA_TGZ%

echo --- Starting data download (may take some time) ...
if not exist %DATA_TGZ%\data.tar.gz powershell iwr -Uri %DATA_URL% -OutFile "%DATA_TGZ%\data.tar.gz"

tar  -C %DATA_EXTRACT% -xzf %DATA_TGZ%\data.tar.gz
if !errorlevel! neq 0 exit /b !errorlevel!

echo %~nx0 Done.