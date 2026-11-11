@echo off
setlocal
chcp 65001 >nul
title Needle Transformer - Check-in Demo
if defined NM_PYTHON goto explicit_python
if exist "%USERPROFILE%\.conda\envs\AI_stock_environment\python.exe" (
  set "NM_PYTHON=%USERPROFILE%\.conda\envs\AI_stock_environment\python.exe"
  goto explicit_python
)
where py >nul 2>nul
if not errorlevel 1 goto python_launcher
where python >nul 2>nul
if not errorlevel 1 goto python_path
echo Python was not found. Install dependencies and set NM_PYTHON to your interpreter.
set "NM_EXIT=1"
goto finish
:explicit_python
pushd "%~dp0.."
"%NM_PYTHON%" -X utf8 "%~dp0demo_checkin.py" %*
set "NM_EXIT=%ERRORLEVEL%"
popd
goto finish
:python_launcher
pushd "%~dp0.."
py -3 -X utf8 "%~dp0demo_checkin.py" %*
set "NM_EXIT=%ERRORLEVEL%"
popd
goto finish
:python_path
pushd "%~dp0.."
python -X utf8 "%~dp0demo_checkin.py" %*
set "NM_EXIT=%ERRORLEVEL%"
popd
:finish
echo.
if /I not "%~1"=="--no-pause" pause
exit /b %NM_EXIT%
