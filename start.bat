@echo off
rem Double-click this file (or type .\start.bat in a terminal) to run Neurova on Windows.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto nopython

if exist ".venv\Scripts\python.exe" goto check
echo Setting up Neurova for the first time. This takes a minute or two...
python -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip

:check
rem Install packages when requirements.txt is new or has changed since the last install.
fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>nul
if not errorlevel 1 goto run
echo Installing the packages Neurova needs...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
copy /y requirements.txt ".venv\installed-requirements.txt" >nul

:run
".venv\Scripts\python.exe" run.py
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.10 to 3.13 from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run this file again.
pause
exit /b 1

:failed
echo.
echo Setup did not finish. Delete the .venv folder and run this file again.
echo If it still fails, copy the messages above and ask for help.
pause
exit /b 1
