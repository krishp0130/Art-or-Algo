@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Project root: %CD%

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found. Install Python 3.10+ and ensure it is on PATH.
  exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo Creating virtual environment (.venv^) ...
  python -m venv .venv
  if errorlevel 1 exit /b 1
) else (
  echo Using existing .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Verifying data\train and data\val ...
if not exist "data\train\ai" goto badlayout
if not exist "data\train\human" goto badlayout
if not exist "data\val\ai" goto badlayout
if not exist "data\val\human" goto badlayout

echo OK: data layout found.
echo Environment ready. Activate with:  .venv\Scripts\activate.bat
exit /b 0

:badlayout
echo ERROR: Expected data\train\ai, data\train\human, data\val\ai, data\val\human
echo Run scripts\download_dataset.sh (or Kaggle download) then:  python scripts\split_train_eval.py
if exist "data\eval" if not exist "data\val" echo Hint: rename data\eval to data\val
exit /b 1
