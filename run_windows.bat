@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   KeenForge - one-click launcher (Windows)
echo ============================================

REM --- locate a Python 3 interpreter ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python 3 was not found.
  echo Install Python 3.8+ from https://www.python.org/downloads/ and re-run.
  pause
  exit /b 1
)

REM --- create a local virtual environment on first run ---
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv || (echo [ERROR] failed to create venv & pause & exit /b 1)
)

call ".venv\Scripts\activate.bat"

REM --- install dependencies once (marker file) ---
if not exist ".venv\.kf_ready" (
  echo Upgrading pip ...
  python -m pip install --upgrade pip
  echo Installing dependencies ^(first run only; this can take several minutes^) ...
  python -m pip install -r requirements.txt || (echo [ERROR] dependency install failed & pause & exit /b 1)
  echo ready> ".venv\.kf_ready"
)

echo Launching KeenForge ...
python src\main.py
