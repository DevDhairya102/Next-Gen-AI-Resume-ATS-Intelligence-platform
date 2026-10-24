@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Resume NLP Lab - setup and launch
echo ==========================================

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.10 or 3.11 and enable "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 goto :error
)

echo Installing / updating dependencies. This may take several minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Starting Resume NLP Lab in your browser...
echo The first semantic analysis may download the language model and take longer.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Setup or launch failed. Read the error above and check the README troubleshooting notes.
pause
exit /b 1
