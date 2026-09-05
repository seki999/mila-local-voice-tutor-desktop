@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
)

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Python 3 was not found. Install Python 3.11 or 3.12 first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Installation completed. Start LM Studio Local Server, then double-click run.bat.
pause
