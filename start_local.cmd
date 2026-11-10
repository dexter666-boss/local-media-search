@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON=.venv\Scripts\python.exe"
set "MEDIA_SEARCH_OBJECT_QUALITY=fast"
if not exist ".venv-verifier\Scripts\python.exe" goto run
".venv-verifier\Scripts\python.exe" -c "import torch,open_clip,cv2; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 goto run
set "PROJECT_PYTHON=.venv-verifier\Scripts\python.exe"
set "MEDIA_SEARCH_OBJECT_QUALITY=accurate"
:run
if not exist "%PROJECT_PYTHON%" (
  echo Python environment is missing. Follow README.md to create .venv and install requirements.
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" launch.py
if errorlevel 1 pause
