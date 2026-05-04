@echo off
setlocal
cd /d "%~dp0"
set PYTHONDONTWRITEBYTECODE=1

echo Installing demo requirements...
python -m pip install -r requirements_demo.txt
if errorlevel 1 (
  echo.
  echo Failed to install requirements.
  pause
  exit /b 1
)

echo.
echo Starting ViT5 demo at http://127.0.0.1:7860
python demo_app.py
pause
