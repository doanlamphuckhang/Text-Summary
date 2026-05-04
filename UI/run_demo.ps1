$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = "1"

Write-Host "Installing demo requirements..." -ForegroundColor Cyan
python -m pip install -r requirements_demo.txt

Write-Host ""
Write-Host "Starting ViT5 demo at http://127.0.0.1:7860" -ForegroundColor Green
python demo_app.py
