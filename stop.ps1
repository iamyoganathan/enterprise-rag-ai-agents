# Enterprise RAG System - Stop Script
# Stops both FastAPI backend and Streamlit frontend

Write-Host "🛑 Stopping Enterprise RAG System..." -ForegroundColor Red
Write-Host ""

# Stop processes by name
Write-Host "Stopping Streamlit..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force

Write-Host "Stopping Uvicorn..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*uvicorn*"} | Stop-Process -Force

Write-Host "Stopping Python processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.CommandLine -like "*src.api.main*"} | Stop-Process -Force
Get-Process | Where-Object {$_.CommandLine -like "*frontend/app.py*"} | Stop-Process -Force

Write-Host ""
Write-Host "✅ All services stopped!" -ForegroundColor Green
Write-Host ""
