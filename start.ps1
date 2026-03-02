# Enterprise RAG System - Startup Script
# Starts both FastAPI backend and Streamlit frontend

Write-Host "🚀 Starting Enterprise RAG System..." -ForegroundColor Cyan
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run setup.ps1 first to install dependencies." -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  Warning: .env file not found!" -ForegroundColor Yellow
    Write-Host "Create .env with your API keys (GROQ_API_KEY, etc.)" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "🔧 Starting services..." -ForegroundColor Cyan
Write-Host ""

# Start FastAPI backend in a new window
Write-Host "1️⃣  Starting FastAPI backend on http://localhost:8000" -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir'; .\venv\Scripts\Activate.ps1; python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"

# Wait a bit for backend to start
Write-Host "   Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start Streamlit frontend in a new window
Write-Host "2️⃣  Starting Streamlit frontend on http://localhost:8501" -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir'; .\venv\Scripts\Activate.ps1; streamlit run frontend/app.py"

Write-Host ""
Write-Host "✅ Both services started!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access the application:" -ForegroundColor Cyan
Write-Host "   • Frontend (Chat UI):  http://localhost:8501" -ForegroundColor White
Write-Host "   • Backend (API):       http://localhost:8000" -ForegroundColor White
Write-Host "   • API Docs:            http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "🛑 To stop: Close both PowerShell windows or press Ctrl+C in each" -ForegroundColor Yellow
Write-Host ""

# Open browser
Write-Host "🌐 Opening browser..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "Press any key to exit this launcher window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
