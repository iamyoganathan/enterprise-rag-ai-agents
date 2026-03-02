#!/bin/bash
# Enterprise RAG System - Startup Script for Linux/Mac
# Starts both FastAPI backend and Streamlit frontend

echo "🚀 Starting Enterprise RAG System..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Create .env with your API keys (GROQ_API_KEY, etc.)"
    echo ""
fi

echo ""
echo "🔧 Starting services..."
echo ""

# Start FastAPI backend in background
echo "1️⃣  Starting FastAPI backend on http://localhost:8000"
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "   Waiting for backend to initialize..."
sleep 5

# Start Streamlit frontend in background
echo "2️⃣  Starting Streamlit frontend on http://localhost:8501"
streamlit run frontend/app.py > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ Both services started!"
echo ""
echo "📍 Access the application:"
echo "   • Frontend (Chat UI):  http://localhost:8501"
echo "   • Backend (API):       http://localhost:8000"
echo "   • API Docs:            http://localhost:8000/docs"
echo ""
echo "📋 Process IDs:"
echo "   • Backend:  $BACKEND_PID"
echo "   • Frontend: $FRONTEND_PID"
echo ""
echo "🛑 To stop, run:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "📄 Logs:"
echo "   • Backend:  logs/backend.log"
echo "   • Frontend: logs/frontend.log"
echo ""

# Save PIDs to file
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

# Open browser (Mac)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🌐 Opening browser..."
    sleep 3
    open http://localhost:8501
fi

# Open browser (Linux with xdg-open)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open &> /dev/null; then
        echo "🌐 Opening browser..."
        sleep 3
        xdg-open http://localhost:8501
    fi
fi
