#!/bin/bash
# Enterprise RAG System - Stop Script for Linux/Mac

echo "🛑 Stopping Enterprise RAG System..."
echo ""

# Check for PID files
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    echo "Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    rm .backend.pid
fi

if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    echo "Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null
    rm .frontend.pid
fi

# Fallback: kill by process name
echo "Cleaning up any remaining processes..."
pkill -f "uvicorn src.api.main:app"
pkill -f "streamlit run frontend/app.py"

echo ""
echo "✅ All services stopped!"
echo ""
