"""
FastAPI Application
Main API entry point for the RAG system.
"""

import sys
import os

# Debug: Print environment info at startup
print("=" * 60, file=sys.stderr)
print("STARTUP DEBUG INFO:", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print(f"Current directory: {os.getcwd()}", file=sys.stderr)
print(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}", file=sys.stderr)
print(f"GROQ_API_KEY exists: {bool(os.environ.get('GROQ_API_KEY'))}", file=sys.stderr)
print("=" * 60, file=sys.stderr)

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import uvicorn
from pathlib import Path
import time

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.api.middleware import rate_limit_middleware, security_headers_middleware

logger = get_logger(__name__)
settings = get_settings()

# Debug: Log environment at startup
logger.info(f"=== STARTUP DEBUG ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}")
logger.info(f"GROQ_API_KEY exists: {bool(os.environ.get('GROQ_API_KEY'))}")
logger.info(f"Current directory: {os.getcwd()}")


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("Starting RAG API server...")
    
    # Startup: Initialize components
    try:
        # Ensure directories exist
        settings.ensure_directories()
        logger.info("Directories verified")
        
        # Log configuration
        logger.info(f"LLM Provider: {settings.default_llm_provider}")
        logger.info(f"Embedding Model: {settings.embedding_model}")
        logger.info(f"Vector DB: ChromaDB")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down RAG API server...")


# Create FastAPI app
logger.info("Creating FastAPI app...")
try:
    app = FastAPI(
        title="Enterprise RAG System API",
        description="RESTful API for document ingestion, semantic search, and RAG-based Q&A with security",
        version="2.0.0",
        lifespan=lifespan
    )
    logger.info("FastAPI app created successfully")
except Exception as e:
    logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
    raise


# CORS middleware (PRODUCTION-READY)
logger.info("Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit frontend
        "http://localhost:3000",  # React frontend (if any)
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
        # Add your production domains here
        # "https://yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
logger.info("CORS middleware added")


# Security middleware (must be added first)
@app.middleware("http")
async def security_middleware(request, call_next):
    """Apply security headers to all responses."""
    return await security_headers_middleware(request, call_next)

logger.info("Security middleware added")

# Rate limiting middleware
@app.middleware("http")
async def rate_limiting_middleware(request, call_next):
    """Apply rate limiting to API requests."""
    return await rate_limit_middleware(request, call_next)

logger.info("Rate limiting middleware added")

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

logger.info("Timing middleware added")

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "message": "Enterprise RAG System API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "documents": "/api/documents",
            "search": "/api/search",
            "chat": "/api/chat"
        }
    }


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        from src.embeddings import get_vector_store
        from src.llm import get_llm_client
        
        # Check vector store
        vector_store = get_vector_store()
        doc_count = vector_store.count()
        
        # Check LLM client
        llm_client = get_llm_client()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "vector_store": "operational",
                "llm_client": "operational",
                "document_count": doc_count
            },
            "version": "1.0.0"
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# Statistics endpoint
@app.get("/api/stats", tags=["Stats"])
async def get_statistics():
    """Get system statistics."""
    try:
        from src.embeddings import get_vector_store
        from src.llm import get_llm_client
        from src.utils.monitoring import get_performance_monitor
        
        vector_store = get_vector_store()
        llm_client = get_llm_client()
        perf_monitor = get_performance_monitor()
        
        return {
            "documents": {
                "total_indexed": vector_store.count(),
                "collection": vector_store.collection_name
            },
            "llm": llm_client.get_stats(),
            "performance": {
                "metrics": perf_monitor.get_metrics(),
                "counters": perf_monitor.get_counters()
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Import routers with error handling
logger.info("Importing routers...")
try:
    from src.api import documents, search, chat
    logger.info("Imported documents, search, chat routers")
    from src.api.routes import auth
    logger.info("Imported auth router")
except Exception as e:
    logger.error(f"Failed to import routers: {e}", exc_info=True)
    raise

# Include routers
logger.info("Adding routers to app...")
try:
    app.include_router(auth.router, prefix="/api")  # Auth routes
    logger.info("Added auth router")
    app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
    logger.info("Added documents router")
    app.include_router(search.router, prefix="/api/search", tags=["Search"])
    logger.info("Added search router")
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    logger.info("Added chat router")
except Exception as e:
    logger.error(f"Failed to add routers: {e}", exc_info=True)
    raise


# Error handlers (standardized error format)
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standardized format."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with standardized format."""
    logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "detail": str(exc) if settings.debug_mode else None,
                "status_code": 500,
                "path": str(request.url.path)
            }
        }
    )


logger.info("=== FastAPI app module loaded successfully ===")

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
