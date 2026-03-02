"""
RAG Pipeline Integration Tests
Test end-to-end RAG functionality.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert data["status"] == "healthy"
    
    def test_health_check_components(self):
        """Test health check includes all components."""
        response = client.get("/health")
        data = response.json()
        
        components = data["components"]
        
        assert "vector_store" in components
        assert "llm_client" in components
        assert "document_count" in components


class TestAPIEndpoints:
    """Test API endpoint availability."""
    
    def test_api_documentation(self):
        """Test OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestRateLimitedEndpoints:
    """Test that endpoints respect rate limiting."""
    
    def test_root_endpoint_rate_limits(self):
        """Test root endpoint includes rate limit headers."""
        response = client.get("/")
        
        assert "X-RateLimit-Limit-Minute" in response.headers
        assert "X-Process-Time" in response.headers


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_error_format(self):
        """Test 404 error response format."""
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
    
    def test_method_not_allowed(self):
        """Test 405 Method Not Allowed."""
        response = client.post("/health")
        
        assert response.status_code == 405


class TestProcessTimeMiddleware:
    """Test process time tracking."""
    
    def test_process_time_header(self):
        """Test that process time is tracked."""
        response = client.get("/")
        
        assert "X-Process-Time" in response.headers
        
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
        assert process_time < 10  # Should be fast
