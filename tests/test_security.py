"""
Security Tests
Test security headers, input sanitization, and validation.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.middleware.security import (
    sanitize_string,
    sanitize_dict,
    validate_file_upload,
    check_content_safety
)

client = TestClient(app)


class TestSecurityHeaders:
    """Test security headers middleware."""
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses."""
        response = client.get("/")
        
        # Check for OWASP recommended headers
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-XSS-Protection" in response.headers
        
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Permissions-Policy" in response.headers
    
    def test_csp_header_content(self):
        """Test Content-Security-Policy header content."""
        response = client.get("/")
        csp = response.headers["Content-Security-Policy"]
        
        assert "default-src 'self'" in csp
        assert "script-src" in csp
        assert "style-src" in csp


class TestInputSanitization:
    """Test input sanitization functions."""
    
    def test_sanitize_simple_string(self):
        """Test sanitizing simple safe string."""
        result = sanitize_string("Hello World")
        assert result == "Hello World"
    
    def test_sanitize_html_escape(self):
        """Test HTML escaping."""
        result = sanitize_string("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result
        assert "<script>" not in result
    
    def test_sanitize_sql_injection(self):
        """Test SQL injection pattern removal."""
        inputs = [
            "test' UNION SELECT * FROM users--",
            "test'; DROP TABLE users--",
            "test' OR '1'='1"
        ]
        
        for inp in inputs:
            result = sanitize_string(inp)
            assert "UNION" not in result.upper()
            assert "DROP" not in result.upper()
    
    def test_sanitize_null_bytes(self):
        """Test null byte removal."""
        result = sanitize_string("test\x00string")
        assert "\x00" not in result
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {
            "name": "<script>alert('xss')</script>",
            "description": "Normal text",
            "nested": {
                "value": "test' UNION SELECT"
            }
        }
        
        result = sanitize_dict(data)
        
        assert "<script>" not in result["name"]
        assert "&lt;script&gt;" in result["name"]
        assert "UNION" not in result["nested"]["value"]
    
    def test_sanitize_list_in_dict(self):
        """Test sanitizing lists within dictionaries."""
        data = {
            "items": ["<b>test</b>", "normal", {"key": "<i>value</i>"}]
        }
        
        result = sanitize_dict(data)
        
        assert "&lt;b&gt;" in result["items"][0]
        assert "<b>" not in result["items"][0]


class TestFileUploadValidation:
    """Test file upload validation."""
    
    def test_validate_pdf_upload(self):
        """Test valid PDF upload."""
        is_valid = validate_file_upload(
            filename="document.pdf",
            content_type="application/pdf"
        )
        assert is_valid is True
    
    def test_validate_docx_upload(self):
        """Test valid DOCX upload."""
        is_valid = validate_file_upload(
            filename="document.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert is_valid is True
    
    def test_validate_txt_upload(self):
        """Test valid TXT upload."""
        is_valid = validate_file_upload(
            filename="document.txt",
            content_type="text/plain"
        )
        assert is_valid is True
    
    def test_invalid_extension(self):
        """Test invalid file extension."""
        with pytest.raises(ValueError, match="Invalid file type"):
            validate_file_upload(
                filename="malware.exe",
                content_type="application/x-msdownload"
            )
    
    def test_invalid_content_type(self):
        """Test invalid content type."""
        with pytest.raises(ValueError, match="Invalid content type"):
            validate_file_upload(
                filename="document.pdf",
                content_type="image/png"
            )
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        with pytest.raises(ValueError, match="path traversal"):
            validate_file_upload(
                filename="../../../etc/passwd",
                content_type="text/plain"
            )
        
        with pytest.raises(ValueError, match="path traversal"):
            validate_file_upload(
                filename="..\\..\\windows\\system32\\config",
                content_type="text/plain"
            )
    
    def test_null_byte_detection(self):
        """Test null byte detection in filename."""
        with pytest.raises(ValueError, match="null byte"):
            validate_file_upload(
                filename="test\x00.pdf",
                content_type="application/pdf"
            )


class TestContentSafety:
    """Test content safety checks."""
    
    def test_safe_content(self):
        """Test safe content detection."""
        result = check_content_safety("This is a normal text document about RAG systems.")
        
        assert result["safe"] is True
        assert len(result["detected_patterns"]) == 0
        assert result["confidence"] == 1.0
    
    def test_unsafe_javascript(self):
        """Test JavaScript detection."""
        result = check_content_safety("<script>alert('xss')</script>")
        
        assert result["safe"] is False
        assert len(result["detected_patterns"]) > 0
    
    def test_unsafe_event_handlers(self):
        """Test event handler detection."""
        unsafe_texts = [
            '<img onerror="alert(1)" src="x">',
            '<div onclick="malicious()">',
        ]
        
        for text in unsafe_texts:
            result = check_content_safety(text)
            assert result["safe"] is False
    
    def test_unsafe_eval_exec(self):
        """Test eval/exec detection."""
        unsafe_texts = [
            "eval('malicious code')",
            "exec('import os; os.system(\"rm -rf /\")')"
        ]
        
        for text in unsafe_texts:
            result = check_content_safety(text)
            assert result["safe"] is False


class TestAPISecurityIntegration:
    """Test security integration in API."""
    
    def test_cors_headers(self):
        """Test CORS configuration."""
        response = client.get("/")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or \
               "Access-Control-Allow-Origin" in response.headers
    
    def test_error_response_format(self):
        """Test standardized error response format."""
        response = client.get("/nonexistent-endpoint")
        
        # Should return 404 with standardized format
        assert response.status_code == 404
        data = response.json()
        
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
