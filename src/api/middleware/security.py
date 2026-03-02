"""
Security Middleware
Security headers and input sanitization.
"""

from fastapi import Request
from fastapi.responses import Response
import html
import re
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


async def security_headers_middleware(request: Request, call_next):
    """
    Add security headers to all responses.
    
    Implements OWASP recommended security headers.
    """
    response = await call_next(request)
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Prevent MIME sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Enable XSS protection
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Strict Transport Security (HTTPS only in production)
    # Uncomment in production with HTTPS
    # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:; "
        "connect-src 'self'"
    )
    
    # Referrer Policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Permissions Policy
    response.headers["Permissions-Policy"] = (
        "geolocation=(), "
        "microphone=(), "
        "camera=(), "
        "payment=()"
    )
    
    return response


def sanitize_string(text: str, allow_html: bool = False) -> str:
    """
    Sanitize string input to prevent XSS and injection attacks.
    
    Args:
        text: Input text
        allow_html: If False, escape HTML characters
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Escape HTML if not allowed
    if not allow_html:
        text = html.escape(text)
    
    # Remove potential SQL injection patterns (basic)
    sql_patterns = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(--\s)",
        r"(/\*.*\*/)"
    ]
    
    for pattern in sql_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def sanitize_dict(data: Dict[str, Any], allow_html: bool = False) -> Dict[str, Any]:
    """
    Recursively sanitize dictionary values.
    
    Args:
        data: Input dictionary
        allow_html: If False, escape HTML in strings
        
    Returns:
        Sanitized dictionary
    """
    sanitized = {}
    
    for key, value in data.items():
        # Sanitize key
        clean_key = sanitize_string(str(key), allow_html=False)
        
        # Sanitize value
        if isinstance(value, str):
            sanitized[clean_key] = sanitize_string(value, allow_html=allow_html)
        elif isinstance(value, dict):
            sanitized[clean_key] = sanitize_dict(value, allow_html=allow_html)
        elif isinstance(value, list):
            sanitized[clean_key] = [
                sanitize_string(item, allow_html=allow_html) if isinstance(item, str)
                else sanitize_dict(item, allow_html=allow_html) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            sanitized[clean_key] = value
    
    return sanitized


def validate_file_upload(filename: str, content_type: str, max_size_mb: int = 10) -> bool:
    """
    Validate file upload.
    
    Args:
        filename: Name of uploaded file
        content_type: MIME type
        max_size_mb: Maximum file size in MB
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Allowed file extensions
    allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.doc', '.rtf'}
    
    # Check extension
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    if f'.{file_ext}' not in allowed_extensions:
        raise ValueError(
            f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check content type
    allowed_content_types = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain',
        'text/markdown',
        'application/rtf'
    }
    
    if content_type not in allowed_content_types:
        raise ValueError(f"Invalid content type: {content_type}")
    
    # Check for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise ValueError("Invalid filename: path traversal detected")
    
    # Check for null bytes
    if '\x00' in filename:
        raise ValueError("Invalid filename: null byte detected")
    
    return True


def check_content_safety(text: str) -> Dict[str, Any]:
    """
    Basic content safety check.
    
    Args:
        text: Text to check
        
    Returns:
        Safety check results
    """
    # Check for potential malicious patterns
    malicious_patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r'eval\(',
        r'exec\(',
        r'base64'
    ]
    
    detected = []
    for pattern in malicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(pattern)
    
    return {
        "safe": len(detected) == 0,
        "detected_patterns": detected,
        "confidence": 1.0 if len(detected) == 0 else 0.3
    }
