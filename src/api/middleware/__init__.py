"""
Middleware Package
Authentication, rate limiting, and security middleware.
"""

from .auth import (
    get_current_user,
    get_current_user_optional,
    require_scope,
    create_access_token,
    authenticate_user,
    get_password_hash,
    User,
    TokenData
)
from .rate_limit import rate_limit_middleware, rate_limiter
from .security import (
    security_headers_middleware,
    sanitize_string,
    sanitize_dict,
    validate_file_upload,
    check_content_safety
)

__all__ = [
    # Auth
    "get_current_user",
    "get_current_user_optional",
    "require_scope",
    "create_access_token",
    "authenticate_user",
    "get_password_hash",
    "User",
    "TokenData",
    # Rate Limiting
    "rate_limit_middleware",
    "rate_limiter",
    # Security
    "security_headers_middleware",
    "sanitize_string",
    "sanitize_dict",
    "validate_file_upload",
    "check_content_safety"
]
