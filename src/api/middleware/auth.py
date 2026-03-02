"""
Authentication Middleware
JWT-based authentication and API key validation for secure API access.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# Models
class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = []


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# In-memory user store (replace with database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "scopes": ["read", "write", "admin"]
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "disabled": False,
        "scopes": ["read"]
    }
}

# In-memory API key store (replace with database in production)
fake_api_keys_db = {
    "sk_test_key123": {
        "key": "sk_test_key123",
        "name": "Test API Key",
        "scopes": ["read", "write"],
        "created_at": "2026-03-02T00:00:00Z",
        "expires_at": None,
        "disabled": False
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {str(e)}")
        return None


def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Validate an API key."""
    if api_key in fake_api_keys_db:
        key_data = fake_api_keys_db[api_key]
        
        # Check if disabled
        if key_data.get("disabled", False):
            return None
        
        # Check expiration
        expires_at = key_data.get("expires_at")
        if expires_at:
            if datetime.fromisoformat(expires_at.replace('Z', '+00:00')) < datetime.now():
                return None
        
        return key_data
    
    return None


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    # Add scopes from token
    user.scopes = payload.get("scopes", [])
    
    return user


async def get_current_user_from_api_key(
    api_key: str = Security(api_key_header)
) -> Dict[str, Any]:
    """Get current user from API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    key_data = validate_api_key(api_key)
    
    if key_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return key_data


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_data: Optional[Dict[str, Any]] = Depends(get_current_user_from_api_key)
) -> User:
    """Get current user from either JWT or API key (flexible auth)."""
    # Try JWT first
    if token_user:
        return token_user
    
    # Try API key
    if api_key_data:
        # Convert API key data to User model
        return User(
            username=api_key_data.get("name", "api_key_user"),
            scopes=api_key_data.get("scopes", [])
        )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


def require_scope(required_scope: str):
    """Dependency to require a specific scope."""
    async def scope_checker(user: User = Depends(get_current_user)) -> User:
        if required_scope not in user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {required_scope}"
            )
        return user
    return scope_checker


# Optional authentication (for public endpoints with optional auth)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """Get current user optionally (doesn't raise if not authenticated)."""
    try:
        if credentials:
            return await get_current_user_from_token(credentials)
        elif api_key:
            key_data = await get_current_user_from_api_key(api_key)
            return User(
                username=key_data.get("name", "api_key_user"),
                scopes=key_data.get("scopes", [])
            )
    except HTTPException:
        pass
    
    return None
