"""
Authentication Routes
Login, token refresh, and user management endpoints.
"""

from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from src.api.middleware import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_password_hash,
    User
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response Models
class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int


class UserCreate(BaseModel):
    """User registration model."""
    username: str
    password: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response model (without password)."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = []


class APIKeyCreate(BaseModel):
    """API key creation model."""
    name: str
    scopes: list[str] = ["read"]
    expires_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key response model."""
    key: str
    name: str
    scopes: list[str]
    created_at: str
    expires_at: Optional[str] = None


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token endpoint.
    
    Login with username and password to get an access token.
    """
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User {user.username} logged in successfully")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60
    }


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user.
    
    Note: In production, this should have additional validation and admin approval.
    """
    from src.api.middleware.auth import fake_users_db, get_user
    
    # Check if user already exists
    if get_user(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user (in production, save to database)
    new_user = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "disabled": False,
        "scopes": ["read"]  # Default scope
    }
    
    fake_users_db[user_data.username] = new_user
    
    logger.info(f"New user registered: {user_data.username}")
    
    return UserResponse(**{k: v for k, v in new_user.items() if k != "hashed_password"})


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return current_user


@router.post("/api-key", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new API key for the current user.
    
    Requires authentication.
    """
    from datetime import datetime
    import secrets
    from src.api.middleware.auth import fake_api_keys_db
    
    # Generate secure API key
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    
    # Calculate expiration
    expires_at = None
    if key_data.expires_days:
        from datetime import datetime, timedelta
        expires_at = (datetime.utcnow() + timedelta(days=key_data.expires_days)).isoformat() + 'Z'
    
    # Create API key entry
    key_entry = {
        "key": api_key,
        "name": key_data.name,
        "scopes": key_data.scopes,
        "created_at": datetime.utcnow().isoformat() + 'Z',
        "expires_at": expires_at,
        "disabled": False,
        "owner": current_user.username
    }
    
    fake_api_keys_db[api_key] = key_entry
    
    logger.info(f"API key created: {key_data.name} for user {current_user.username}")
    
    return APIKeyResponse(**key_entry)


@router.get("/health")
async def auth_health():
    """Authentication system health check."""
    from src.api.middleware.auth import fake_users_db, fake_api_keys_db
    
    return {
        "status": "healthy",
        "users_count": len(fake_users_db),
        "api_keys_count": len(fake_api_keys_db),
        "auth_method": "JWT + API Key"
    }
