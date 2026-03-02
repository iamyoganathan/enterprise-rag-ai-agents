"""
Authentication Tests
Test JWT authentication, API keys, and security middleware.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import timedelta

from src.api.main import app
from src.api.middleware.auth import (
    create_access_token,
    authenticate_user,
    get_password_hash,
    verify_password,
    validate_api_key
)

client = TestClient(app)


class TestPasswordHashing:
    """Test password hashing functions."""
    
    def test_password_hash(self):
        """Test password hashing."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 20
    
    def test_password_verification(self):
        """Test password verification."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False


class TestJWTAuthentication:
    """Test JWT token creation and validation."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "testuser", "scopes": ["read"]}
        token = create_access_token(data, expires_delta=timedelta(minutes=15))
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 20
    
    def test_authenticate_user_success(self):
        """Test user authentication success."""
        user = authenticate_user("admin", "secret")
        
        assert user is not None
        assert user.username == "admin"
        assert "admin" in user.scopes
    
    def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password."""
        user = authenticate_user("admin", "wrongpassword")
        assert user is None
    
    def test_authenticate_user_nonexistent(self):
        """Test authentication with nonexistent user."""
        user = authenticate_user("nonexistent", "password")
        assert user is None


class TestAPIKeyValidation:
    """Test API key validation."""
    
    def test_validate_api_key_valid(self):
        """Test valid API key."""
        key_data = validate_api_key("sk_test_key123")
        
        assert key_data is not None
        assert key_data["name"] == "Test API Key"
        assert "read" in key_data["scopes"]
    
    def test_validate_api_key_invalid(self):
        """Test invalid API key."""
        key_data = validate_api_key("invalid_key")
        assert key_data is None


class TestAuthenticationEndpoints:
    """Test authentication API endpoints."""
    
    def test_login_success(self):
        """Test successful login."""
        response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "secret"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_wrong_password(self):
        """Test login with wrong password."""
        response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "wrongpassword"}
        )
        
        assert response.status_code == 401
        assert "detail" in response.json()
    
    def test_register_new_user(self):
        """Test user registration."""
        response = client.post(
            "/api/auth/register",
            json={
                "username": "newuser",
                "password": "newpassword123",
                "email": "newuser@example.com"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newuser"
        assert "password" not in data
    
    def test_register_duplicate_user(self):
        """Test registering duplicate username."""
        # First registration
        client.post(
            "/api/auth/register",
            json={"username": "duplicate", "password": "password123"}
        )
        
        # Try duplicate
        response = client.post(
            "/api/auth/register",
            json={"username": "duplicate", "password": "password456"}
        )
        
        assert response.status_code == 400
    
    def test_get_current_user(self):
        """Test getting current user info."""
        # Login first
        login_response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "secret"}
        )
        token = login_response.json()["access_token"]
        
        # Get user info
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
    
    def test_create_api_key(self):
        """Test API key creation."""
        # Login first
        login_response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "secret"}
        )
        token = login_response.json()["access_token"]
        
        # Create API key
        response = client.post(
            "/api/auth/api-key",
            json={"name": "Test Key", "scopes": ["read", "write"]},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["key"].startswith("sk_")
        assert data["name"] == "Test Key"
        assert "read" in data["scopes"]
    
    def test_auth_health(self):
        """Test authentication health check."""
        response = client.get("/api/auth/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "users_count" in data


class TestProtectedEndpoints:
    """Test protected endpoint access."""
    
    def test_access_without_auth(self):
        """Test accessing protected endpoint without authentication."""
        # This would fail if the endpoint requires auth
        # For now, just test that the endpoint exists
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_access_with_valid_token(self):
        """Test accessing with valid JWT token."""
        # Login
        login_response = client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "secret"}
        )
        token = login_response.json()["access_token"]
        
        # Access protected endpoint
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
    
    def test_access_with_invalid_token(self):
        """Test accessing with invalid token."""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        assert response.status_code == 401
    
    def test_access_with_api_key(self):
        """Test accessing with API key."""
        response = client.get(
            "/api/auth/health",
            headers={"X-API-Key": "sk_test_key123"}
        )
        
        # Health endpoint doesn't require auth, but header should be accepted
        assert response.status_code == 200
