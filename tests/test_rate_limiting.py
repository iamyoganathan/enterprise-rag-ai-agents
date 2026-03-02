"""
Rate Limiting Tests
Test rate limiting middleware and token bucket algorithm.
"""

import pytest
import time
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.middleware.rate_limit import TokenBucket, RateLimiter

client = TestClient(app)


class TestTokenBucket:
    """Test token bucket algorithm."""
    
    def test_token_bucket_creation(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        assert bucket.capacity == 10
        assert bucket.refill_rate == 1.0
        assert bucket.tokens == 10
    
    def test_consume_tokens_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        assert bucket.consume(5) is True
        assert bucket.tokens == 5
    
    def test_consume_tokens_insufficient(self):
        """Test token consumption when insufficient."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        bucket.tokens = 3
        
        assert bucket.consume(5) is False
        assert bucket.tokens == 3  # Tokens not consumed
    
    def test_token_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens per second
        bucket.tokens = 0
        
        # Wait 1 second
        time.sleep(1.1)
        
        # Should have refilled ~2 tokens
        bucket._refill()
        assert bucket.tokens >= 2.0
        assert bucket.tokens <= 2.5
    
    def test_token_refill_cap(self):
        """Test that refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        bucket.tokens = 8
        
        # Wait to refill
        time.sleep(1.0)
        bucket._refill()
        
        # Should cap at 10
        assert bucket.tokens == 10
    
    def test_time_until_available(self):
        """Test calculating wait time for tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.tokens = 3
        
        # Need 5 tokens, have 3, deficit = 2
        # At 2 tokens/sec, need 1 second
        wait_time = bucket.time_until_available(5)
        
        assert wait_time == pytest.approx(1.0, abs=0.1)


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_rate_limiter_creation(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter()
        
        assert limiter.minute_buckets is not None
        assert limiter.hour_buckets is not None
    
    def test_rate_limit_within_limits(self):
        """Test requests within rate limits."""
        limiter = RateLimiter()
        
        allowed, info = limiter.check_rate_limit("test_client_1")
        
        assert allowed is True
        assert "minute_remaining" in info
        assert "hour_remaining" in info
    
    def test_rate_limit_exceeded_minute(self):
        """Test exceeding minute rate limit."""
        limiter = RateLimiter()
        identifier = "test_client_2"
        
        # Consume all minute tokens
        from src.utils.config import get_settings
        settings = get_settings()
        
        for _ in range(settings.rate_limit_per_minute):
            limiter.check_rate_limit(identifier)
        
        # Next request should be denied
        allowed, info = limiter.check_rate_limit(identifier)
        
        assert allowed is False
        assert info["limit_type"] == "per_minute"
        assert "retry_after" in info
    
    def test_rate_limit_different_clients(self):
        """Test that different clients have separate limits."""
        limiter = RateLimiter()
        
        # Client 1 makes requests
        allowed1, _ = limiter.check_rate_limit("client_1")
        
        # Client 2 should still have full quota
        allowed2, info2 = limiter.check_rate_limit("client_2")
        
        assert allowed1 is True
        assert allowed2 is True
        
        from src.utils.config import get_settings
        settings = get_settings()
        
        # Client 2 should have nearly full quota (minus 1 request)
        assert info2["minute_remaining"] == settings.rate_limit_per_minute - 1


class TestRateLimitMiddleware:
    """Test rate limit middleware in API."""
    
    def test_rate_limit_headers(self):
        """Test that rate limit headers are added."""
        response = client.get("/")
        
        assert "X-RateLimit-Limit-Minute" in response.headers
        assert "X-RateLimit-Remaining-Minute" in response.headers
        assert "X-RateLimit-Limit-Hour" in response.headers
        assert "X-RateLimit-Remaining-Hour" in response.headers
    
    def test_rate_limit_enforcement(self):
        """Test rate limit is enforced."""
        from src.utils.config import get_settings
        settings = get_settings()
        
        # Make many requests quickly
        responses = []
        for i in range(settings.rate_limit_per_minute + 5):
            response = client.get("/")
            responses.append(response)
        
        # Some should be rate limited (429)
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes
    
    def test_rate_limit_response_format(self):
        """Test rate limit error response format."""
        from src.utils.config import get_settings
        settings = get_settings()
        
        # Exceed rate limit
        for _ in range(settings.rate_limit_per_minute + 1):
            response = client.get("/")
        
        if response.status_code == 429:
            data = response.json()
            
            assert "error" in data
            assert "retry_after" in data
            assert "Retry-After" in response.headers
    
    def test_health_endpoint_not_rate_limited(self):
        """Test that health check bypass rate limiting."""
        from src.utils.config import get_settings
        settings = get_settings()
        
        # Make many health check requests
        for _ in range(settings.rate_limit_per_minute + 10):
            response = client.get("/health")
            assert response.status_code == 200  # Should always work
