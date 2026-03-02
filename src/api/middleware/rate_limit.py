"""
Rate Limiting Middleware
Token bucket algorithm for API rate limiting.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TokenBucket:
    """Token bucket for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate seconds until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        deficit = tokens - self.tokens
        return deficit / self.refill_rate


class RateLimiter:
    """Rate limiter with token bucket algorithm."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.minute_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=settings.rate_limit_per_minute,
                refill_rate=settings.rate_limit_per_minute / 60.0  # tokens per second
            )
        )
        self.hour_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=settings.rate_limit_per_hour,
                refill_rate=settings.rate_limit_per_hour / 3600.0  # tokens per second
            )
        )
        self.last_cleanup = time.time()
    
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier (IP, user ID, API key)
            
        Returns:
            (allowed, rate_limit_info)
        """
        # Cleanup old buckets periodically
        self._cleanup()
        
        # Check minute limit
        minute_bucket = self.minute_buckets[identifier]
        if not minute_bucket.consume():
            wait_time = minute_bucket.time_until_available()
            return False, {
                "limit_type": "per_minute",
                "limit": settings.rate_limit_per_minute,
                "remaining": int(minute_bucket.tokens),
                "retry_after": wait_time
            }
        
        # Check hour limit
        hour_bucket = self.hour_buckets[identifier]
        if not hour_bucket.consume():
            # Refund the minute bucket token since we didn't allow the request
            minute_bucket.tokens = min(minute_bucket.capacity, minute_bucket.tokens + 1)
            
            wait_time = hour_bucket.time_until_available()
            return False, {
                "limit_type": "per_hour",
                "limit": settings.rate_limit_per_hour,
                "remaining": int(hour_bucket.tokens),
                "retry_after": wait_time
            }
        
        # Request allowed
        return True, {
            "minute_remaining": int(minute_bucket.tokens),
            "hour_remaining": int(hour_bucket.tokens),
            "minute_limit": settings.rate_limit_per_minute,
            "hour_limit": settings.rate_limit_per_hour
        }
    
    def _cleanup(self):
        """Clean up old buckets."""
        now = time.time()
        
        # Cleanup every hour
        if now - self.last_cleanup > 3600:
            # Remove buckets that are full (inactive users)
            self.minute_buckets = defaultdict(
                lambda: TokenBucket(
                    capacity=settings.rate_limit_per_minute,
                    refill_rate=settings.rate_limit_per_minute / 60.0
                ),
                {k: v for k, v in self.minute_buckets.items() 
                 if v.tokens < v.capacity * 0.9}
            )
            
            self.hour_buckets = defaultdict(
                lambda: TokenBucket(
                    capacity=settings.rate_limit_per_hour,
                    refill_rate=settings.rate_limit_per_hour / 3600.0
                ),
                {k: v for k, v in self.hour_buckets.items() 
                 if v.tokens < v.capacity * 0.9}
            )
            
            self.last_cleanup = now
            logger.info("Rate limiter cleanup completed")


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_identifier(request: Request) -> str:
    """
    Get unique identifier for client.
    
    Priority: API Key > User ID > IP Address
    """
    # Try to get API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"
    
    # Try to get user from JWT (if available)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # Extract username from token (simplified)
        # In production, decode the JWT properly
        return f"user:{auth_header[7:20]}"  # Use part of token as identifier
    
    # Fall back to IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    
    return f"ip:{ip}"


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware.
    
    Applies rate limits based on client identifier.
    """
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get client identifier
    identifier = get_client_identifier(request)
    
    # Check rate limit
    allowed, rate_info = rate_limiter.check_rate_limit(identifier)
    
    if not allowed:
        logger.warning(
            f"Rate limit exceeded for {identifier} - "
            f"{rate_info['limit_type']}: {rate_info['limit']}"
        )
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded: {rate_info['limit']} requests {rate_info['limit_type']}",
                "limit": rate_info["limit"],
                "remaining": rate_info["remaining"],
                "retry_after": int(rate_info["retry_after"]) + 1
            },
            headers={
                "Retry-After": str(int(rate_info["retry_after"]) + 1),
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"])
            }
        )
    
    # Add rate limit headers to response
    response = await call_next(request)
    
    response.headers["X-RateLimit-Limit-Minute"] = str(rate_info["minute_limit"])
    response.headers["X-RateLimit-Remaining-Minute"] = str(rate_info["minute_remaining"])
    response.headers["X-RateLimit-Limit-Hour"] = str(rate_info["hour_limit"])
    response.headers["X-RateLimit-Remaining-Hour"] = str(rate_info["hour_remaining"])
    
    return response
