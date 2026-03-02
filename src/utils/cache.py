"""
Caching module for response caching and performance optimization.
Supports disk-based, Redis, and in-memory caching.
"""

import hashlib
import json
import pickle
from typing import Any, Optional, Callable
from functools import wraps
from pathlib import Path

import diskcache
from .config import get_settings
from .logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CacheManager:
    """Manager for caching responses and intermediate results."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.settings = get_settings()
        self.cache = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the appropriate cache backend."""
        if not self.settings.enable_cache:
            logger.info("Caching is disabled")
            return
        
        cache_type = self.settings.cache_type.lower()
        
        if cache_type == "disk":
            self._initialize_disk_cache()
        elif cache_type == "redis":
            self._initialize_redis_cache()
        elif cache_type == "memory":
            self._initialize_memory_cache()
        else:
            logger.warning(f"Unknown cache type: {cache_type}, falling back to disk")
            self._initialize_disk_cache()
    
    def _initialize_disk_cache(self):
        """Initialize disk-based cache using diskcache."""
        try:
            cache_dir = Path(self.settings.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.cache = diskcache.Cache(
                str(cache_dir),
                size_limit=1e9,  # 1GB limit
                eviction_policy='least-recently-used'
            )
            logger.info(f"Disk cache initialized at {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize disk cache: {e}")
            self.cache = None
    
    def _initialize_redis_cache(self):
        """Initialize Redis cache (requires redis-py)."""
        try:
            import redis
            
            if not self.settings.redis_url:
                logger.error("Redis URL not configured")
                self._initialize_disk_cache()
                return
            
            self.cache = redis.from_url(
                self.settings.redis_url,
                decode_responses=False
            )
            # Test connection
            self.cache.ping()
            logger.info("Redis cache initialized")
        except ImportError:
            logger.warning("Redis not installed, falling back to disk cache")
            self._initialize_disk_cache()
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._initialize_disk_cache()
    
    def _initialize_memory_cache(self):
        """Initialize in-memory cache using dict."""
        self.cache = {}
        logger.info("In-memory cache initialized")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create a string representation of arguments
        key_parts = [prefix]
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        # Create hash of the key
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.settings.enable_cache or self.cache is None:
            return None
        
        try:
            if isinstance(self.cache, dict):
                return self.cache.get(key)
            elif isinstance(self.cache, diskcache.Cache):
                return self.cache.get(key)
            else:  # Redis
                value = self.cache.get(key)
                if value:
                    return pickle.loads(value)
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        if not self.settings.enable_cache or self.cache is None:
            return
        
        ttl = ttl or self.settings.cache_ttl
        
        try:
            if isinstance(self.cache, dict):
                self.cache[key] = value
            elif isinstance(self.cache, diskcache.Cache):
                self.cache.set(key, value, expire=ttl)
            else:  # Redis
                serialized = pickle.dumps(value)
                self.cache.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache."""
        if not self.settings.enable_cache or self.cache is None:
            return
        
        try:
            if isinstance(self.cache, dict):
                self.cache.pop(key, None)
            elif isinstance(self.cache, diskcache.Cache):
                self.cache.delete(key)
            else:  # Redis
                self.cache.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        if not self.settings.enable_cache or self.cache is None:
            return
        
        try:
            if isinstance(self.cache, dict):
                self.cache.clear()
            elif isinstance(self.cache, diskcache.Cache):
                self.cache.clear()
            else:  # Redis
                self.cache.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.settings.enable_cache or self.cache is None:
            return {"enabled": False}
        
        try:
            if isinstance(self.cache, dict):
                return {
                    "enabled": True,
                    "type": "memory",
                    "size": len(self.cache)
                }
            elif isinstance(self.cache, diskcache.Cache):
                return {
                    "enabled": True,
                    "type": "disk",
                    "size": self.cache.volume(),
                    "count": len(self.cache)
                }
            else:  # Redis
                info = self.cache.info("stats")
                return {
                    "enabled": True,
                    "type": "redis",
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0)
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}


# Global cache instance
cache_manager = CacheManager()


def cached(prefix: str = "default", ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl)
            logger.debug(f"Cache miss for {func.__name__}, result cached")
            
            return result
        
        return wrapper
    return decorator


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance (alias for get_cache)."""
    return cache_manager


if __name__ == "__main__":
    # Test caching
    cache = get_cache()
    
    # Test basic operations
    cache.set("test_key", "test_value")
    print(f"Get test_key: {cache.get('test_key')}")
    
    # Test decorator
    @cached(prefix="test_func", ttl=60)
    def expensive_function(x: int) -> int:
        print(f"Computing {x}...")
        return x * x
    
    print(f"First call: {expensive_function(5)}")
    print(f"Second call (cached): {expensive_function(5)}")
    
    # Test stats
    print(f"Cache stats: {cache.get_stats()}")
