"""
Monitoring module for tracking system performance and LLM usage.
Integrates with LangSmith for tracing and provides custom metrics.
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from functools import wraps
from collections import defaultdict

from .config import get_settings
from .logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PerformanceMonitor:
    """Monitor system performance and track metrics."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = {}
        
    def record_metric(self, name: str, value: float, metadata: Optional[Dict] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Additional metadata
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "value": value
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        self.metrics[name].append(entry)
        logger.debug(f"Recorded metric {name}: {value}")
    
    def increment_counter(self, name: str, amount: int = 1):
        """
        Increment a counter.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        self.counters[name] += amount
    
    def start_timer(self, name: str):
        """
        Start a timer.
        
        Args:
            name: Timer name
        """
        self.timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and record duration.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds
        """
        if name not in self.timers:
            logger.warning(f"Timer {name} not found")
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.record_metric(f"duration_{name}", duration)
        del self.timers[name]
        
        return duration
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List]:
        """
        Get recorded metrics.
        
        Args:
            name: Metric name (optional, returns all if None)
            
        Returns:
            Dictionary of metrics
        """
        if name:
            return {name: self.metrics.get(name, [])}
        return dict(self.metrics)
    
    def get_counters(self) -> Dict[str, int]:
        """Get all counters."""
        return dict(self.counters)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "counters": self.get_counters(),
            "metrics": {}
        }
        
        for name, values in self.metrics.items():
            if values:
                numeric_values = [v["value"] for v in values]
                summary["metrics"][name] = {
                    "count": len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "avg": sum(numeric_values) / len(numeric_values),
                    "total": sum(numeric_values)
                }
        
        return summary
    
    def reset(self):
        """Reset all metrics and counters."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        logger.info("Performance monitor reset")


class LLMUsageTracker:
    """Track LLM API usage and costs."""
    
    def __init__(self):
        """Initialize the LLM usage tracker."""
        self.calls = []
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Cost per 1M tokens (approximate)
        self.cost_map = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "groq": 0.0,  # Free
            "llama": 0.0  # Free/Local
        }
    
    def track_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        metadata: Optional[Dict] = None
    ):
        """
        Track an LLM API call.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            metadata: Additional metadata
        """
        total_tokens = prompt_tokens + completion_tokens
        
        # Estimate cost
        cost_per_1m = self.cost_map.get(model, 0.01)  # Default cost
        cost = (total_tokens / 1_000_000) * cost_per_1m
        
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost
        }
        
        if metadata:
            call_record["metadata"] = metadata
        
        self.calls.append(call_record)
        self.total_tokens += total_tokens
        self.total_cost += cost
        
        logger.debug(f"LLM call tracked: {model}, {total_tokens} tokens, ${cost:.6f}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get usage summary.
        
        Returns:
            Usage summary dictionary
        """
        if not self.calls:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0
            }
        
        summary = {
            "total_calls": len(self.calls),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_tokens_per_call": self.total_tokens / len(self.calls),
            "models": defaultdict(int)
        }
        
        for call in self.calls:
            summary["models"][call["model"]] += 1
        
        summary["models"] = dict(summary["models"])
        
        return summary
    
    def reset(self):
        """Reset usage tracking."""
        self.calls.clear()
        self.total_tokens = 0
        self.total_cost = 0.0
        logger.info("LLM usage tracker reset")


# Global monitoring instances
performance_monitor = PerformanceMonitor()
llm_usage_tracker = LLMUsageTracker()


def monitor_performance(name: str):
    """
    Decorator to monitor function performance.
    
    Args:
        name: Metric name
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            performance_monitor.increment_counter(f"{name}_calls")
            
            try:
                result = await func(*args, **kwargs)
                performance_monitor.increment_counter(f"{name}_success")
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}_errors")
                raise e
            finally:
                duration = time.time() - start_time
                performance_monitor.record_metric(f"{name}_duration", duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            performance_monitor.increment_counter(f"{name}_calls")
            
            try:
                result = func(*args, **kwargs)
                performance_monitor.increment_counter(f"{name}_success")
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}_errors")
                raise e
            finally:
                duration = time.time() - start_time
                performance_monitor.record_metric(f"{name}_duration", duration)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


def get_llm_usage_tracker() -> LLMUsageTracker:
    """Get the global LLM usage tracker instance."""
    return llm_usage_tracker


def setup_langsmith():
    """Setup LangSmith tracing if configured."""
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(f"LangSmith tracing enabled for project: {settings.langchain_project}")
    else:
        logger.info("LangSmith tracing not configured")


# Initialize LangSmith on import
setup_langsmith()


if __name__ == "__main__":
    # Test monitoring
    monitor = get_performance_monitor()
    tracker = get_llm_usage_tracker()
    
    # Test performance monitoring
    @monitor_performance("test_function")
    def test_function(x: int) -> int:
        time.sleep(0.1)
        return x * 2
    
    for i in range(5):
        test_function(i)
    
    print("Performance Summary:")
    print(monitor.get_summary())
    
    # Test LLM tracking
    tracker.track_call("gpt-3.5-turbo", 100, 50)
    tracker.track_call("groq", 200, 100)
    
    print("\nLLM Usage Summary:")
    print(tracker.get_usage_summary())
