"""
Custom Metrics Module
Custom evaluation metrics for RAG system performance.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single query."""
    query: str
    response_time: float
    retrieval_time: float
    generation_time: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    num_retrieved_docs: int
    num_relevant_docs: int
    retrieval_precision: float
    cost_usd: float


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple queries."""
    num_queries: int
    avg_response_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    avg_total_tokens: int
    avg_retrieval_precision: float
    total_cost_usd: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float


class CustomMetricsEvaluator:
    """
    Custom metrics evaluator for RAG system.
    
    Tracks:
    - Response times (total, retrieval, generation)
    - Token usage (prompt, completion, total)
    - Cost estimation
    - Retrieval accuracy (precision, recall)
    - System throughput
    """
    
    # Pricing per 1M tokens (adjust based on provider)
    PRICING = {
        "groq": {
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
            "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
        },
        "openai": {
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        }
    }
    
    def __init__(self, provider: str = "groq", model: str = "llama-3.3-70b-versatile"):
        """Initialize custom metrics evaluator.
        
        Args:
            provider: LLM provider (groq, openai)
            model: Model name
        """
        self.provider = provider
        self.model = model
        self.metrics_history: List[PerformanceMetrics] = []
        logger.info(f"Custom metrics evaluator initialized: {provider}/{model}")
    
    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Calculate cost in USD.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Cost in USD
        """
        pricing = self.PRICING.get(self.provider, {}).get(self.model)
        
        if not pricing:
            logger.warning(f"No pricing found for {self.provider}/{self.model}")
            return 0.0
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def calculate_retrieval_precision(
        self,
        retrieved_docs: List[Dict[str, Any]],
        relevance_threshold: float = 0.5
    ) -> float:
        """Calculate retrieval precision.
        
        Args:
            retrieved_docs: List of retrieved documents with similarity scores
            relevance_threshold: Minimum similarity to consider relevant
            
        Returns:
            Precision (relevant docs / retrieved docs)
        """
        if not retrieved_docs:
            return 0.0
        
        relevant_count = sum(
            1 for doc in retrieved_docs
            if doc.get("similarity", 0) >= relevance_threshold
        )
        
        return relevant_count / len(retrieved_docs)
    
    def record_query_metrics(
        self,
        query: str,
        response_time: float,
        retrieval_time: float,
        generation_time: float,
        total_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        retrieved_docs: List[Dict[str, Any]],
        relevance_threshold: float = 0.5
    ) -> PerformanceMetrics:
        """Record metrics for a single query.
        
        Args:
            query: User query
            response_time: Total response time
            retrieval_time: Time spent on retrieval
            generation_time: Time spent on generation
            total_tokens: Total tokens used
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            retrieved_docs: Retrieved documents with scores
            relevance_threshold: Relevance threshold for precision
            
        Returns:
            Performance metrics
        """
        num_retrieved = len(retrieved_docs)
        precision = self.calculate_retrieval_precision(retrieved_docs, relevance_threshold)
        num_relevant = int(num_retrieved * precision)
        cost = self.calculate_cost(prompt_tokens, completion_tokens)
        
        metrics = PerformanceMetrics(
            query=query,
            response_time=response_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            num_retrieved_docs=num_retrieved,
            num_relevant_docs=num_relevant,
            retrieval_precision=precision,
            cost_usd=cost
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """Get aggregated metrics across all queries.
        
        Returns:
            Aggregated metrics
        """
        if not self.metrics_history:
            raise ValueError("No metrics recorded yet")
        
        response_times = [m.response_time for m in self.metrics_history]
        retrieval_times = [m.retrieval_time for m in self.metrics_history]
        generation_times = [m.generation_time for m in self.metrics_history]
        total_tokens = [m.total_tokens for m in self.metrics_history]
        precisions = [m.retrieval_precision for m in self.metrics_history]
        costs = [m.cost_usd for m in self.metrics_history]
        
        # Sort for percentiles
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        
        return AggregatedMetrics(
            num_queries=len(self.metrics_history),
            avg_response_time=sum(response_times) / n,
            avg_retrieval_time=sum(retrieval_times) / n,
            avg_generation_time=sum(generation_times) / n,
            avg_total_tokens=int(sum(total_tokens) / n),
            avg_retrieval_precision=sum(precisions) / n,
            total_cost_usd=sum(costs),
            p95_response_time=sorted_times[int(n * 0.95)] if n > 0 else 0,
            p99_response_time=sorted_times[int(n * 0.99)] if n > 0 else 0,
            min_response_time=min(response_times),
            max_response_time=max(response_times)
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary as dictionary.
        
        Returns:
            Metrics summary
        """
        aggregated = self.get_aggregated_metrics()
        return {
            "aggregated": asdict(aggregated),
            "individual_queries": [asdict(m) for m in self.metrics_history]
        }
    
    def calculate_throughput(self, time_window_seconds: float = 60.0) -> float:
        """Calculate queries per second in recent time window.
        
        Args:
            time_window_seconds: Time window in seconds
            
        Returns:
            Queries per second
        """
        if not self.metrics_history:
            return 0.0
        
        # This is a simplified calculation
        # In production, you'd track actual timestamps
        avg_response_time = sum(m.response_time for m in self.metrics_history) / len(self.metrics_history)
        
        if avg_response_time == 0:
            return 0.0
        
        return 1.0 / avg_response_time
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown.
        
        Returns:
            Cost breakdown by component
        """
        total_prompt_tokens = sum(m.prompt_tokens for m in self.metrics_history)
        total_completion_tokens = sum(m.completion_tokens for m in self.metrics_history)
        
        pricing = self.PRICING.get(self.provider, {}).get(self.model, {})
        
        return {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "prompt_cost_usd": (total_prompt_tokens / 1_000_000) * pricing.get("input", 0),
            "completion_cost_usd": (total_completion_tokens / 1_000_000) * pricing.get("output", 0),
            "total_cost_usd": sum(m.cost_usd for m in self.metrics_history)
        }
    
    def reset_metrics(self):
        """Reset all recorded metrics."""
        self.metrics_history.clear()
        logger.info("Metrics history reset")


if __name__ == "__main__":
    # Test custom metrics
    print("Testing Custom Metrics Evaluator\n" + "="*60)
    
    evaluator = CustomMetricsEvaluator(provider="groq", model="llama-3.3-70b-versatile")
    
    # Simulate some queries
    test_data = [
        {
            "query": "What is machine learning?",
            "response_time": 1.2,
            "retrieval_time": 0.3,
            "generation_time": 0.9,
            "total_tokens": 500,
            "prompt_tokens": 300,
            "completion_tokens": 200,
            "retrieved_docs": [
                {"similarity": 0.8},
                {"similarity": 0.6},
                {"similarity": 0.4}
            ]
        },
        {
            "query": "Explain neural networks",
            "response_time": 1.5,
            "retrieval_time": 0.4,
            "generation_time": 1.1,
            "total_tokens": 600,
            "prompt_tokens": 350,
            "completion_tokens": 250,
            "retrieved_docs": [
                {"similarity": 0.9},
                {"similarity": 0.7},
                {"similarity": 0.5}
            ]
        }
    ]
    
    for data in test_data:
        metrics = evaluator.record_query_metrics(**data)
        print(f"\nQuery: {metrics.query}")
        print(f"  Response Time: {metrics.response_time:.2f}s")
        print(f"  Retrieval Precision: {metrics.retrieval_precision:.2%}")
        print(f"  Total Tokens: {metrics.total_tokens}")
        print(f"  Cost: ${metrics.cost_usd:.6f}")
    
    # Get aggregated metrics
    print("\n" + "="*60)
    print("Aggregated Metrics:")
    aggregated = evaluator.get_aggregated_metrics()
    print(f"  Queries: {aggregated.num_queries}")
    print(f"  Avg Response Time: {aggregated.avg_response_time:.2f}s")
    print(f"  Avg Retrieval Precision: {aggregated.avg_retrieval_precision:.2%}")
    print(f"  Total Cost: ${aggregated.total_cost_usd:.6f}")
    print(f"  P95 Response Time: {aggregated.p95_response_time:.2f}s")
    
    # Cost breakdown
    print("\n" + "="*60)
    print("Cost Breakdown:")
    costs = evaluator.get_cost_breakdown()
    print(f"  Total Tokens: {costs['total_tokens']:,}")
    print(f"  Prompt Cost: ${costs['prompt_cost_usd']:.6f}")
    print(f"  Completion Cost: ${costs['completion_cost_usd']:.6f}")
    print(f"  Total Cost: ${costs['total_cost_usd']:.6f}")
