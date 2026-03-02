"""
Evaluation Module
Tools for evaluating RAG system performance.
"""

from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGASResult, RAGAS_AVAILABLE
from src.evaluation.custom_metrics import (
    CustomMetricsEvaluator,
    PerformanceMetrics,
    AggregatedMetrics
)
from src.evaluation.benchmark import Benchmark, BenchmarkQuery, BenchmarkResult

__version__ = "1.0.0"

__all__ = [
    "RAGASEvaluator",
    "RAGASResult",
    "RAGAS_AVAILABLE",
    "CustomMetricsEvaluator",
    "PerformanceMetrics",
    "AggregatedMetrics",
    "Benchmark",
    "BenchmarkQuery",
    "BenchmarkResult",
]
