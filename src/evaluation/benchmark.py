"""
Benchmark Suite Module
Automated benchmark testing for RAG system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import time
from pathlib import Path

from src.llm.rag_chain import RAGChain
from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGAS_AVAILABLE
from src.evaluation.custom_metrics import CustomMetricsEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkQuery:
    """Single benchmark query."""
    id: str
    question: str
    expected_answer: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Benchmark result for a single query."""
    query_id: str
    question: str
    answer: str
    response_time: float
    retrieval_time: float
    generation_time: float
    num_sources: int
    sources: List[Dict[str, Any]]
    ragas_scores: Optional[Dict[str, float]] = None
    custom_metrics: Optional[Dict[str, Any]] = None


class Benchmark:
    """
    Automated benchmark suite for RAG system.
    
    Features:
    - Load test queries from JSON
    - Run queries through RAG pipeline
    - Collect RAGAS and custom metrics
    - Generate benchmark report
    - Compare against baseline
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        use_ragas: bool = True
    ):
        """Initialize benchmark suite.
        
        Args:
            collection_name: Vector collection name
            use_ragas: Whether to use RAGAS evaluation
        """
        self.collection_name = collection_name
        self.use_ragas = use_ragas and RAGAS_AVAILABLE
        
        # Initialize evaluators
        if self.use_ragas:
            self.ragas_evaluator = RAGASEvaluator()
        
        self.custom_evaluator = CustomMetricsEvaluator()
        
        logger.info(f"Benchmark initialized: collection={collection_name}, ragas={self.use_ragas}")
    
    def load_queries(self, filepath: Path) -> List[BenchmarkQuery]:
        """Load benchmark queries from JSON file.
        
        Args:
            filepath: Path to queries JSON file
            
        Returns:
            List of benchmark queries
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        queries = [
            BenchmarkQuery(
                id=q.get('id', str(i)),
                question=q['question'],
                expected_answer=q.get('expected_answer'),
                category=q.get('category'),
                difficulty=q.get('difficulty')
            )
            for i, q in enumerate(data.get('queries', []))
        ]
        
        logger.info(f"Loaded {len(queries)} benchmark queries from {filepath}")
        return queries
    
    def run_single_query(
        self,
        rag: RAGChain,
        query: BenchmarkQuery
    ) -> BenchmarkResult:
        """Run a single benchmark query.
        
        Args:
            rag: RAG chain instance
            query: Benchmark query
            
        Returns:
            Benchmark result
        """
        logger.info(f"Running query: {query.id} - {query.question[:50]}...")
        
        # Run query
        start_time = time.time()
        response = rag.query(query.question)
        response_time = time.time() - start_time
        
        # Extract metrics
        result = BenchmarkResult(
            query_id=query.id,
            question=query.question,
            answer=response.answer,
            response_time=response_time,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            num_sources=len(response.sources),
            sources=response.sources
        )
        
        # RAGAS evaluation
        if self.use_ragas and query.expected_answer:
            try:
                contexts = [s.get('content', '') for s in response.sources]
                ragas_result = self.ragas_evaluator.evaluate_single(
                    question=query.question,
                    answer=response.answer,
                    contexts=contexts,
                    ground_truth=query.expected_answer
                )
                result.ragas_scores = asdict(ragas_result)
            except Exception as e:
                logger.error(f"RAGAS evaluation failed for query {query.id}: {str(e)}")
        
        # Custom metrics
        retrieved_docs = [
            {"similarity": s.get('similarity', 0)}
            for s in response.sources
        ]
        
        custom_metrics = self.custom_evaluator.record_query_metrics(
            query=query.question,
            response_time=response_time,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time,
            total_tokens=response.tokens_used.get('total', 0),
            prompt_tokens=response.tokens_used.get('prompt', 0),
            completion_tokens=response.tokens_used.get('completion', 0),
            retrieved_docs=retrieved_docs
        )
        
        result.custom_metrics = asdict(custom_metrics)
        
        return result
    
    def run_benchmark(
        self,
        queries: List[BenchmarkQuery],
        save_results: bool = True,
        output_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run full benchmark suite.
        
        Args:
            queries: List of benchmark queries
            save_results: Whether to save results to file
            output_file: Output file path (optional)
            
        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark with {len(queries)} queries")
        
        # Initialize RAG chain
        rag = RAGChain(collection_name=self.collection_name)
        
        # Run queries
        results: List[BenchmarkResult] = []
        for query in queries:
            try:
                result = self.run_single_query(rag, query)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run query {query.id}: {str(e)}")
        
        # Aggregate metrics
        aggregated_custom = self.custom_evaluator.get_aggregated_metrics()
        
        benchmark_report = {
            "metadata": {
                "total_queries": len(queries),
                "successful_queries": len(results),
                "failed_queries": len(queries) - len(results),
                "collection": self.collection_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": [asdict(r) for r in results],
            "aggregated_metrics": {
                "custom": asdict(aggregated_custom),
            }
        }
        
        # Add RAGAS aggregated scores if available
        if self.use_ragas and any(r.ragas_scores for r in results):
            ragas_scores = [r.ragas_scores for r in results if r.ragas_scores]
            if ragas_scores:
                benchmark_report["aggregated_metrics"]["ragas"] = {
                    "avg_faithfulness": sum(s['faithfulness'] for s in ragas_scores) / len(ragas_scores),
                    "avg_answer_relevancy": sum(s['answer_relevancy'] for s in ragas_scores) / len(ragas_scores),
                    "avg_context_precision": sum(s['context_precision'] for s in ragas_scores) / len(ragas_scores),
                    "avg_context_recall": sum(s['context_recall'] for s in ragas_scores) / len(ragas_scores),
                    "avg_overall_score": sum(s['overall_score'] for s in ragas_scores) / len(ragas_scores),
                }
        
        # Cost breakdown
        benchmark_report["cost_breakdown"] = self.custom_evaluator.get_cost_breakdown()
        
        logger.info(
            f"Benchmark completed: {len(results)}/{len(queries)} successful, "
            f"avg_time={aggregated_custom.avg_response_time:.2f}s"
        )
        
        # Save results
        if save_results:
            if output_file is None:
                output_file = Path("data/benchmark_results.json")
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Benchmark results saved to {output_file}")
        
        return benchmark_report
    
    def compare_with_baseline(
        self,
        current_results: Dict[str, Any],
        baseline_file: Path
    ) -> Dict[str, Any]:
        """Compare current results with baseline.
        
        Args:
            current_results: Current benchmark results
            baseline_file: Path to baseline results JSON
            
        Returns:
            Comparison report
        """
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        
        current_metrics = current_results['aggregated_metrics']['custom']
        baseline_metrics = baseline['aggregated_metrics']['custom']
        
        comparison = {
            "response_time": {
                "current": current_metrics['avg_response_time'],
                "baseline": baseline_metrics['avg_response_time'],
                "improvement": (
                    (baseline_metrics['avg_response_time'] - current_metrics['avg_response_time'])
                    / baseline_metrics['avg_response_time'] * 100
                )
            },
            "retrieval_precision": {
                "current": current_metrics['avg_retrieval_precision'],
                "baseline": baseline_metrics['avg_retrieval_precision'],
                "improvement": (
                    (current_metrics['avg_retrieval_precision'] - baseline_metrics['avg_retrieval_precision'])
                    / baseline_metrics['avg_retrieval_precision'] * 100
                )
            },
            "cost": {
                "current": current_metrics['total_cost_usd'],
                "baseline": baseline_metrics['total_cost_usd'],
                "change": (
                    (current_metrics['total_cost_usd'] - baseline_metrics['total_cost_usd'])
                    / baseline_metrics['total_cost_usd'] * 100
                )
            }
        }
        
        logger.info("Baseline comparison completed")
        return comparison


if __name__ == "__main__":
    # Test benchmark
    print("Testing Benchmark Suite\n" + "="*60)
    
    # Create sample queries
    sample_queries = [
        BenchmarkQuery(
            id="q1",
            question="What is machine learning?",
            expected_answer="Machine learning is a type of AI that allows systems to learn from data.",
            category="basic",
            difficulty="easy"
        ),
        BenchmarkQuery(
            id="q2",
            question="Explain neural networks",
            expected_answer="Neural networks are computational models inspired by biological neurons.",
            category="intermediate",
            difficulty="medium"
        )
    ]
    
    # Note: This is a demo. In practice, you'd have documents indexed
    print(f"Created {len(sample_queries)} sample queries")
    print("\nTo run full benchmark:")
    print("1. Index your documents")
    print("2. Create evaluation_dataset.json with test queries")
    print("3. Run: python -m src.evaluation.benchmark")
