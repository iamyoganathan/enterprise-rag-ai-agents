"""
Evaluation Script
Run comprehensive evaluation of the RAG system.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.benchmark import Benchmark, BenchmarkQuery
from src.utils.logger import get_logger

logger = get_logger(__name__)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_metrics(metrics: Dict[str, Any], indent: int = 0):
    """Print metrics in a formatted way."""
    prefix = "  " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_metrics(value, indent + 1)
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{prefix}{key}: {value:,}")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    """Run evaluation."""
    print_section("Enterprise RAG System - Evaluation Suite")
    
    # Load evaluation dataset
    dataset_path = project_root / "data" / "evaluation_dataset.json"
    
    if not dataset_path.exists():
        print(f"\n❌ Error: Evaluation dataset not found at {dataset_path}")
        print("Please create data/evaluation_dataset.json with test queries.")
        return
    
    print(f"\n📂 Loading evaluation dataset from: {dataset_path}")
    
    # Initialize benchmark
    benchmark = Benchmark(collection_name="documents", use_ragas=True)
    
    # Load queries
    queries = benchmark.load_queries(dataset_path)
    
    print(f"✅ Loaded {len(queries)} test queries")
    
    # Display query categories
    categories = {}
    for q in queries:
        cat = q.category or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Query Distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  • {cat}: {count} queries")
    
    # Run benchmark
    print_section("Running Benchmark")
    print("⏳ This may take a few minutes depending on the number of queries...")
    print("   Each query involves retrieval, LLM generation, and evaluation.")
    
    try:
        results = benchmark.run_benchmark(
            queries=queries,
            save_results=True,
            output_file=project_root / "data" / "benchmark_results.json"
        )
        
        # Display results
        print_section("Benchmark Results")
        
        metadata = results['metadata']
        print(f"\n📈 Summary:")
        print(f"  Total Queries: {metadata['total_queries']}")
        print(f"  Successful: {metadata['successful_queries']}")
        print(f"  Failed: {metadata['failed_queries']}")
        print(f"  Success Rate: {metadata['successful_queries'] / metadata['total_queries'] * 100:.1f}%")
        
        # Custom metrics
        print_section("Performance Metrics")
        custom = results['aggregated_metrics']['custom']
        print(f"\n⚡ Response Times:")
        print(f"  Average: {custom['avg_response_time']:.2f}s")
        print(f"  P95: {custom['p95_response_time']:.2f}s")
        print(f"  P99: {custom['p99_response_time']:.2f}s")
        print(f"  Min: {custom['min_response_time']:.2f}s")
        print(f"  Max: {custom['max_response_time']:.2f}s")
        
        print(f"\n🔍 Retrieval:")
        print(f"  Average Time: {custom['avg_retrieval_time']:.2f}s")
        print(f"  Average Precision: {custom['avg_retrieval_precision']:.2%}")
        
        print(f"\n🤖 Generation:")
        print(f"  Average Time: {custom['avg_generation_time']:.2f}s")
        print(f"  Average Tokens: {custom['avg_total_tokens']:,}")
        
        # Cost
        print_section("Cost Analysis")
        cost = results['cost_breakdown']
        print(f"\n💰 Token Usage:")
        print(f"  Total Tokens: {cost['total_tokens']:,}")
        print(f"  Prompt Tokens: {cost['total_prompt_tokens']:,}")
        print(f"  Completion Tokens: {cost['total_completion_tokens']:,}")
        
        print(f"\n💵 Estimated Cost:")
        print(f"  Prompt Cost: ${cost['prompt_cost_usd']:.6f}")
        print(f"  Completion Cost: ${cost['completion_cost_usd']:.6f}")
        print(f"  Total Cost: ${cost['total_cost_usd']:.6f}")
        print(f"  Cost per Query: ${cost['total_cost_usd'] / metadata['successful_queries']:.6f}")
        
        # RAGAS metrics (if available)
        if 'ragas' in results['aggregated_metrics']:
            print_section("RAGAS Quality Metrics")
            ragas = results['aggregated_metrics']['ragas']
            
            print(f"\n📊 Quality Scores (0.0 - 1.0):")
            print(f"  Faithfulness: {ragas['avg_faithfulness']:.3f}")
            print(f"    └─ Answer is grounded in context (no hallucination)")
            
            print(f"  Answer Relevancy: {ragas['avg_answer_relevancy']:.3f}")
            print(f"    └─ Answer addresses the question")
            
            print(f"  Context Precision: {ragas['avg_context_precision']:.3f}")
            print(f"    └─ Retrieved contexts are relevant")
            
            print(f"  Context Recall: {ragas['avg_context_recall']:.3f}")
            print(f"    └─ All relevant context was retrieved")
            
            print(f"\n  Overall Score: {ragas['avg_overall_score']:.3f}")
        
        # Sample results
        print_section("Sample Results")
        sample_count = min(3, len(results['results']))
        print(f"\n📝 Showing {sample_count} sample query results:\n")
        
        for i, result in enumerate(results['results'][:sample_count], 1):
            print(f"{i}. Query: {result['question']}")
            print(f"   Answer: {result['answer'][:150]}...")
            print(f"   Time: {result['response_time']:.2f}s | Sources: {result['num_sources']}")
            if result.get('ragas_scores'):
                print(f"   RAGAS Overall: {result['ragas_scores']['overall_score']:.3f}")
            print()
        
        # Save location
        print_section("Results Saved")
        print(f"\n💾 Full results saved to:")
        print(f"   {project_root / 'data' / 'benchmark_results.json'}")
        print(f"\n   You can review detailed results including:")
        print(f"   • Individual query results")
        print(f"   • Retrieved sources for each query")
        print(f"   • Detailed RAGAS scores")
        print(f"   • Token usage breakdown")
        
        print_section("Evaluation Complete")
        print("\n✅ Evaluation completed successfully!")
        print("\n📊 Use these metrics for your MCA project report:")
        print("   • System response time and throughput")
        print("   • Retrieval precision and accuracy")
        print("   • RAGAS quality metrics (faithfulness, relevancy)")
        print("   • Cost analysis and efficiency")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure documents are indexed in ChromaDB")
        print("2. Check that the backend is running")
        print("3. Verify evaluation_dataset.json is properly formatted")
        print("4. Run: pip install ragas datasets")
        return


if __name__ == "__main__":
    main()
