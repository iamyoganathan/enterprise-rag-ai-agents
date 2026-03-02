"""
RAGAS Evaluator Module
Evaluate RAG system using RAGAS metrics.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not installed. Run: pip install ragas datasets")

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGASResult:
    """RAGAS evaluation result."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: Optional[float] = None
    answer_similarity: Optional[float] = None
    overall_score: float = 0.0
    execution_time: float = 0.0


class RAGASEvaluator:
    """
    Evaluate RAG system using RAGAS metrics.
    
    RAGAS Metrics:
    - Faithfulness: Answer is grounded in context (no hallucination)
    - Answer Relevancy: Answer addresses the question
    - Context Precision: Retrieved context is relevant
    - Context Recall: All relevant context is retrieved
    - Answer Correctness: Answer matches ground truth (if available)
    - Answer Similarity: Semantic similarity to ground truth
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "all-mpnet-base-v2"
    ):
        """Initialize RAGAS evaluator.
        
        Args:
            llm_model: LLM model for evaluation
            embedding_model: Embedding model for similarity
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS not installed. Run: pip install ragas datasets")
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        logger.info(f"RAGAS evaluator initialized: {llm_model}")
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGASResult:
        """Evaluate a single Q&A pair.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved context documents
            ground_truth: Expected answer (optional)
            
        Returns:
            RAGAS evaluation result
        """
        start_time = time.time()
        
        # Prepare data for RAGAS
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        # Add ground truth if available
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        if ground_truth:
            metrics.extend([answer_correctness, answer_similarity])
        
        # Run evaluation
        try:
            result = evaluate(dataset, metrics=metrics)
            
            execution_time = time.time() - start_time
            
            # Extract scores
            scores = result.to_pandas().iloc[0].to_dict()
            
            ragas_result = RAGASResult(
                faithfulness=scores.get('faithfulness', 0.0),
                answer_relevancy=scores.get('answer_relevancy', 0.0),
                context_precision=scores.get('context_precision', 0.0),
                context_recall=scores.get('context_recall', 0.0),
                answer_correctness=scores.get('answer_correctness') if ground_truth else None,
                answer_similarity=scores.get('answer_similarity') if ground_truth else None,
                execution_time=execution_time
            )
            
            # Calculate overall score
            base_scores = [
                ragas_result.faithfulness,
                ragas_result.answer_relevancy,
                ragas_result.context_precision,
                ragas_result.context_recall
            ]
            
            if ground_truth:
                base_scores.extend([
                    ragas_result.answer_correctness or 0.0,
                    ragas_result.answer_similarity or 0.0
                ])
            
            ragas_result.overall_score = sum(base_scores) / len(base_scores)
            
            logger.info(f"RAGAS evaluation completed: overall={ragas_result.overall_score:.3f}")
            return ragas_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            raise
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple Q&A pairs.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: List of expected answers (optional)
            
        Returns:
            Aggregated RAGAS metrics
        """
        start_time = time.time()
        
        # Prepare data
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        if ground_truths:
            metrics.extend([answer_correctness, answer_similarity])
        
        # Run evaluation
        try:
            result = evaluate(dataset, metrics=metrics)
            execution_time = time.time() - start_time
            
            # Convert to dataframe
            df = result.to_pandas()
            
            # Calculate aggregated metrics
            aggregated = {
                "num_samples": len(questions),
                "faithfulness_mean": float(df['faithfulness'].mean()),
                "faithfulness_std": float(df['faithfulness'].std()),
                "answer_relevancy_mean": float(df['answer_relevancy'].mean()),
                "answer_relevancy_std": float(df['answer_relevancy'].std()),
                "context_precision_mean": float(df['context_precision'].mean()),
                "context_precision_std": float(df['context_precision'].std()),
                "context_recall_mean": float(df['context_recall'].mean()),
                "context_recall_std": float(df['context_recall'].std()),
                "execution_time": execution_time,
            }
            
            if ground_truths:
                aggregated.update({
                    "answer_correctness_mean": float(df['answer_correctness'].mean()),
                    "answer_correctness_std": float(df['answer_correctness'].std()),
                    "answer_similarity_mean": float(df['answer_similarity'].mean()),
                    "answer_similarity_std": float(df['answer_similarity'].std()),
                })
            
            # Calculate overall score
            mean_scores = [
                aggregated["faithfulness_mean"],
                aggregated["answer_relevancy_mean"],
                aggregated["context_precision_mean"],
                aggregated["context_recall_mean"]
            ]
            
            if ground_truths:
                mean_scores.extend([
                    aggregated["answer_correctness_mean"],
                    aggregated["answer_similarity_mean"]
                ])
            
            aggregated["overall_score"] = sum(mean_scores) / len(mean_scores)
            
            logger.info(
                f"Batch RAGAS evaluation completed: "
                f"{len(questions)} samples, overall={aggregated['overall_score']:.3f}"
            )
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {str(e)}")
            raise
    
    def get_metrics_explanation(self) -> Dict[str, str]:
        """Get explanation of RAGAS metrics.
        
        Returns:
            Dictionary of metric explanations
        """
        return {
            "faithfulness": (
                "Measures if the answer is grounded in the provided context. "
                "Score of 1.0 means no hallucination."
            ),
            "answer_relevancy": (
                "Measures how well the answer addresses the question. "
                "Higher score means more relevant answer."
            ),
            "context_precision": (
                "Measures if the retrieved contexts are relevant to the question. "
                "Higher score means better retrieval."
            ),
            "context_recall": (
                "Measures if all relevant information was retrieved. "
                "Higher score means complete retrieval."
            ),
            "answer_correctness": (
                "Measures factual accuracy against ground truth. "
                "Requires reference answer."
            ),
            "answer_similarity": (
                "Measures semantic similarity to ground truth. "
                "Requires reference answer."
            )
        }


if __name__ == "__main__":
    # Test RAGAS evaluator
    print("Testing RAGAS Evaluator\n" + "="*60)
    
    if not RAGAS_AVAILABLE:
        print("RAGAS not available. Install with: pip install ragas datasets")
    else:
        evaluator = RAGASEvaluator()
        
        # Test single evaluation
        question = "What is machine learning?"
        answer = "Machine learning is a subset of AI that enables computers to learn from data."
        contexts = [
            "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data.",
            "ML algorithms improve their performance through experience without being explicitly programmed."
        ]
        ground_truth = "Machine learning is a type of AI that allows systems to learn from data."
        
        result = evaluator.evaluate_single(question, answer, contexts, ground_truth)
        
        print(f"\nSingle Evaluation Results:")
        print(f"  Faithfulness: {result.faithfulness:.3f}")
        print(f"  Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"  Context Precision: {result.context_precision:.3f}")
        print(f"  Context Recall: {result.context_recall:.3f}")
        if result.answer_correctness:
            print(f"  Answer Correctness: {result.answer_correctness:.3f}")
        if result.answer_similarity:
            print(f"  Answer Similarity: {result.answer_similarity:.3f}")
        print(f"  Overall Score: {result.overall_score:.3f}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
