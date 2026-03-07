"""
Reranker Module
Improves retrieval quality by re-ranking initial results.
Supports multiple re-ranking strategies including cross-encoder models.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
settings = get_settings()
perf_monitor = get_performance_monitor()


@dataclass
class RankedResult:
    """A re-ranked search result."""
    document: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    final_score: float
    rank: int
    id: str


class Reranker:
    """
    Re-ranks search results to improve relevance.
    
    Strategies:
    - Score-based: Uses original similarity scores
    - Diversity: Promotes diverse results
    - MMR: Maximal Marginal Relevance
    - Cross-encoder: Uses transformer cross-encoder (future enhancement)
    """
    
    def __init__(
        self,
        strategy: str = "score",
        diversity_weight: float = 0.3,
        use_metadata: bool = True
    ):
        """
        Initialize the reranker.
        
        Args:
            strategy: Reranking strategy ('score', 'diversity', 'mmr')
            diversity_weight: Weight for diversity in MMR (0-1)
            use_metadata: Whether to consider metadata in ranking
        """
        self.strategy = strategy
        self.diversity_weight = diversity_weight
        self.use_metadata = use_metadata
        
        logger.info(f"Reranker initialized: strategy={strategy}")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """
        Re-rank search results.
        
        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        logger.debug(f"Re-ranking {len(results)} results with strategy={self.strategy}")
        
        perf_monitor.start_timer("reranking")
        try:
            if self.strategy == "score":
                reranked = self._rerank_by_score(results)
            
            elif self.strategy == "diversity":
                reranked = self._rerank_by_diversity(results)
            
            elif self.strategy == "mmr":
                reranked = self._rerank_mmr(query, results)
            
            else:
                logger.warning(f"Unknown strategy {self.strategy}, using score-based")
                reranked = self._rerank_by_score(results)
            
            # Apply top_k limit
            if top_k:
                reranked = reranked[:top_k]
            
            perf_monitor.increment_counter("reranking_operations", 1)
        finally:
            perf_monitor.stop_timer("reranking")
        
        return reranked
    
    def _rerank_by_score(
        self,
        results: List[Dict[str, Any]]
    ) -> List[RankedResult]:
        """
        Simple reranking based on original similarity scores.
        
        Args:
            results: Original results
            
        Returns:
            Sorted results
        """
        ranked = []
        
        for rank, result in enumerate(results, 1):
            original_score = result.get('similarity', result.get('rrf_score', 0))
            
            # Apply metadata boost if enabled
            metadata_boost = 0.0
            if self.use_metadata:
                metadata_boost = self._calculate_metadata_boost(result.get('metadata', {}))
            
            final_score = original_score + metadata_boost
            
            ranked.append(RankedResult(
                document=result['document'],
                metadata=result.get('metadata', {}),
                original_score=original_score,
                rerank_score=metadata_boost,
                final_score=final_score,
                rank=rank,
                id=result.get('id', f'doc_{rank}')
            ))
        
        # Sort by final score
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Update ranks
        for i, item in enumerate(ranked, 1):
            item.rank = i
        
        return ranked
    
    def _rerank_by_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> List[RankedResult]:
        """
        Rerank to promote diversity in results.
        
        Penalizes results that are too similar to already selected ones.
        
        Args:
            results: Original results
            
        Returns:
            Diversified results
        """
        if not results:
            return []
        
        # Start with highest scoring result
        selected = []
        remaining = results.copy()
        
        # Select first result
        first = remaining.pop(0)
        selected.append(first)
        
        # Iteratively select diverse results
        while remaining and len(selected) < len(results):
            best_idx = 0
            best_score = -1
            
            for idx, candidate in enumerate(remaining):
                # Base score
                base_score = candidate.get('similarity', 0)
                
                # Calculate diversity penalty
                diversity_penalty = 0
                for selected_doc in selected:
                    # Simple diversity: penalize if same category or topic
                    if (candidate.get('metadata', {}).get('category') ==
                        selected_doc.get('metadata', {}).get('category')):
                        diversity_penalty += 0.2
                    
                    # Content similarity (approximate via text length difference)
                    len_diff = abs(len(candidate['document']) - len(selected_doc['document']))
                    if len_diff < 50:  # Very similar length
                        diversity_penalty += 0.1
                
                # Combined score
                final_score = base_score - diversity_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        # Convert to RankedResult
        ranked = []
        for rank, result in enumerate(selected, 1):
            original_score = result.get('similarity', 0)
            
            ranked.append(RankedResult(
                document=result['document'],
                metadata=result.get('metadata', {}),
                original_score=original_score,
                rerank_score=0.0,
                final_score=original_score,
                rank=rank,
                id=result.get('id', f'doc_{rank}')
            ))
        
        return ranked
    
    def _rerank_mmr(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[RankedResult]:
        """
        Maximal Marginal Relevance (MMR) reranking.
        
        Balances relevance and diversity:
        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        
        Args:
            query: Search query
            results: Original results
            
        Returns:
            MMR-reranked results
        """
        if not results:
            return []
        
        lambda_param = 1.0 - self.diversity_weight
        
        selected = []
        remaining = list(range(len(results)))
        
        # Select first (highest relevance)
        first_idx = 0
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Iteratively select using MMR
        while remaining and len(selected) < len(results):
            best_mmr = -float('inf')
            best_idx = None
            
            for idx in remaining:
                relevance = results[idx].get('similarity', 0)
                
                # Calculate max similarity to already selected documents
                max_sim = 0
                for sel_idx in selected:
                    # Approximate similarity via text overlap
                    sim = self._approximate_similarity(
                        results[idx]['document'],
                        results[sel_idx]['document']
                    )
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        # Build ranked results
        ranked = []
        for rank, idx in enumerate(selected, 1):
            result = results[idx]
            original_score = result.get('similarity', 0)
            
            ranked.append(RankedResult(
                document=result['document'],
                metadata=result.get('metadata', {}),
                original_score=original_score,
                rerank_score=0.0,
                final_score=original_score,
                rank=rank,
                id=result.get('id', f'doc_{rank}')
            ))
        
        return ranked
    
    def _calculate_metadata_boost(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate score boost based on metadata.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Boost score
        """
        boost = 0.0
        
        # Boost for certain categories
        category = metadata.get('category', '').lower()
        if category in ['technical', 'official', 'documentation']:
            boost += 0.05
        
        # Boost for recent documents (if timestamp available)
        # This is a placeholder - implement based on your metadata
        
        # Boost for verified/authoritative sources
        if metadata.get('verified', False):
            boost += 0.1
        
        return min(boost, 0.2)  # Cap boost at 0.2
    
    def _approximate_similarity(self, text1: str, text2: str) -> float:
        """
        Approximate text similarity (simple word overlap).
        
        In production, use proper embedding similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    def filter_by_threshold(
        self,
        ranked_results: List[RankedResult],
        threshold: float
    ) -> List[RankedResult]:
        """
        Filter results by score threshold.
        
        Args:
            ranked_results: Ranked results
            threshold: Minimum score threshold
            
        Returns:
            Filtered results
        """
        return [r for r in ranked_results if r.final_score >= threshold]
    
    def get_top_k(
        self,
        ranked_results: List[RankedResult],
        k: int
    ) -> List[RankedResult]:
        """
        Get top K results.
        
        Args:
            ranked_results: Ranked results
            k: Number of results
            
        Returns:
            Top K results
        """
        return ranked_results[:k]


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    strategy: str = "mmr",
    top_k: Optional[int] = None,
    diversity_weight: float = 0.3
) -> List[RankedResult]:
    """
    Convenience function to rerank results.
    
    Args:
        query: Search query
        results: Initial results
        strategy: Reranking strategy
        top_k: Number of results to return
        diversity_weight: Diversity weight for MMR
        
    Returns:
        Reranked results
    """
    reranker = Reranker(strategy=strategy, diversity_weight=diversity_weight)
    return reranker.rerank(query, results, top_k)


if __name__ == "__main__":
    # Test the reranker
    print("Testing Reranker\n" + "="*60)
    
    # Sample results
    sample_results = [
        {
            'id': 'doc1',
            'document': 'Python is a programming language used for AI and machine learning.',
            'metadata': {'category': 'technical', 'topic': 'python'},
            'similarity': 0.85
        },
        {
            'id': 'doc2',
            'document': 'Machine learning algorithms learn patterns from data automatically.',
            'metadata': {'category': 'technical', 'topic': 'ml'},
            'similarity': 0.82
        },
        {
            'id': 'doc3',
            'document': 'Python has extensive libraries for data science and visualization.',
            'metadata': {'category': 'technical', 'topic': 'python'},
            'similarity': 0.80
        },
        {
            'id': 'doc4',
            'document': 'Deep learning is a subset of machine learning using neural networks.',
            'metadata': {'category': 'academic', 'topic': 'dl'},
            'similarity': 0.75
        },
        {
            'id': 'doc5',
            'document': 'Data science combines statistics, programming, and domain knowledge.',
            'metadata': {'category': 'general', 'topic': 'data-science'},
            'similarity': 0.70
        }
    ]
    
    query = "What is machine learning in Python?"
    
    strategies = ['score', 'diversity', 'mmr']
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.upper()}")
        print("-" * 60)
        
        reranker = Reranker(strategy=strategy, diversity_weight=0.4)
        reranked = reranker.rerank(query, sample_results, top_k=3)
        
        for result in reranked:
            print(f"\nRank {result.rank}:")
            print(f"  Document: {result.document[:60]}...")
            print(f"  Original: {result.original_score:.4f}")
            print(f"  Final: {result.final_score:.4f}")
            print(f"  Category: {result.metadata.get('category')}")
    
    print("\nReranker test completed!")
