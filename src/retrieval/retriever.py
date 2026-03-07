"""
Retriever Module
Advanced retrieval strategies for RAG systems.
Supports semantic search, hybrid search, and multi-query retrieval.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
import numpy as np
from collections import defaultdict

from src.embeddings.vector_store import get_vector_store
from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
settings = get_settings()
perf_monitor = get_performance_monitor()


class RetrievalStrategy(Enum):
    """Supported retrieval strategies."""
    SEMANTIC = "semantic"  # Pure vector similarity
    HYBRID = "hybrid"  # Combines dense and sparse
    MULTI_QUERY = "multi_query"  # Multiple query variations
    CONTEXTUAL = "contextual"  # Considers document context


class Retriever:
    """
    Advanced retriever for RAG systems.
    
    Features:
    - Multiple retrieval strategies
    - Configurable similarity thresholds
    - Result deduplication
    - Score normalization
    - Metadata filtering
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model_name: Optional[str] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Vector store collection name
            embedding_model_name: Embedding model to use
            strategy: Retrieval strategy
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.collection_name = collection_name
        self.strategy = strategy
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize components (ChromaDB handles embeddings via built-in ONNX)
        self.vector_store = get_vector_store(collection_name)
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "avg_similarity": 0.0
        }
        
        logger.info(
            f"Retriever initialized: strategy={strategy.value}, "
            f"top_k={top_k}, threshold={similarity_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        strategy: Optional[RetrievalStrategy] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results (uses default if None)
            filter: Metadata filter
            strategy: Retrieval strategy (uses default if None)
            
        Returns:
            List of retrieved documents with scores and metadata
        """
        top_k = top_k or self.top_k
        strategy = strategy or self.strategy
        
        logger.debug(f"Retrieving with strategy={strategy.value}, query='{query[:50]}...'")
        
        perf_monitor.start_timer("retrieval")
        try:
            if strategy == RetrievalStrategy.SEMANTIC:
                results = self._semantic_search(query, top_k, filter)
            
            elif strategy == RetrievalStrategy.HYBRID:
                results = self._hybrid_search(query, top_k, filter)
            
            elif strategy == RetrievalStrategy.MULTI_QUERY:
                results = self._multi_query_search(query, top_k, filter)
            
            elif strategy == RetrievalStrategy.CONTEXTUAL:
                results = self._contextual_search(query, top_k, filter)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Filter by similarity threshold
            logger.info(f"[DEBUG] Before threshold filter: {len(results)} results")
            if results:
                logger.info(f"[DEBUG] Similarity scores: {[r.get('similarity', 0) for r in results[:3]]}")
            
            results = [
                r for r in results
                if r.get("similarity", 0) >= self.similarity_threshold
            ]
            
            logger.info(f"[DEBUG] After threshold filter ({self.similarity_threshold}): {len(results)} results")
            
            # Update statistics
            self.stats["total_queries"] += 1
            self.stats["total_results"] += len(results)
            if results:
                avg_sim = np.mean([r["similarity"] for r in results])
                self.stats["avg_similarity"] = (
                    (self.stats["avg_similarity"] * (self.stats["total_queries"] - 1) + avg_sim)
                    / self.stats["total_queries"]
                )
            
            perf_monitor.increment_counter("retrieval_queries", 1)
            perf_monitor.increment_counter("documents_retrieved", len(results))
        finally:
            perf_monitor.stop_timer("retrieval")
            
            logger.debug(f"Retrieved {len(results)} documents")
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Pure semantic search using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            Search results
        """
        results = self.vector_store.similarity_search(
            query_text=query,
            k=top_k,
            filter=filter
        )
        
        return results
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense (semantic) and sparse (keyword) retrieval.
        
        Uses reciprocal rank fusion to combine results.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            Fused search results
        """
        # Dense (semantic) retrieval
        semantic_results = self._semantic_search(query, top_k * 2, filter)
        
        # Sparse (keyword) retrieval - simulate with document text search
        keyword_results = self._keyword_search(query, top_k * 2, filter)
        
        # Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            [semantic_results, keyword_results],
            k=60  # RRF constant
        )
        
        return fused_results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword-based search.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            Keyword search results
        """
        # Use ChromaDB's where_document for keyword matching
        query_terms = query.lower().split()
        
        try:
            # Search using document content filter
            where_document = {
                "$or": [
                    {"$contains": term} for term in query_terms
                ]
            }
            
            results = self.vector_store.query(
                query_texts=[query],
                n_results=top_k,
                where=filter,
                where_document=where_document
            )
            
            # Format results
            formatted = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    # Calculate keyword match score
                    doc = results['documents'][0][i].lower()
                    match_score = sum(1 for term in query_terms if term in doc) / len(query_terms)
                    
                    formatted.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': match_score,  # Use match score as similarity
                        'distance': 1 - match_score
                    })
            
            return formatted
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {str(e)}, falling back to semantic")
            return self._semantic_search(query, top_k, filter)
    
    def _multi_query_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Multi-query retrieval using query variations.
        
        Generates multiple query variations and retrieves results for each,
        then combines using reciprocal rank fusion.
        
        Args:
            query: Original query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            Combined search results
        """
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        logger.debug(f"Generated {len(query_variations)} query variations")
        
        # Retrieve for each variation
        all_results = []
        for q in query_variations:
            results = self._semantic_search(q, top_k, filter)
            all_results.append(results)
        
        # Combine using RRF
        fused_results = self._reciprocal_rank_fusion(all_results)
        
        return fused_results[:top_k]
    
    def _contextual_search(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Contextual search that considers surrounding document context.
        
        Retrieves relevant chunks and includes neighboring chunks for context.
        
        Args:
            query: Search query
            top_k: Number of results
            filter: Metadata filter
            
        Returns:
            Results with additional context
        """
        # Get initial results
        initial_results = self._semantic_search(query, top_k, filter)
        
        # For each result, try to fetch neighboring chunks
        enhanced_results = []
        
        for result in initial_results:
            # Try to get neighboring chunks based on chunk_id
            chunk_id = result['metadata'].get('chunk_id', '')
            
            # Add the original result
            enhanced_results.append(result)
            
            # Note: In a production system, you'd implement logic to fetch
            # neighboring chunks based on chunk numbering or document structure
        
        return enhanced_results[:top_k]
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """
        Generate query variations for multi-query retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Simple variations (in production, use LLM for better variations)
        query_lower = query.lower()
        
        # Question variations
        if "what" in query_lower:
            variations.append(query.replace("what", "which").replace("What", "Which"))
        if "how" in query_lower:
            variations.append(query.replace("how", "what method").replace("How", "What method"))
        
        # Add a more specific variation
        if "?" in query:
            variations.append(query.replace("?", " in detail?"))
        else:
            variations.append(f"{query} - provide details")
        
        # Add a simpler variation (extract key terms)
        words = query.split()
        if len(words) > 3:
            # Take first and last meaningful words
            key_terms = " ".join([w for w in words if len(w) > 3][:3])
            variations.append(key_terms)
        
        return variations
    
    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple result lists using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank_i))
        
        Args:
            result_lists: List of result lists to fuse
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Calculate RRF scores
        doc_scores = defaultdict(float)
        doc_data = {}
        
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                
                # RRF score
                doc_scores[doc_id] += 1.0 / (k + rank)
                
                # Store document data (from first occurrence)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result
        
        # Sort by RRF score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final result list
        fused_results = []
        for doc_id, score in sorted_docs:
            result = doc_data[doc_id].copy()
            result['similarity'] = min(score / len(result_lists), 1.0)  # Normalize
            result['rrf_score'] = score
            fused_results.append(result)
        
        return fused_results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve results for multiple queries.
        
        Args:
            queries: List of queries
            top_k: Number of results per query
            filter: Metadata filter
            
        Returns:
            List of result lists
        """
        logger.info(f"Batch retrieving for {len(queries)} queries")
        
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k, filter)
            all_results.append(results)
        
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        stats.update({
            "collection": self.collection_name,
            "strategy": self.strategy.value,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "vector_store_count": self.vector_store.count()
        })
        return stats


# Convenience functions

def retrieve_documents(
    query: str,
    top_k: int = 5,
    collection_name: str = "documents",
    strategy: str = "semantic",
    filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve documents.
    
    Args:
        query: Search query
        top_k: Number of results
        collection_name: Collection to search
        strategy: Retrieval strategy
        filter: Metadata filter
        
    Returns:
        Retrieved documents
    """
    strategy_enum = RetrievalStrategy(strategy)
    retriever = Retriever(
        collection_name=collection_name,
        strategy=strategy_enum,
        top_k=top_k
    )
    
    return retriever.retrieve(query, filter=filter)


if __name__ == "__main__":
    # Test the retriever
    print("Testing Retriever\n" + "="*60)
    
    # Note: Requires existing indexed documents
    from src.embeddings.indexing import IndexingPipeline
    from src.ingestion.chunker import Chunk
    
    # Setup test data
    pipeline = IndexingPipeline(collection_name="test_retriever")
    pipeline.reset()
    
    # Create and index test documents
    test_docs = [
        ("Python is a versatile programming language for AI and ML.", "python"),
        ("Machine learning enables computers to learn from data.", "ml"),
        ("Deep learning uses neural networks with multiple layers.", "dl"),
        ("Natural language processing helps computers understand text.", "nlp"),
        ("Computer vision enables image recognition and analysis.", "cv")
    ]
    
    chunks = []
    for i, (text, topic) in enumerate(test_docs):
        chunk = Chunk(
            text=text,
            chunk_id=f"chunk_{i}",
            start_char=0,
            end_char=len(text),
            metadata={"topic": topic}
        )
        chunks.append(chunk)
    
    print("Indexing test documents...")
    pipeline.index_chunks(chunks, show_progress=False)
    print(f"Indexed {len(chunks)} documents\n")
    
    # Test different retrieval strategies
    retriever = Retriever(collection_name="test_retriever", top_k=3)
    
    test_queries = [
        "What is machine learning?",
        "Tell me about neural networks",
        "Python programming"
    ]
    
    strategies = [
        RetrievalStrategy.SEMANTIC,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.MULTI_QUERY
    ]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.value.upper()}")
        print("-" * 60)
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.retrieve(query, strategy=strategy, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['document'][:50]}...")
                print(f"     Similarity: {result['similarity']:.4f}")
    
    # Statistics
    print(f"\nRetrieval Statistics:")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    pipeline.reset()
    
    print("\nRetriever test completed!")
