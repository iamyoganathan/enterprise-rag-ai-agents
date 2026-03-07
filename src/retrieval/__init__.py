"""
Retrieval Module
Advanced document retrieval and context building for RAG systems.
"""

from src.retrieval.retriever import (
    Retriever,
    RetrievalStrategy,
    retrieve_documents
)
from src.retrieval.reranker import (
    Reranker,
    RankedResult,
    rerank_results
)
from src.retrieval.query_processor import (
    QueryProcessor,
    ProcessedQuery,
    process_query
)
from src.retrieval.context_builder import (
    ContextBuilder,
    Context,
    build_context
)

# Singleton instances
_retriever_instance = None
_query_processor_instance = None
_reranker_instance = None
_context_builder_instance = None

def get_retriever(collection_name="documents", retrieval_strategy="semantic", top_k=5):
    """Get or create a Retriever instance (singleton per collection)."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever(
            collection_name=collection_name,
            strategy=RetrievalStrategy(retrieval_strategy),
            top_k=top_k
        )
    return _retriever_instance

def get_query_processor():
    """Get or create a QueryProcessor instance."""
    global _query_processor_instance
    if _query_processor_instance is None:
        _query_processor_instance = QueryProcessor()
    return _query_processor_instance

def get_reranker(strategy="mmr"):
    """Get or create a Reranker instance (singleton)."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(strategy=strategy)
    return _reranker_instance

def get_context_builder():
    """Get or create a ContextBuilder instance."""
    global _context_builder_instance
    if _context_builder_instance is None:
        _context_builder_instance = ContextBuilder()
    return _context_builder_instance

__all__ = [
    "Retriever",
    "RetrievalStrategy",
    "retrieve_documents",
    "get_retriever",
    "Reranker",
    "RankedResult",
    "rerank_results",
    "get_reranker",
    "QueryProcessor",
    "ProcessedQuery",
    "process_query",
    "get_query_processor",
    "ContextBuilder",
    "Context",
    "build_context",
    "get_context_builder"
]

__version__ = "1.0.0"
