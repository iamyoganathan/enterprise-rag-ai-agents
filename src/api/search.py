"""
Search API Endpoints
Handle semantic search and document retrieval.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    strategy: str = Field(default="semantic", description="Search strategy: semantic, hybrid, multi_query")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class SearchResult(BaseModel):
    """Single search result."""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total: int
    latency: float


class SimilarDocRequest(BaseModel):
    """Similar document request."""
    document_id: str
    top_k: int = Field(default=5, ge=1, le=50)


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform semantic search.
    
    Strategies:
    - semantic: Pure semantic similarity search
    - hybrid: Combines semantic and keyword matching
    - multi_query: Generates multiple query variations
    """
    import time
    start_time = time.time()
    
    try:
        from src.retrieval import get_retriever, get_query_processor
        
        # Process query
        query_processor = get_query_processor()
        processed = query_processor.process(request.query)
        
        logger.info(
            f"Search query: '{request.query}' | "
            f"Strategy: {request.strategy} | "
            f"Top K: {request.top_k}"
        )
        
        # Retrieve documents
        retriever = get_retriever(
            collection_name="documents",
            retrieval_strategy=request.strategy,
            top_k=request.top_k
        )
        
        results = retriever.retrieve(
            query=processed.cleaned,
            filter=request.filter
        )
        
        # Format results
        search_results = [
            SearchResult(
                id=result["id"],
                text=result["document"],
                score=result["similarity"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        latency = time.time() - start_time
        
        logger.info(f"Search completed: {len(search_results)} results in {latency:.2f}s")
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results),
            latency=latency
        )
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similar", response_model=SearchResponse)
async def find_similar_documents(request: SimilarDocRequest):
    """
    Find documents similar to a given document.
    """
    import time
    start_time = time.time()
    
    try:
        from src.embeddings import get_vector_store
        
        vector_store = get_vector_store()
        
        # Get the document
        doc_results = vector_store.get_documents(
            where={"document_id": request.document_id},
            limit=1
        )
        
        if not doc_results["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Use the document text as query
        query_text = doc_results["documents"][0]
        
        # Search for similar
        from src.retrieval import get_retriever
        
        retriever = get_retriever(
            collection_name="documents",
            retrieval_strategy="semantic",
            top_k=request.top_k + 1  # +1 to exclude self
        )
        
        results = retriever.retrieve(query=query_text)
        
        # Exclude the original document
        results = [r for r in results if r["metadata"].get("document_id") != request.document_id][:request.top_k]
        
        # Format results
        search_results = [
            SearchResult(
                id=result["id"],
                text=result["document"],
                score=result["similarity"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        latency = time.time() - start_time
        
        return SearchResponse(
            query=f"Similar to document: {request.document_id}",
            results=search_results,
            total=len(search_results),
            latency=latency
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-analysis")
async def analyze_query(query: str):
    """
    Analyze a query to extract intent, keywords, and generate expansions.
    """
    try:
        from src.retrieval import get_query_processor
        
        query_processor = get_query_processor()
        result = query_processor.process(query)
        
        return {
            "original_query": query,
            "cleaned_query": result.cleaned,
            "intent": result.intent,
            "keywords": result.keywords,
            "expanded_queries": result.expanded
        }
    
    except Exception as e:
        logger.error(f"Query analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_search_strategies():
    """
    Get available search strategies and their descriptions.
    """
    return {
        "strategies": [
            {
                "name": "semantic",
                "description": "Pure semantic similarity using vector embeddings",
                "best_for": "Conceptual understanding and meaning-based search"
            },
            {
                "name": "hybrid",
                "description": "Combines semantic and keyword matching",
                "best_for": "Balanced search with both meaning and exact terms"
            },
            {
                "name": "multi_query",
                "description": "Generates multiple query variations for comprehensive search",
                "best_for": "Complex queries requiring multiple perspectives"
            }
        ]
    }
