"""
Search Agent Module
Specialized agent for enhanced information retrieval.
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent, AgentTask, AgentResult, AgentType, AgentStatus
from src.retrieval import Retriever, RetrievalStrategy
from src.retrieval.query_processor import QueryProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SearchAgent(BaseAgent):
    """
    Search Agent for enhanced information retrieval.
    
    Capabilities:
    - Query expansion and refinement
    - Multi-strategy search
    - Source ranking
    - Relevance filtering
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        top_k: int = 10
    ):
        """
        Initialize search agent.
        
        Args:
            collection_name: Vector collection name
            top_k: Number of results to retrieve
        """
        super().__init__(
            name="SearchAgent",
            agent_type=AgentType.SEARCH,
            description="Enhanced information retrieval with query refinement"
        )
        
        self.retriever = Retriever(
            collection_name=collection_name,
            strategy=RetrievalStrategy.SEMANTIC,
            top_k=top_k
        )
        self.query_processor = QueryProcessor()
        
        logger.info(f"SearchAgent initialized: collection={collection_name}, top_k={top_k}")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Must contain 'query' key
            
        Returns:
            True if valid
        """
        return "query" in input_data and isinstance(input_data["query"], str)
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with variations.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add processed query
        processed = self.query_processor.process(query)
        if processed.cleaned != query:
            variations.append(processed.cleaned)
        
        # Add expanded terms
        if processed.expanded_terms:
            expanded = f"{query} {' '.join(processed.expanded_terms)}"
            variations.append(expanded)
        
        return variations
    
    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute enhanced search.
        
        Args:
            task: Search task
            
        Returns:
            AgentResult with retrieved documents
        """
        query = task.input_data["query"]
        enhanced = task.input_data.get("enhanced", False)
        
        logger.info(f"SearchAgent executing: query='{query[:50]}...', enhanced={enhanced}")
        
        # Standard search
        if not enhanced:
            results = self.retriever.retrieve(query)
            
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=AgentStatus.COMPLETED,
                output={
                    "sources": results,
                    "query": query,
                    "count": len(results)
                },
                metadata={
                    "strategy": "standard",
                    "query_variations": 1
                }
            )
        
        # Enhanced search with query expansion
        query_variations = self.expand_query(query)
        all_results = []
        seen_ids = set()
        
        for variation in query_variations:
            results = self.retriever.retrieve(variation)
            
            # Deduplicate by document ID
            for doc in results:
                doc_id = doc.get("id", doc.get("content", "")[:50])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append(doc)
        
        # Sort by relevance (similarity score)
        all_results.sort(
            key=lambda x: x.get("similarity", 0),
            reverse=True
        )
        
        # Return top results
        top_results = all_results[:self.retriever.top_k]
        
        logger.info(
            f"SearchAgent completed: {len(top_results)} results "
            f"from {len(query_variations)} query variations"
        )
        
        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=AgentStatus.COMPLETED,
            output={
                "sources": top_results,
                "query": query,
                "count": len(top_results)
            },
            metadata={
                "strategy": "enhanced",
                "query_variations": len(query_variations),
                "total_retrieved": len(all_results),
                "variations": query_variations
            }
        )
