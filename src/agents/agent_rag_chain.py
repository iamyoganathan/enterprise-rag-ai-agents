"""
Agent-Enhanced RAG Chain Module
Integrates LangGraph agent system with traditional RAG pipeline.
"""

from typing import Dict, Any, Optional, Iterator
from dataclasses import dataclass
import time

from src.agents.orchestrator import get_orchestrator
from src.agents.base import get_agent_registry
from src.agents.search_agent import SearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.synthesis_agent import SynthesisAgent
from src.llm.rag_chain import RAGChain, RAGResponse
from src.utils.logger import get_logger
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
perf_monitor = get_performance_monitor()


class AgentRAGChain:
    """
    Agent-enhanced RAG chain with LangGraph orchestration.

    Features:
    - Automatic complexity detection
    - LangGraph multi-agent pipeline for complex queries
    - Fallback to basic RAG for simple queries
    - Performance monitoring
    """

    def __init__(
        self,
        collection_name: str = "documents",
        enable_agents: bool = True,
        agent_threshold: str = "moderate"
    ):
        # Initialize basic RAG chain
        self.basic_rag = RAGChain(collection_name=collection_name)

        # Agent system
        self.enable_agents = enable_agents
        self.agent_threshold = agent_threshold

        if enable_agents:
            self._initialize_agents(collection_name)

        logger.info(
            f"AgentRAGChain initialized: agents={enable_agents}, "
            f"threshold={agent_threshold}, orchestration=langgraph"
        )

    def _initialize_agents(self, collection_name: str):
        """Initialize and register agents."""
        registry = get_agent_registry()

        registry.register(SearchAgent(collection_name=collection_name))
        registry.register(AnalysisAgent())
        registry.register(SynthesisAgent())

        self.orchestrator = get_orchestrator()

        logger.info("LangGraph agent system initialized with 3 agents")

    def should_use_agents(self, query: str) -> bool:
        """Determine if agents should be used for this query."""
        if not self.enable_agents:
            return False

        intent = self.orchestrator.analyze_intent(query)

        complexity_levels = ["simple", "moderate", "complex"]
        threshold_idx = complexity_levels.index(self.agent_threshold)
        query_idx = complexity_levels.index(intent.complexity)

        return query_idx >= threshold_idx

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        stream: bool = False,
        force_agents: bool = False,
        **kwargs
    ) -> RAGResponse:
        """Query with intelligent agent routing via LangGraph."""
        logger.info(f"AgentRAGChain query: '{query[:50]}...'")

        start_time = time.time()
        perf_monitor.start_timer("agent_rag_query")

        use_agents = force_agents or self.should_use_agents(query)

        if stream:
            logger.info("Streaming mode - using basic RAG")
            return self.basic_rag.query(query, conversation_id, stream=True, **kwargs)

        try:
            if use_agents:
                logger.info("Using LangGraph agent orchestration")
                response = self._query_with_agents(query, **kwargs)
            else:
                logger.info("Using basic RAG")
                response = self.basic_rag.query(query, conversation_id, stream=False, **kwargs)

            perf_monitor.end_timer("agent_rag_query")

            return response

        except Exception as e:
            logger.error(f"AgentRAGChain query failed: {str(e)}")
            logger.info("Falling back to basic RAG")
            return self.basic_rag.query(query, conversation_id, stream=False, **kwargs)

    def _query_with_agents(self, query: str, **kwargs) -> RAGResponse:
        """Query using LangGraph agent orchestration."""
        start_time = time.time()

        result = self.orchestrator.process_query(query, **kwargs)

        total_time = time.time() - start_time

        return RAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=query,
            tokens_used=result["metadata"].get("tokens_used", {
                "prompt": 0, "completion": 0, "total": 0
            }),
            latency=total_time,
            retrieval_time=result["metadata"].get("search_time", 0),
            generation_time=result["metadata"].get("synthesis_time", 0),
            context_used="",
            metadata={
                **result["metadata"],
                "mode": "langgraph-agent",
                "orchestrated": True
            }
        )

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent system status."""
        if not self.enable_agents:
            return {
                "enabled": False,
                "agents": [],
                "threshold": self.agent_threshold,
                "orchestration": "none"
            }

        registry = get_agent_registry()

        return {
            "enabled": True,
            "agents": registry.list_agents(),
            "threshold": self.agent_threshold,
            "agent_count": len(registry.list_agents()),
            "orchestration": "langgraph"
        }


def get_agent_rag_chain(
    collection_name: str = "documents",
    enable_agents: bool = True,
    agent_threshold: str = "moderate"
) -> AgentRAGChain:
    """Get agent-enhanced RAG chain instance."""
    return AgentRAGChain(
        collection_name=collection_name,
        enable_agents=enable_agents,
        agent_threshold=agent_threshold
    )
