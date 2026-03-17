"""
Agent System Module
LangGraph-powered multi-agent orchestration for complex query handling.

This module provides:
- Base agent abstractions
- Specialized agents (Search, Analysis, Synthesis)
- LangGraph StateGraph orchestration
- Agent-enhanced RAG pipeline
"""

from src.agents.base import (
    BaseAgent,
    AgentTask,
    AgentResult,
    AgentType,
    AgentStatus,
    AgentRegistry,
    get_agent_registry
)

from src.agents.orchestrator import (
    AgentOrchestrator,
    AgentState,
    QueryIntent,
    get_orchestrator
)

from src.agents.search_agent import SearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.synthesis_agent import SynthesisAgent

from src.agents.agent_rag_chain import (
    AgentRAGChain,
    get_agent_rag_chain
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentTask",
    "AgentResult",
    "AgentType",
    "AgentStatus",
    "AgentRegistry",
    "get_agent_registry",
    
    # Orchestration
    "AgentOrchestrator",
    "QueryIntent",
    "get_orchestrator",
    
    # Specialized agents
    "SearchAgent",
    "AnalysisAgent",
    "SynthesisAgent",
    
    # Agent-enhanced RAG
    "AgentRAGChain",
    "get_agent_rag_chain",
]
