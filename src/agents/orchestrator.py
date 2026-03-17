"""
LangGraph Agent Orchestrator Module
Coordinates multiple agents using LangGraph StateGraph.
"""

from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator

from langgraph.graph import StateGraph, END

from src.agents.base import (
    AgentType, AgentStatus, get_agent_registry
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    """State that flows through the LangGraph agent pipeline."""
    query: str
    intent: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    synthesis: Dict[str, Any]
    final_answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    errors: Annotated[List[str], operator.add]


@dataclass
class QueryIntent:
    """Detected intent from user query."""
    primary_intent: str
    complexity: str
    requires_agents: List[AgentType]
    confidence: float
    reasoning: str


class AgentOrchestrator:
    """
    LangGraph-powered orchestrator for multi-agent RAG.

    Builds a StateGraph with nodes:
      search_node -> analysis_node -> synthesis_node
    Uses conditional edges to route based on query complexity.
    """

    def __init__(self):
        """Initialize orchestrator and build the graph."""
        self.agent_registry = get_agent_registry()
        self._graph = None
        logger.info("LangGraph agent orchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        # Add nodes (prefixed to avoid conflict with state keys)
        graph.add_node("search_node", self._search_node)
        graph.add_node("analysis_node", self._analysis_node)
        graph.add_node("synthesis_node", self._synthesis_node)

        # Define edges: search -> analysis -> synthesis -> END
        graph.set_entry_point("search_node")
        graph.add_edge("search_node", "analysis_node")
        graph.add_edge("analysis_node", "synthesis_node")
        graph.add_edge("synthesis_node", END)

        return graph.compile()

    @property
    def graph(self):
        """Lazy-build the compiled graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _search_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node: execute SearchAgent."""
        from src.agents.search_agent import SearchAgent
        from src.agents.base import AgentTask
        import uuid

        query = state["query"]
        logger.info(f"[LangGraph] Search node executing: '{query[:50]}...'")

        agents = self.agent_registry.get_agents_by_type(AgentType.SEARCH)
        if not agents:
            return {"search_results": [], "errors": [f"No SearchAgent registered"]}

        agent = agents[0]
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.SEARCH,
            input_data={"query": query, "enhanced": True},
            dependencies=[],
            metadata={"priority": 1}
        )

        result = agent.run(task)

        if result.status == AgentStatus.COMPLETED:
            sources = result.output.get("sources", [])
            return {
                "search_results": sources,
                "metadata": {
                    **state.get("metadata", {}),
                    "search_time": result.execution_time,
                    "search_count": len(sources)
                }
            }
        else:
            return {"search_results": [], "errors": [f"Search failed: {result.error}"]}

    def _analysis_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node: execute AnalysisAgent."""
        from src.agents.analysis_agent import AnalysisAgent
        from src.agents.base import AgentTask, AgentResult
        import uuid

        query = state["query"]
        search_results = state.get("search_results", [])
        logger.info(f"[LangGraph] Analysis node executing with {len(search_results)} sources")

        if not search_results:
            return {
                "analysis": {"analysis": "", "key_points": [], "insights": ""},
                "errors": ["No search results for analysis"]
            }

        agents = self.agent_registry.get_agents_by_type(AgentType.ANALYSIS)
        if not agents:
            return {"analysis": {}, "errors": ["No AnalysisAgent registered"]}

        agent = agents[0]

        # Build dependency results in the format AnalysisAgent expects
        mock_search_result = AgentResult(
            agent_name="SearchAgent",
            agent_type=AgentType.SEARCH,
            status=AgentStatus.COMPLETED,
            output={"sources": search_results, "query": query, "count": len(search_results)}
        )

        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.ANALYSIS,
            input_data={
                "query": query,
                "dependency_results": {"search": mock_search_result}
            },
            dependencies=[],
            metadata={"priority": 2}
        )

        result = agent.run(task)

        if result.status == AgentStatus.COMPLETED:
            return {
                "analysis": result.output,
                "metadata": {
                    **state.get("metadata", {}),
                    "analysis_time": result.execution_time
                }
            }
        else:
            return {"analysis": {}, "errors": [f"Analysis failed: {result.error}"]}

    def _synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node: execute SynthesisAgent."""
        from src.agents.synthesis_agent import SynthesisAgent
        from src.agents.base import AgentTask, AgentResult
        import uuid

        query = state["query"]
        analysis = state.get("analysis", {})
        search_results = state.get("search_results", [])
        logger.info(f"[LangGraph] Synthesis node executing")

        agents = self.agent_registry.get_agents_by_type(AgentType.SYNTHESIS)
        if not agents:
            return {"final_answer": "", "sources": [], "errors": ["No SynthesisAgent registered"]}

        agent = agents[0]

        # Build dependency results in the format SynthesisAgent expects
        mock_analysis_result = AgentResult(
            agent_name="AnalysisAgent",
            agent_type=AgentType.ANALYSIS,
            status=AgentStatus.COMPLETED,
            output={
                "analysis": analysis.get("analysis", ""),
                "key_points": analysis.get("key_points", []),
                "insights": analysis.get("insights", ""),
                "sources": search_results
            }
        )

        task = AgentTask(
            task_id=str(uuid.uuid4()),
            agent_type=AgentType.SYNTHESIS,
            input_data={
                "query": query,
                "dependency_results": {"analysis": mock_analysis_result}
            },
            dependencies=[],
            metadata={"priority": 3}
        )

        result = agent.run(task)

        if result.status == AgentStatus.COMPLETED:
            return {
                "final_answer": result.output.get("answer", ""),
                "sources": result.output.get("sources", []),
                "metadata": {
                    **state.get("metadata", {}),
                    "synthesis_time": result.execution_time,
                    "tokens_used": result.metadata.get("tokens_used", {}),
                    "orchestration": "langgraph"
                }
            }
        else:
            return {"final_answer": "", "sources": [], "errors": [f"Synthesis failed: {result.error}"]}

    def analyze_intent(self, query: str) -> QueryIntent:
        """Analyze query to determine intent and required agents."""
        query_lower = query.lower()

        search_keywords = ["find", "search", "retrieve", "get", "show me"]
        analyze_keywords = ["analyze", "explain", "why", "how", "breakdown"]
        synthesize_keywords = ["compare", "summarize", "combine", "overall"]

        search_score = sum(1 for kw in search_keywords if kw in query_lower)
        analyze_score = sum(1 for kw in analyze_keywords if kw in query_lower)
        synthesize_score = sum(1 for kw in synthesize_keywords if kw in query_lower)

        scores = {
            "search": search_score,
            "analyze": analyze_score,
            "synthesize": synthesize_score
        }
        primary_intent = max(scores, key=scores.get)

        word_count = len(query.split())
        has_multiple_intents = sum(1 for s in scores.values() if s > 0) > 1

        if word_count < 5 and not has_multiple_intents:
            complexity = "simple"
            required_agents = []
        elif word_count < 15 or not has_multiple_intents:
            complexity = "moderate"
            required_agents = [AgentType.SEARCH]
        else:
            complexity = "complex"
            required_agents = [AgentType.SEARCH, AgentType.ANALYSIS, AgentType.SYNTHESIS]

        max_score = max(scores.values())
        confidence = max_score / (word_count + 1) if word_count > 0 else 0.5

        return QueryIntent(
            primary_intent=primary_intent,
            complexity=complexity,
            requires_agents=required_agents,
            confidence=min(confidence, 1.0),
            reasoning=f"Detected {primary_intent} with {complexity} complexity"
        )

    def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process query through the LangGraph multi-agent pipeline.

        Runs: search_node -> analysis_node -> synthesis_node
        """
        logger.info(f"[LangGraph] Processing query: '{query[:50]}...'")

        initial_state: AgentState = {
            "query": query,
            "intent": {},
            "search_results": [],
            "analysis": {},
            "synthesis": {},
            "final_answer": "",
            "sources": [],
            "metadata": {},
            "errors": []
        }

        # Run the LangGraph
        final_state = self.graph.invoke(initial_state)

        answer = final_state.get("final_answer", "")
        sources = final_state.get("sources", [])
        metadata = final_state.get("metadata", {})
        errors = final_state.get("errors", [])

        if errors:
            logger.warning(f"[LangGraph] Errors during execution: {errors}")

        if not answer and errors:
            answer = "The agent pipeline encountered errors. Please try again."

        logger.info(f"[LangGraph] Query complete: {len(sources)} sources")

        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                **metadata,
                "errors": errors,
                "orchestration": "langgraph"
            }
        }


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
