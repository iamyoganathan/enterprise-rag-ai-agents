"""
Agent Orchestrator Module
Coordinates multiple agents to handle complex queries.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from src.agents.base import (
    BaseAgent, AgentTask, AgentResult, AgentType, 
    AgentStatus, get_agent_registry
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryIntent:
    """Detected intent from user query."""
    primary_intent: str  # search, analyze, synthesize, compare
    complexity: str  # simple, moderate, complex
    requires_agents: List[AgentType]
    confidence: float
    reasoning: str


class AgentOrchestrator:
    """
    Orchestrates multiple agents to handle complex queries.
    
    Workflow:
    1. Analyze query intent
    2. Plan execution strategy
    3. Create tasks for agents
    4. Execute tasks (with dependencies)
    5. Aggregate results
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.agent_registry = get_agent_registry()
        logger.info("Agent orchestrator initialized")
    
    def analyze_intent(self, query: str) -> QueryIntent:
        """
        Analyze query to determine intent and required agents.
        
        Args:
            query: User query
            
        Returns:
            QueryIntent with detected intent
        """
        query_lower = query.lower()
        
        # Detect keywords for intent
        search_keywords = ["find", "search", "retrieve", "get", "show me"]
        analyze_keywords = ["analyze", "explain", "why", "how", "breakdown"]
        synthesize_keywords = ["compare", "summarize", "combine", "overall"]
        
        # Count keyword matches
        search_score = sum(1 for kw in search_keywords if kw in query_lower)
        analyze_score = sum(1 for kw in analyze_keywords if kw in query_lower)
        synthesize_score = sum(1 for kw in synthesize_keywords if kw in query_lower)
        
        # Determine primary intent
        scores = {
            "search": search_score,
            "analyze": analyze_score,
            "synthesize": synthesize_score
        }
        primary_intent = max(scores, key=scores.get)
        
        # Determine complexity
        word_count = len(query.split())
        has_multiple_intents = sum(1 for s in scores.values() if s > 0) > 1
        
        if word_count < 5 and not has_multiple_intents:
            complexity = "simple"
            required_agents = []  # Use basic RAG
        elif word_count < 15 or not has_multiple_intents:
            complexity = "moderate"
            required_agents = [AgentType.SEARCH]
        else:
            complexity = "complex"
            required_agents = [AgentType.SEARCH, AgentType.ANALYSIS, AgentType.SYNTHESIS]
        
        # Calculate confidence
        max_score = max(scores.values())
        confidence = max_score / (word_count + 1) if word_count > 0 else 0.5
        
        return QueryIntent(
            primary_intent=primary_intent,
            complexity=complexity,
            requires_agents=required_agents,
            confidence=min(confidence, 1.0),
            reasoning=f"Detected {primary_intent} with {complexity} complexity"
        )
    
    def plan_execution(
        self,
        query: str,
        intent: QueryIntent
    ) -> List[AgentTask]:
        """
        Plan task execution based on intent.
        
        Args:
            query: User query
            intent: Detected intent
            
        Returns:
            List of tasks to execute
        """
        tasks = []
        
        # Simple queries - no agents needed
        if intent.complexity == "simple":
            logger.info("Simple query - using basic RAG")
            return tasks
        
        # Moderate queries - search agent only
        if intent.complexity == "moderate":
            tasks.append(AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type=AgentType.SEARCH,
                input_data={"query": query, "intent": intent.primary_intent},
                dependencies=[],
                metadata={"priority": 1}
            ))
            return tasks
        
        # Complex queries - multi-agent workflow
        if intent.complexity == "complex":
            # Task 1: Enhanced search
            search_task_id = str(uuid.uuid4())
            tasks.append(AgentTask(
                task_id=search_task_id,
                agent_type=AgentType.SEARCH,
                input_data={"query": query, "enhanced": True},
                dependencies=[],
                metadata={"priority": 1}
            ))
            
            # Task 2: Analysis (depends on search)
            analysis_task_id = str(uuid.uuid4())
            tasks.append(AgentTask(
                task_id=analysis_task_id,
                agent_type=AgentType.ANALYSIS,
                input_data={"query": query},
                dependencies=[search_task_id],
                metadata={"priority": 2}
            ))
            
            # Task 3: Synthesis (depends on analysis)
            tasks.append(AgentTask(
                task_id=str(uuid.uuid4()),
                agent_type=AgentType.SYNTHESIS,
                input_data={"query": query},
                dependencies=[analysis_task_id],
                metadata={"priority": 3}
            ))
        
        logger.info(f"Planned {len(tasks)} tasks for query execution")
        return tasks
    
    def execute_tasks(
        self,
        tasks: List[AgentTask]
    ) -> Dict[str, AgentResult]:
        """
        Execute tasks respecting dependencies.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Dictionary mapping task_id to AgentResult
        """
        results: Dict[str, AgentResult] = {}
        completed_tasks = set()
        
        # Sort tasks by priority
        tasks = sorted(tasks, key=lambda t: t.metadata.get("priority", 0))
        
        for task in tasks:
            # Check if dependencies are met
            dependencies_met = all(
                dep_id in completed_tasks 
                for dep_id in task.dependencies
            )
            
            if not dependencies_met:
                logger.warning(
                    f"Dependencies not met for task {task.task_id}, skipping"
                )
                continue
            
            # Get appropriate agent
            agents = self.agent_registry.get_agents_by_type(task.agent_type)
            if not agents:
                logger.error(f"No agent found for type {task.agent_type}")
                results[task.task_id] = AgentResult(
                    agent_name="unknown",
                    agent_type=task.agent_type,
                    status=AgentStatus.FAILED,
                    output=None,
                    error=f"No agent available for {task.agent_type}"
                )
                continue
            
            # Use first available agent of this type
            agent = agents[0]
            
            # Add results from dependencies to task input
            task.input_data["dependency_results"] = {
                dep_id: results[dep_id] 
                for dep_id in task.dependencies 
                if dep_id in results
            }
            
            # Execute agent
            logger.info(f"Executing agent {agent.name} for task {task.task_id}")
            result = agent.run(task)
            results[task.task_id] = result
            
            if result.status == AgentStatus.COMPLETED:
                completed_tasks.add(task.task_id)
        
        return results
    
    def aggregate_results(
        self,
        results: Dict[str, AgentResult],
        query: str
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents.
        
        Args:
            results: Dictionary of agent results
            query: Original query
            
        Returns:
            Aggregated response
        """
        # Extract successful results
        successful_results = [
            r for r in results.values() 
            if r.status == AgentStatus.COMPLETED
        ]
        
        if not successful_results:
            return {
                "answer": "No results were successfully generated by agents.",
                "sources": [],
                "metadata": {
                    "agent_count": 0,
                    "total_time": 0,
                    "query": query
                }
            }
        
        # Get final result (highest priority task)
        final_result = successful_results[-1]
        
        # Collect all sources
        all_sources = []
        for result in successful_results:
            if isinstance(result.output, dict):
                sources = result.output.get("sources", [])
                if isinstance(sources, list):
                    all_sources.extend(sources)
        
        # Aggregate metadata
        total_time = sum(r.execution_time for r in results.values())
        agent_details = [
            {
                "agent": r.agent_name,
                "type": r.agent_type.value,
                "status": r.status.value,
                "time": r.execution_time
            }
            for r in results.values()
        ]
        
        return {
            "answer": final_result.output.get("answer", str(final_result.output)),
            "sources": all_sources,
            "metadata": {
                "agent_count": len(results),
                "successful_agents": len(successful_results),
                "total_time": total_time,
                "query": query,
                "agents": agent_details,
                "orchestration": "multi-agent"
            }
        }
    
    def process_query(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query through agent orchestration.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Aggregated response from agents
        """
        logger.info(f"Orchestrator processing query: {query[:50]}...")
        
        # Step 1: Analyze intent
        intent = self.analyze_intent(query)
        logger.info(
            f"Intent: {intent.primary_intent}, "
            f"Complexity: {intent.complexity}, "
            f"Confidence: {intent.confidence:.2f}"
        )
        
        # Step 2: Plan execution
        tasks = self.plan_execution(query, intent)
        
        # Step 3: Execute tasks
        results = self.execute_tasks(tasks)
        
        # Step 4: Aggregate results
        final_response = self.aggregate_results(results, query)
        final_response["metadata"]["intent"] = {
            "primary": intent.primary_intent,
            "complexity": intent.complexity,
            "confidence": intent.confidence,
            "reasoning": intent.reasoning
        }
        
        logger.info(
            f"Orchestration complete: {len(results)} agents, "
            f"{final_response['metadata']['total_time']:.2f}s"
        )
        
        return final_response


# Global orchestrator instance
_orchestrator = None


def get_orchestrator() -> AgentOrchestrator:
    """Get global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
