"""
Base Agent Module
Provides abstract base class and interfaces for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class AgentType(Enum):
    """Types of agents in the system."""
    SEARCH = "search"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    TOOL = "tool"
    ORCHESTRATOR = "orchestrator"


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentResult:
    """Result from agent execution."""
    agent_name: str
    agent_type: AgentType
    status: AgentStatus
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "output": self.output,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "error": self.error
        }


@dataclass
class AgentTask:
    """Task to be executed by an agent."""
    task_id: str
    agent_type: AgentType
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    All agents must implement:
    - execute(): Main execution logic
    - validate_input(): Input validation
    """
    
    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        description: str = ""
    ):
        """
        Initialize base agent.
        
        Args:
            name: Agent name
            agent_type: Type of agent
            description: Agent description
        """
        self.name = name
        self.agent_type = agent_type
        self.description = description
        self.status = AgentStatus.PENDING
        
    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the agent's main logic.
        
        Args:
            task: Task to execute
            
        Returns:
            AgentResult with execution results
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def run(self, task: AgentTask) -> AgentResult:
        """
        Run the agent with error handling and timing.
        
        Args:
            task: Task to execute
            
        Returns:
            AgentResult with execution results
        """
        start_time = time.time()
        self.status = AgentStatus.RUNNING
        
        try:
            # Validate input
            if not self.validate_input(task.input_data):
                raise ValueError(f"Invalid input data for agent {self.name}")
            
            # Execute agent logic
            result = self.execute(task)
            result.execution_time = time.time() - start_time
            
            self.status = AgentStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=AgentStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "description": self.description,
            "status": self.status.value
        }


class AgentRegistry:
    """
    Registry for managing available agents.
    Allows dynamic agent registration and lookup.
    """
    
    def __init__(self):
        """Initialize agent registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._agents_by_type: Dict[AgentType, List[BaseAgent]] = {
            agent_type: [] for agent_type in AgentType
        }
    
    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent.
        
        Args:
            agent: Agent to register
        """
        self._agents[agent.name] = agent
        self._agents_by_type[agent.agent_type].append(agent)
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None
        """
        return self._agents.get(name)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """
        Get all agents of a specific type.
        
        Args:
            agent_type: Type of agents to retrieve
            
        Returns:
            List of agents
        """
        return self._agents_by_type.get(agent_type, [])
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all registered agents.
        
        Returns:
            List of agent capabilities
        """
        return [agent.get_capabilities() for agent in self._agents.values()]
    
    def unregister(self, name: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            name: Agent name
            
        Returns:
            True if unregistered, False if not found
        """
        agent = self._agents.pop(name, None)
        if agent:
            self._agents_by_type[agent.agent_type].remove(agent)
            return True
        return False


# Global agent registry instance
_agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get global agent registry instance."""
    return _agent_registry
