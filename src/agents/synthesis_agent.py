"""
Synthesis Agent Module
Specialized agent for multi-source synthesis and comprehensive response generation.
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent, AgentTask, AgentResult, AgentType, AgentStatus
from src.llm.llm_client import get_llm_client
from src.llm.prompt_templates import get_template_manager
from src.retrieval.context_builder import ContextBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent for comprehensive response generation.
    
    Capabilities:
    - Multi-source synthesis
    - Structured response formatting
    - Citation management
    - Quality enhancement
    """
    
    def __init__(self):
        """Initialize synthesis agent."""
        super().__init__(
            name="SynthesisAgent",
            agent_type=AgentType.SYNTHESIS,
            description="Multi-source synthesis and response generation"
        )
        
        self.llm_client = get_llm_client()
        self.template_manager = get_template_manager()
        self.context_builder = ContextBuilder(max_tokens=4000)
        
        logger.info("SynthesisAgent initialized")
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Must contain 'query' and 'dependency_results'
            
        Returns:
            True if valid
        """
        return (
            "query" in input_data and 
            "dependency_results" in input_data
        )
    
    def build_synthesis_prompt(
        self,
        query: str,
        analysis: str,
        key_points: List[str],
        insights: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Build comprehensive synthesis prompt.
        
        Args:
            query: Original query
            analysis: Analysis summary
            key_points: Extracted key points
            insights: Generated insights
            sources: Source documents
            
        Returns:
            Synthesis prompt
        """
        # Build context from sources
        context = self.context_builder.build_context(sources)
        
        prompt = f"""You are synthesizing information from multiple sources to answer a query.

Query: {query}

Analysis Summary:
{analysis}

Key Points:
{chr(10).join(f"- {point}" for point in key_points)}

Analytical Insights:
{insights}

Source Context:
{context}

Instructions:
1. Provide a comprehensive, well-structured answer using markdown formatting
2. Use ## for main sections, ### for subsections
3. Include **bold** for key terms
4. Cite sources using [1], [2], etc. at the END of sentences
5. Synthesize information from analysis, key points, and insights
6. Be confident and thorough

Generate a complete, structured answer:"""

        return prompt
    
    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute synthesis.
        
        Args:
            task: Synthesis task
            
        Returns:
            AgentResult with synthesized response
        """
        query = task.input_data["query"]
        dependency_results = task.input_data["dependency_results"]
        
        logger.info(f"SynthesisAgent executing: query='{query[:50]}...'")
        
        # Extract analysis results
        analysis_results = [
            result for result in dependency_results.values()
            if result.agent_type == AgentType.ANALYSIS
        ]
        
        if not analysis_results:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=AgentStatus.FAILED,
                output=None,
                error="No analysis results available for synthesis"
            )
        
        analysis_output = analysis_results[0].output
        analysis = analysis_output.get("analysis", "")
        key_points = analysis_output.get("key_points", [])
        insights = analysis_output.get("insights", "")
        sources = analysis_output.get("sources", [])
        
        # Build synthesis prompt
        prompt = self.build_synthesis_prompt(
            query, analysis, key_points, insights, sources
        )
        
        # Generate synthesis
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(
            messages,
            temperature=0.7,
            max_tokens=1500
        )
        
        answer = response.content.strip()
        
        logger.info(
            f"SynthesisAgent completed: {response.tokens_used['total']} tokens, "
            f"{len(sources)} sources"
        )
        
        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=AgentStatus.COMPLETED,
            output={
                "answer": answer,
                "sources": sources,
                "analysis_used": True,
                "key_points": key_points
            },
            metadata={
                "tokens_used": response.tokens_used,
                "source_count": len(sources),
                "synthesis_approach": "multi-agent",
                "llm_model": response.model
            }
        )
