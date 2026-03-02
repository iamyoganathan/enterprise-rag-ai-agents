"""
Analysis Agent Module
Specialized agent for deep content analysis and insight extraction.
"""

from typing import Dict, Any, List
from src.agents.base import BaseAgent, AgentTask, AgentResult, AgentType, AgentStatus
from src.llm.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent for deep content understanding.
    
    Capabilities:
    - Content analysis and summarization
    - Key point extraction
    - Pattern identification
    - Insight generation
    """
    
    def __init__(self):
        """Initialize analysis agent."""
        super().__init__(
            name="AnalysisAgent",
            agent_type=AgentType.ANALYSIS,
            description="Deep content analysis and insight extraction"
        )
        
        self.llm_client = get_llm_client()
        logger.info("AnalysisAgent initialized")
    
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
    
    def extract_key_points(self, content: str, query: str) -> List[str]:
        """
        Extract key points from content.
        
        Args:
            content: Content to analyze
            query: Original query for context
            
        Returns:
            List of key points
        """
        prompt = f"""Analyze the following content and extract 3-5 key points related to the query.

Query: {query}

Content:
{content[:3000]}

Extract the most important points as a bullet list."""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(messages, temperature=0.3, max_tokens=500)
        
        # Parse bullet points
        lines = response.content.strip().split('\n')
        key_points = [
            line.strip('- •*').strip() 
            for line in lines 
            if line.strip() and any(c in line for c in '-•*')
        ]
        
        return key_points[:5]
    
    def generate_insights(
        self,
        key_points: List[str],
        query: str
    ) -> str:
        """
        Generate insights from key points.
        
        Args:
            key_points: Extracted key points
            query: Original query
            
        Returns:
            Generated insights
        """
        points_text = '\n'.join(f"- {point}" for point in key_points)
        
        prompt = f"""Based on these key points, provide analytical insights for the query.

Query: {query}

Key Points:
{points_text}

Provide 2-3 sentences of analytical insights that connect these points to answer the query."""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.generate(messages, temperature=0.5, max_tokens=300)
        
        return response.content.strip()
    
    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute content analysis.
        
        Args:
            task: Analysis task
            
        Returns:
            AgentResult with analysis
        """
        query = task.input_data["query"]
        dependency_results = task.input_data["dependency_results"]
        
        logger.info(f"AnalysisAgent executing: query='{query[:50]}...'")
        
        # Get sources from search agent
        search_results = list(dependency_results.values())
        if not search_results:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=AgentStatus.FAILED,
                output=None,
                error="No search results available for analysis"
            )
        
        search_output = search_results[0].output
        sources = search_output.get("sources", [])
        
        if not sources:
            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=AgentStatus.COMPLETED,
                output={
                    "analysis": "No content available to analyze.",
                    "key_points": [],
                    "insights": "",
                    "sources": []
                },
                metadata={"source_count": 0}
            )
        
        # Combine content from sources
        combined_content = "\n\n".join([
            source.get("content", "") 
            for source in sources[:5]  # Analyze top 5 sources
        ])
        
        # Extract key points
        key_points = self.extract_key_points(combined_content, query)
        
        # Generate insights
        insights = self.generate_insights(key_points, query)
        
        # Create analysis summary
        analysis = f"""## Analysis Summary

**Key Points:**
{chr(10).join(f"- {point}" for point in key_points)}

**Insights:**
{insights}"""
        
        logger.info(f"AnalysisAgent completed: {len(key_points)} key points extracted")
        
        return AgentResult(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=AgentStatus.COMPLETED,
            output={
                "analysis": analysis,
                "key_points": key_points,
                "insights": insights,
                "sources": sources
            },
            metadata={
                "source_count": len(sources),
                "analyzed_sources": min(5, len(sources)),
                "key_point_count": len(key_points)
            }
        )
