"""
Prompt Templates Module
Manages reusable prompt templates for different use cases.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from string import Template

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplate:
    """A prompt template with variables."""
    name: str
    template: str
    variables: List[str]
    description: Optional[str] = None


class PromptTemplateManager:
    """
    Manages prompt templates for RAG system.
    
    Features:
    - Pre-defined templates for common tasks
    - Variable substitution
    - Template validation
    - Custom template support
    """
    
    # System prompts
    SYSTEM_PROMPTS = {
        "default": """You are a helpful AI assistant. Provide accurate, clear, and concise responses to user queries.""",
        
        "rag": """You are a helpful AI assistant. Use the provided context to answer questions accurately.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which parts of the context you used for your answer.""",
        
        "technical": """You are a technical documentation assistant. Provide precise, detailed explanations.
Use technical terminology appropriately and include examples when helpful.
Base your answers on the provided documentation context.""",
        
        "conversational": """You are a friendly AI assistant having a natural conversation.
Be helpful, engaging, and clear in your responses. Show personality while remaining professional.""",
        
        "summarizer": """You are an expert at summarizing documents. Create concise, accurate summaries.
Capture key points, main ideas, and important details from the provided content."""
    }
    
    # RAG prompts
    RAG_PROMPTS = {
        "qa": PromptTemplate(
            name="qa",
            template="""Context information is below:

$context

Based on the above context, answer the following question:
Question: $query

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so.""",
            variables=["context", "query"],
            description="Basic question-answering with context"
        ),
        
        "qa_with_citations": PromptTemplate(
            name="qa_with_citations",
            template="""Context information is below:

$context

Based on the above context, answer the following question:
Question: $query

Instructions:
1. Structure your response with clear markdown formatting:
   - Use ## for main headings
   - Use ### for subheadings
   - Use **bold** for key terms
   - Use bullet points (-) for lists
   - Use numbered lists (1., 2., 3.) for sequential steps

2. Organize complex answers into logical sections:
   - Overview (brief summary)
   - Key Points (bullet list)
   - Detailed Explanation (if needed)
   - Additional Context (if relevant)

3. **CRITICAL - Reference Guidelines:**
   - **Cite sources using [1], [2], etc. at the END of sentences**
   - **When available, mention PAGE NUMBERS and SECTIONS** in your citations
   - **Format precise references as: "Information here [1, Page 5, Section 2.3]"**
   - **For documents without pages, use: "Information here [2, Section: Introduction]"**
   - **Always check the context metadata for page/section information**
   - **Be as specific as possible - help users locate the exact information**

4. For simple questions, provide concise answers without excessive formatting

5. Be confident and comprehensive when the context supports it

Example format for complex topics:
## Topic Name

**Key Definition:** [Brief explanation] [1, Page 3]

### Main Components:
- **Component 1** - Description [1, Page 4, Section 2.1]
- **Component 2** - Description [2, Page 7-8]

### How It Works:
Step-by-step or detailed explanation with precise citations [3, Section: Implementation, Page 12]""",
            variables=["context", "query"],
            description="Q&A with structured markdown formatting and citations"
        ),
        
        "conversational_qa": PromptTemplate(
            name="conversational_qa",
            template="""Previous conversation:
$history

Context information:
$context

Current question: $query

Based on the conversation history and context, provide a helpful response.""",
            variables=["history", "context", "query"],
            description="Conversational Q&A with history"
        ),
        
        "multi_doc_qa": PromptTemplate(
            name="multi_doc_qa",
            template="""You have access to information from multiple documents:

$context

Question: $query

Instructions:
1. Synthesize information from all relevant documents
2. Compare and contrast different sources if applicable
3. Provide a comprehensive answer
4. Cite which documents you used""",
            variables=["context", "query"],
            description="Multi-document question answering"
        )
    }
    
    # Analysis prompts
    ANALYSIS_PROMPTS = {
        "summarize": PromptTemplate(
            name="summarize",
            template="""Summarize the following content:

$content

Provide a concise summary that captures the main points and key information.
Length: $length""",
            variables=["content", "length"],
            description="Document summarization"
        ),
        
        "extract_key_points": PromptTemplate(
            name="extract_key_points",
            template="""Extract the key points from the following content:

$content

List the most important points in a clear, organized manner.""",
            variables=["content"],
            description="Key point extraction"
        ),
        
        "compare": PromptTemplate(
            name="compare",
            template="""Compare the following two items:

Item 1: $item1

Item 2: $item2

Provide a detailed comparison highlighting similarities and differences.""",
            variables=["item1", "item2"],
            description="Comparison analysis"
        )
    }
    
    def __init__(self):
        """Initialize the template manager."""
        self.custom_templates: Dict[str, PromptTemplate] = {}
        logger.info("Prompt template manager initialized")
    
    def get_system_prompt(self, name: str = "default") -> str:
        """
        Get a system prompt by name.
        
        Args:
            name: System prompt name
            
        Returns:
            System prompt text
        """
        if name not in self.SYSTEM_PROMPTS:
            logger.warning(f"System prompt '{name}' not found, using default")
            name = "default"
        
        return self.SYSTEM_PROMPTS[name]
    
    def get_template(self, name: str, category: str = "rag") -> Optional[PromptTemplate]:
        """
        Get a prompt template.
        
        Args:
            name: Template name
            category: Template category ('rag', 'analysis')
            
        Returns:
            PromptTemplate or None
        """
        # Check custom templates first
        if name in self.custom_templates:
            return self.custom_templates[name]
        
        # Check predefined templates
        if category == "rag" and name in self.RAG_PROMPTS:
            return self.RAG_PROMPTS[name]
        elif category == "analysis" and name in self.ANALYSIS_PROMPTS:
            return self.ANALYSIS_PROMPTS[name]
        
        logger.warning(f"Template '{name}' not found in category '{category}'")
        return None
    
    def format_template(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any]
    ) -> str:
        """
        Format a template with variables.
        
        Args:
            template: PromptTemplate object
            variables: Dictionary of variable values
            
        Returns:
            Formatted prompt string
        """
        # Validate variables
        missing = set(template.variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        # Format template
        t = Template(template.template)
        try:
            return t.substitute(variables)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e}")
    
    def format_rag_prompt(
        self,
        query: str,
        context: str,
        template_name: str = "qa",
        history: Optional[str] = None
    ) -> str:
        """
        Convenience method to format RAG prompts.
        
        Args:
            query: User query
            context: Retrieved context
            template_name: Template to use
            history: Optional conversation history
            
        Returns:
            Formatted prompt
        """
        template = self.get_template(template_name, category="rag")
        if not template:
            # Fallback to basic template
            template = self.RAG_PROMPTS["qa"]
        
        variables = {
            "query": query,
            "context": context
        }
        
        if history and "history" in template.variables:
            variables["history"] = history
        
        return self.format_template(template, variables)
    
    def add_custom_template(
        self,
        name: str,
        template_str: str,
        variables: List[str],
        description: Optional[str] = None
    ) -> None:
        """
        Add a custom template.
        
        Args:
            name: Template name
            template_str: Template string with $variable placeholders
            variables: List of variable names
            description: Optional description
        """
        template = PromptTemplate(
            name=name,
            template=template_str,
            variables=variables,
            description=description
        )
        
        self.custom_templates[name] = template
        logger.info(f"Added custom template: {name}")
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of template names
        """
        templates = []
        
        if category is None or category == "rag":
            templates.extend(self.RAG_PROMPTS.keys())
        
        if category is None or category == "analysis":
            templates.extend(self.ANALYSIS_PROMPTS.keys())
        
        if category is None or category == "custom":
            templates.extend(self.custom_templates.keys())
        
        return templates
    
    def list_system_prompts(self) -> List[str]:
        """List available system prompts."""
        return list(self.SYSTEM_PROMPTS.keys())


# Singleton instance
_template_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get or create template manager singleton."""
    global _template_manager
    
    if _template_manager is None:
        _template_manager = PromptTemplateManager()
    
    return _template_manager


# Convenience functions

def format_rag_prompt(
    query: str,
    context: str,
    template: str = "qa_with_citations"
) -> str:
    """
    Format a RAG prompt quickly.
    
    Args:
        query: User query
        context: Retrieved context
        template: Template name
        
    Returns:
        Formatted prompt
    """
    manager = get_template_manager()
    return manager.format_rag_prompt(query, context, template)


def get_system_prompt(name: str = "rag") -> str:
    """
    Get a system prompt quickly.
    
    Args:
        name: System prompt name
        
    Returns:
        System prompt
    """
    manager = get_template_manager()
    return manager.get_system_prompt(name)


if __name__ == "__main__":
    # Test prompt templates
    print("Testing Prompt Templates\n" + "="*60)
    
    manager = PromptTemplateManager()
    
    # List available templates
    print("\nSystem Prompts:")
    for name in manager.list_system_prompts():
        print(f"  - {name}")
    
    print("\nRAG Templates:")
    for name in manager.list_templates("rag"):
        template = manager.get_template(name, "rag")
        print(f"  - {name}: {template.description}")
    
    # Test formatting
    print("\n" + "="*60)
    print("Test: Format RAG Prompt")
    print("="*60)
    
    context = """[1] Machine learning is a method of data analysis.
[2] It automates analytical model building.
[3] ML is a branch of artificial intelligence."""
    
    query = "What is machine learning?"
    
    prompt = manager.format_rag_prompt(query, context, "qa_with_citations")
    print(prompt)
    
    # Test custom template
    print("\n" + "="*60)
    print("Test: Custom Template")
    print("="*60)
    
    manager.add_custom_template(
        name="explain_like_5",
        template_str="Explain this concept in simple terms a 5-year-old would understand:\n\n$concept",
        variables=["concept"],
        description="ELI5 explanations"
    )
    
    custom_template = manager.get_template("explain_like_5", "custom")
    formatted = manager.format_template(custom_template, {"concept": "Quantum computing"})
    print(formatted)
    
    print("\nPrompt templates test completed!")
