"""
LLM Module
Handles LLM interactions, prompts, conversations, and RAG chain.
"""

from src.llm.llm_client import (
    LLMClient,
    LLMProvider,
    LLMResponse,
    get_llm_client
)
from src.llm.prompt_templates import (
    PromptTemplateManager,
    get_template_manager,
    format_rag_prompt,
    get_system_prompt
)
from src.llm.conversation import (
    ConversationManager,
    Conversation,
    Message,
    get_conversation_manager
)
from src.llm.rag_chain import (
    RAGChain,
    RAGResponse,
    ask
)

__all__ = [
    "LLMClient",
    "LLMProvider",
    "LLMResponse",
    "get_llm_client",
    "PromptTemplateManager",
    "get_template_manager",
    "format_rag_prompt",
    "get_system_prompt",
    "ConversationManager",
    "Conversation",
    "Message",
    "get_conversation_manager",
    "RAGChain",
    "RAGResponse",
    "ask"
]

__version__ = "1.0.0"
