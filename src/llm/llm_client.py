"""
LLM Client Module
Handles interaction with LLM providers via LangChain.
Supports streaming, multiple models, and error handling.
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import time
from enum import Enum

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
settings = get_settings()
perf_monitor = get_performance_monitor()


class LLMProvider(Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    provider: str
    tokens_used: Dict[str, int]
    latency: float
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _dict_to_langchain_messages(messages: List[Dict[str, str]]):
    """Convert dict messages to LangChain message objects."""
    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


class LLMClient:
    """
    Unified LLM client powered by LangChain.

    Features:
    - LangChain ChatGroq integration
    - Streaming responses
    - Token tracking
    - Error handling
    """

    MODELS = {
        LLMProvider.GROQ: [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        LLMProvider.OPENAI: [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None
    ):
        self.provider = LLMProvider(provider or settings.default_llm_provider)
        self.model = model or self._get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize LangChain chat model
        if self.provider == LLMProvider.GROQ:
            api_key = api_key or settings.groq_api_key
            if not api_key:
                raise ValueError("Groq API key not found")
            self.chat_model: BaseChatModel = ChatGroq(
                model=self.model,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info(f"Initialized LangChain ChatGroq with model: {self.model}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "errors": 0
        }

    def get_chat_model(self) -> BaseChatModel:
        """Get the underlying LangChain chat model for use in chains."""
        return self.chat_model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[LLMResponse, Iterator[str]]:
        """Generate response from LLM via LangChain."""
        logger.debug(
            f"Generating with {self.provider.value}/{self.model}, "
            f"messages={len(messages)}, stream={stream}"
        )

        start_time = time.time()

        try:
            if stream:
                return self._generate_stream(messages, temperature, max_tokens)
            else:
                return self._generate_complete(messages, temperature, max_tokens, start_time)
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    def _generate_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        start_time: float
    ) -> LLMResponse:
        """Generate complete (non-streaming) response via LangChain."""
        perf_monitor.start_timer("llm_generation")
        try:
            lc_messages = _dict_to_langchain_messages(messages)

            model = self.chat_model
            bind_kwargs = {}
            if temperature is not None and temperature != self.temperature:
                bind_kwargs["temperature"] = temperature
            if max_tokens is not None and max_tokens != self.max_tokens:
                bind_kwargs["max_tokens"] = max_tokens
            if bind_kwargs:
                model = model.bind(**bind_kwargs)

            response = model.invoke(lc_messages)

            content = response.content
            token_usage = response.response_metadata.get("token_usage", {})
            tokens_used = {
                "prompt": token_usage.get("prompt_tokens", 0),
                "completion": token_usage.get("completion_tokens", 0),
                "total": token_usage.get("total_tokens", 0),
            }
            finish_reason = response.response_metadata.get("finish_reason")

            latency = time.time() - start_time

            self.stats["total_requests"] += 1
            self.stats["total_tokens"] += tokens_used["total"]
            self.stats["total_latency"] += latency

            perf_monitor.increment_counter("llm_requests", 1)
            perf_monitor.increment_counter("llm_tokens", tokens_used["total"])

            logger.info(
                f"Generated response: {tokens_used['total']} tokens, "
                f"{latency:.2f}s"
            )
        finally:
            perf_monitor.stop_timer("llm_generation")

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            tokens_used=tokens_used,
            latency=latency,
            finish_reason=finish_reason
        )

    def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Iterator[str]:
        """Generate streaming response via LangChain."""
        logger.debug("Starting streaming generation")

        lc_messages = _dict_to_langchain_messages(messages)

        model = self.chat_model
        bind_kwargs = {}
        if temperature is not None and temperature != self.temperature:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None and max_tokens != self.max_tokens:
            bind_kwargs["max_tokens"] = max_tokens
        if bind_kwargs:
            model = model.bind(**bind_kwargs)

        for chunk in model.stream(lc_messages):
            if chunk.content:
                yield chunk.content

        self.stats["total_requests"] += 1
        perf_monitor.increment_counter("llm_requests", 1)

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[LLMResponse, Iterator[str]]:
        """Simple chat interface."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_message})

        return self.generate(messages, stream=stream)

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        return self.MODELS[self.provider][0]

    def get_available_models(self) -> List[str]:
        """Get list of available models for current provider."""
        return self.MODELS[self.provider]

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = self.stats.copy()
        if stats["total_requests"] > 0:
            stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
            stats["avg_tokens_per_request"] = stats["total_tokens"] / stats["total_requests"]
        return stats

    def count_tokens(self, text: str) -> int:
        """Approximate token count for text."""
        return len(text) // 4


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    force_reload: bool = False
) -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client

    if _llm_client is None or force_reload:
        _llm_client = LLMClient(provider=provider, model=model)

    return _llm_client
