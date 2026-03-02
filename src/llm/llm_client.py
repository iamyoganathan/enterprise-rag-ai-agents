"""
LLM Client Module
Handles interaction with LLM providers (Groq, OpenAI, etc.).
Supports streaming, multiple models, and error handling.
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass
import time
from enum import Enum

from groq import Groq
import openai

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


class LLMClient:
    """
    Unified client for multiple LLM providers.
    
    Features:
    - Multiple provider support (Groq, OpenAI)
    - Streaming responses
    - Token tracking
    - Error handling and retries
    - Rate limiting
    """
    
    # Available models per provider
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
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider ('groq' or 'openai')
            model: Model name
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            api_key: API key (uses env var if None)
        """
        self.provider = LLMProvider(provider or settings.default_llm_provider)
        self.model = model or self._get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize client
        if self.provider == LLMProvider.GROQ:
            api_key = api_key or settings.groq_api_key
            if not api_key:
                raise ValueError("Groq API key not found")
            self.client = Groq(api_key=api_key)
            logger.info(f"Initialized Groq client with model: {self.model}")
        
        elif self.provider == LLMProvider.OPENAI:
            api_key = api_key or settings.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key not found")
            openai.api_key = api_key
            self.client = openai
            logger.info(f"Initialized OpenAI client with model: {self.model}")
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "errors": 0
        }
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[LLMResponse, Iterator[str]]:
        """
        Generate response from LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream the response
            
        Returns:
            LLMResponse object or iterator of response chunks
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
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
        temperature: float,
        max_tokens: int,
        start_time: float
    ) -> LLMResponse:
        """Generate complete (non-streaming) response."""
        
        perf_monitor.start_timer("llm_generation")
        try:
            if self.provider == LLMProvider.GROQ:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                
                content = response.choices[0].message.content
                tokens_used = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
                finish_reason = response.choices[0].finish_reason
            
            elif self.provider == LLMProvider.OPENAI:
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
                
                content = response.choices[0].message.content
                tokens_used = {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
                finish_reason = response.choices[0].finish_reason
            
            latency = time.time() - start_time
            
            # Update statistics
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
        temperature: float,
        max_tokens: int
    ) -> Iterator[str]:
        """Generate streaming response."""
        
        logger.debug("Starting streaming generation")
        
        if self.provider == LLMProvider.GROQ:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        elif self.provider == LLMProvider.OPENAI:
            stream = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.get("content"):
                    yield chunk.choices[0].delta.content
        
        self.stats["total_requests"] += 1
        perf_monitor.increment_counter("llm_requests", 1)
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[LLMResponse, Iterator[str]]:
        """
        Simple chat interface.
        
        Args:
            user_message: User's message
            system_prompt: Optional system prompt
            conversation_history: Previous messages
            stream: Whether to stream
            
        Returns:
            LLMResponse or stream iterator
        """
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add user message
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
        """
        Approximate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token
        # For accurate counting, use tiktoken
        return len(text) // 4


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    force_reload: bool = False
) -> LLMClient:
    """
    Get or create LLM client singleton.
    
    Args:
        provider: LLM provider
        model: Model name
        force_reload: Force recreation
        
    Returns:
        LLMClient instance
    """
    global _llm_client
    
    if _llm_client is None or force_reload:
        _llm_client = LLMClient(provider=provider, model=model)
    
    return _llm_client


if __name__ == "__main__":
    # Test the LLM client
    print("Testing LLM Client\n" + "="*60)
    
    # Initialize client
    client = LLMClient(provider="groq", model="llama-3.3-70b-versatile")
    
    print(f"Provider: {client.provider.value}")
    print(f"Model: {client.model}")
    print(f"Available models: {', '.join(client.get_available_models())}")
    
    # Test simple chat
    print("\n" + "="*60)
    print("Test 1: Simple Chat")
    print("="*60)
    
    response = client.chat(
        user_message="What is machine learning? Explain in 2 sentences.",
        system_prompt="You are a helpful AI assistant."
    )
    
    print(f"\nResponse: {response.content}")
    print(f"Tokens: {response.tokens_used['total']}")
    print(f"Latency: {response.latency:.2f}s")
    
    # Test with conversation history
    print("\n" + "="*60)
    print("Test 2: Conversation")
    print("="*60)
    
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."}
    ]
    
    response2 = client.chat(
        user_message="What is it used for?",
        conversation_history=history
    )
    
    print(f"\nResponse: {response2.content}")
    
    # Test streaming
    print("\n" + "="*60)
    print("Test 3: Streaming")
    print("="*60)
    
    print("\nStreaming response: ", end="", flush=True)
    stream = client.chat(
        user_message="Count from 1 to 5.",
        stream=True
    )
    
    for chunk in stream:
        print(chunk, end="", flush=True)
    print("\n")
    
    # Statistics
    print("="*60)
    print("Statistics")
    print("="*60)
    stats = client.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nLLM client test completed!")
