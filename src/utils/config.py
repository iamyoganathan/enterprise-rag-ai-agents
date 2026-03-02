"""
Configuration management module.
Handles environment variables and application settings using Pydantic.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    # LangSmith Monitoring
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langchain_project: str = Field(default="enterprise-rag-system", description="LangSmith project name")
    
    # Model Configuration
    default_llm_provider: str = Field(default="groq", description="Default LLM provider")
    default_llm_model: str = Field(default="llama-3.3-70b-versatile", description="Default LLM model")
    embedding_model: str = Field(default="all-mpnet-base-v2", description="Embedding model name")
    max_tokens: int = Field(default=8192, description="Maximum tokens for LLM")
    temperature: float = Field(default=0.7, description="LLM temperature")
    
    # Vector Database
    vector_db_type: str = Field(default="chromadb", description="Vector database type")
    vector_db_path: str = Field(default="./data/vector_db", description="Vector DB path")
    collection_name: str = Field(default="documents", description="Collection name")
    
    # Document Processing
    chunk_size: int = Field(default=1024, description="Text chunk size")
    chunk_overlap: int = Field(default=128, description="Chunk overlap")
    max_chunks_per_doc: int = Field(default=1000, description="Max chunks per document")
    
    # Retrieval Configuration
    top_k_results: int = Field(default=30, description="Initial retrieval count")
    rerank_top_k: int = Field(default=15, description="Final re-ranked results")
    similarity_threshold: float = Field(default=0.0, description="Similarity threshold")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins"
    )
    
    # Security
    secret_key: str = Field(default="change_this_secret_key", description="Secret key for JWT")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Token expiry minutes")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=30, description="Requests per minute")
    rate_limit_per_hour: int = Field(default=500, description="Requests per hour")
    
    # Caching
    enable_cache: bool = Field(default=True, description="Enable caching")
    cache_type: str = Field(default="disk", description="Cache type (disk/redis/memory)")
    cache_dir: str = Field(default="./data/cache", description="Cache directory")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    redis_url: Optional[str] = Field(default=None, description="Redis URL")
    
    # Frontend
    streamlit_port: int = Field(default=8501, description="Streamlit port")
    streamlit_server_address: str = Field(default="localhost", description="Streamlit address")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="./logs/app.log", description="Log file path")
    
    # Development
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    @property
    def debug_mode(self) -> bool:
        """Alias for debug (backwards compatibility)."""
        return self.debug
    
    # Performance
    max_concurrent_requests: int = Field(default=10, description="Max concurrent requests")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    # File Upload
    max_upload_size_mb: int = Field(default=50, description="Max upload size in MB")
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt", ".md"],
        description="Allowed file extensions"
    )
    
    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration."""
        return {
            "provider": self.default_llm_provider,
            "model": self.default_llm_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def get_vector_db_config(self) -> dict:
        """Get vector database configuration."""
        return {
            "type": self.vector_db_type,
            "path": self.vector_db_path,
            "collection": self.collection_name
        }
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            Path(self.vector_db_path).parent,
            Path(self.cache_dir),
            Path(self.log_file).parent,
            Path("data/raw"),
            Path("data/processed"),
            Path("data/sample_documents")
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # Log but don't crash - directory might be created later
                print(f"Warning: Could not create directory {directory}: {e}")


# Global settings instance
settings = Settings()

# Ensure directories exist on import (non-critical, will retry at startup)
try:
    settings.ensure_directories()
except Exception as e:
    print(f"Warning: Directory creation deferred due to: {e}")


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


if __name__ == "__main__":
    # Test configuration
    config = get_settings()
    print("Configuration loaded successfully!")
    print(f"LLM Provider: {config.default_llm_provider}")
    print(f"LLM Model: {config.default_llm_model}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Vector DB Type: {config.vector_db_type}")
    print(f"Cache Enabled: {config.enable_cache}")
    print(f"Log Level: {config.log_level}")
