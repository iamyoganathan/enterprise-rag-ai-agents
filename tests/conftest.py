"""
Test Configuration
pytest configuration and fixtures.
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Settings


@pytest.fixture
def test_settings():
    """Provide test settings."""
    return Settings(
        debug=True,
        vector_db_path="./test_data/vector_db",
        cache_dir="./test_data/cache",
        log_file="./test_data/logs/test.log",
        secret_key="test_secret_key_for_testing_only",
        groq_api_key="test_key"
    )


@pytest.fixture
def test_documents_dir(tmp_path):
    """Create temporary documents directory."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    
    # Create sample test document
    test_doc = docs_dir / "test.txt"
    test_doc.write_text("This is a test document for testing purposes.")
    
    return docs_dir


@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """
    Enterprise RAG System is a production-ready Retrieval-Augmented Generation system.
    It uses advanced NLP techniques including semantic search, vector embeddings,
    and large language models to provide intelligent question answering capabilities.
    The system includes multi-agent orchestration for complex query handling.
    """


@pytest.fixture
def sample_chunks():
    """Provide sample text chunks."""
    return [
        "Enterprise RAG System is a production-ready system.",
        "It uses semantic search and vector embeddings.",
        "The system includes multi-agent orchestration.",
        "Large language models provide intelligent answers."
    ]


@pytest.fixture
def mock_llm_response():
    """Provide mock LLM response."""
    return {
        "content": "This is a test response from the LLM.",
        "tokens": 10,
        "model": "test-model"
    }


@pytest.fixture(scope="session")
def test_vector_db_path(tmp_path_factory):
    """Create temporary vector database path."""
    return tmp_path_factory.mktemp("vector_db")
