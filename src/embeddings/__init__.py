"""
Embeddings Module
Provides text embedding generation and vector storage capabilities.
"""

from src.embeddings.embedding_model import (
    EmbeddingModel,
    get_embedding_model,
    embed_text,
    embed_texts
)
from src.embeddings.vector_store import (
    VectorStore,
    get_vector_store
)
from src.embeddings.indexing import (
    IndexingPipeline,
    index_file,
    search_index
)

__all__ = [
    "EmbeddingModel",
    "get_embedding_model",
    "embed_text",
    "embed_texts",
    "VectorStore",
    "get_vector_store",
    "IndexingPipeline",
    "index_file",
    "search_index"
]

__version__ = "1.0.0"
