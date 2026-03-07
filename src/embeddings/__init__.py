"""
Embeddings Module
Provides text embedding generation and vector storage capabilities.
"""

# Note: EmbeddingModel, get_embedding_model, embed_text, embed_texts are no longer
# used at runtime. ChromaDB handles embeddings via built-in ONNX model.
# Kept as lazy imports for backward compatibility (tests, etc.)

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
    "VectorStore",
    "get_vector_store",
    "IndexingPipeline",
    "index_file",
    "search_index"
]

__version__ = "1.0.0"
