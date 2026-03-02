"""
Embedding Model Module
Handles text embedding generation using sentence-transformers.
Supports multiple embedding models, batch processing, and caching.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.cache import get_cache_manager
from src.utils.config import get_settings
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
cache_manager = get_cache_manager()
settings = get_settings()
perf_monitor = get_performance_monitor()


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding models.
    
    Supports:
    - Multiple embedding models
    - Batch processing for efficiency
    - Response caching
    - Dimension normalization
    - Progress tracking
    """
    
    # Supported embedding models with their properties
    MODELS = {
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "max_seq_length": 384,
            "description": "Best overall quality, balanced speed/performance"
        },
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_seq_length": 256,
            "description": "Faster, smaller model, good quality"
        },
        "multi-qa-mpnet-base-dot-v1": {
            "dimensions": 768,
            "max_seq_length": 512,
            "description": "Optimized for question-answering tasks"
        },
        "paraphrase-multilingual-mpnet-base-v2": {
            "dimensions": 768,
            "max_seq_length": 128,
            "description": "Supports 50+ languages"
        }
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_embeddings: bool = True,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', or None for auto)
            cache_embeddings: Whether to cache embedding results
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        self.model_name = model_name or settings.embedding_model
        self.cache_embeddings = cache_embeddings
        self.normalize_embeddings = normalize_embeddings
        
        # Validate model
        if self.model_name not in self.MODELS:
            logger.warning(
                f"Model {self.model_name} not in predefined list. "
                f"Available: {list(self.MODELS.keys())}"
            )
        
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            # Load the model
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Get model properties
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.max_seq_length = self.model.max_seq_length
            
            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dim={self.embedding_dim}, max_len={self.max_seq_length})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            convert_to_numpy: Whether to return numpy array
            
        Returns:
            Array of embeddings with shape (n_texts, embedding_dim)
        """
        # Handle single text
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # Check cache if enabled
        if self.cache_embeddings:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for idx, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached = cache_manager.get(cache_key)
                
                if cached is not None:
                    cached_embeddings.append((idx, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)
            
            if cached_embeddings:
                logger.debug(
                    f"Found {len(cached_embeddings)}/{len(texts)} "
                    f"embeddings in cache"
                )
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_embeddings = []
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            perf_monitor.start_timer("embedding_generation")
            try:
                try:
                    embeddings = self.model.encode(
                        uncached_texts,
                        batch_size=batch_size,
                        show_progress_bar=show_progress,
                        convert_to_numpy=convert_to_numpy,
                        normalize_embeddings=self.normalize_embeddings
                    )
                    
                    # Cache new embeddings
                    if self.cache_embeddings:
                        for text, embedding in zip(uncached_texts, embeddings):
                            cache_key = self._get_cache_key(text)
                            cache_manager.set(
                                cache_key,
                                embedding,
                                ttl=settings.cache_ttl
                            )
                    
                    perf_monitor.increment_counter(
                        "embeddings_generated",
                        len(uncached_texts)
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {str(e)}")
                    raise
            finally:
                perf_monitor.stop_timer("embedding_generation")
        else:
            embeddings = np.array([])
        
        # Combine cached and new embeddings in correct order
        if cached_embeddings:
            result = np.zeros((len(texts), self.embedding_dim))
            
            # Place cached embeddings
            for idx, emb in cached_embeddings:
                result[idx] = emb
            
            # Place new embeddings
            for idx, emb in zip(uncached_indices, embeddings):
                result[idx] = emb
            
            embeddings = result
        
        # Return single embedding if input was single text
        if is_single:
            return embeddings[0]
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Convenience method for batch encoding with progress.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        return self.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ('cosine' or 'dot')
            
        Returns:
            Similarity score
        """
        if method == "cosine":
            # Cosine similarity
            if self.normalize_embeddings:
                # If embeddings are normalized, dot product = cosine similarity
                return float(np.dot(embedding1, embedding2))
            else:
                # Calculate cosine similarity
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        
        elif method == "dot":
            # Dot product similarity
            return float(np.dot(embedding1, embedding2))
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def get_text_length(self, text: str) -> int:
        """
        Get the token length of a text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        tokens = self.model.tokenize([text])
        return len(tokens['input_ids'][0])
    
    def truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Truncate text to fit within model's max sequence length.
        
        Args:
            text: Input text
            max_length: Maximum length (uses model's max if None)
            
        Returns:
            Truncated text
        """
        max_length = max_length or self.max_seq_length
        
        # Tokenize and truncate
        tokens = self.model.tokenize([text])
        if len(tokens['input_ids'][0]) > max_length:
            # Simple character-based truncation (approximate)
            # More sophisticated: decode truncated tokens
            char_ratio = len(text) / len(tokens['input_ids'][0])
            truncated_chars = int(max_length * char_ratio * 0.95)  # 95% safety margin
            return text[:truncated_chars]
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for a text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key
        """
        # Create a hash of model name + text
        content = f"{self.model_name}:{text}"
        return f"emb_{hashlib.md5(content.encode()).hexdigest()}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_seq_length,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_enabled": self.cache_embeddings
        }
        
        # Add predefined model info if available
        if self.model_name in self.MODELS:
            info.update(self.MODELS[self.model_name])
        
        return info
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all available predefined models.
        
        Returns:
            Dictionary of model names and their properties
        """
        return cls.MODELS


# Singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: Optional[str] = None,
    force_reload: bool = False
) -> EmbeddingModel:
    """
    Get or create the embedding model singleton.
    
    Args:
        model_name: Model name (uses default if None)
        force_reload: Whether to force reload the model
        
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    
    if _embedding_model is None or force_reload:
        _embedding_model = EmbeddingModel(model_name=model_name)
    
    return _embedding_model


def embed_text(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to embed a single text.
    
    Args:
        text: Text to embed
        model_name: Model to use (uses default if None)
        
    Returns:
        Embedding vector
    """
    model = get_embedding_model(model_name)
    return model.encode(text)


def embed_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = False
) -> np.ndarray:
    """
    Convenience function to embed multiple texts.
    
    Args:
        texts: List of texts to embed
        model_name: Model to use (uses default if None)
        batch_size: Batch size for processing
        show_progress: Whether to show progress
        
    Returns:
        Array of embeddings
    """
    model = get_embedding_model(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress=show_progress
    )


if __name__ == "__main__":
    # Test the embedding model
    print("Testing Embedding Model\n" + "="*50)
    
    # Initialize model
    model = EmbeddingModel()
    print(f"\nModel Info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test single text
    text = "This is a test sentence for embedding generation."
    embedding = model.encode(text)
    print(f"\nSingle text embedding:")
    print(f"  Text: {text}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test batch
    texts = [
        "Machine learning is fascinating.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of data."
    ]
    embeddings = model.encode_batch(texts, show_progress=False)
    print(f"\nBatch embeddings:")
    print(f"  Texts: {len(texts)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    sim = model.similarity(embeddings[0], embeddings[1])
    print(f"\nSimilarity between texts 0 and 1: {sim:.4f}")
    
    print("\nEmbedding model test completed!")
