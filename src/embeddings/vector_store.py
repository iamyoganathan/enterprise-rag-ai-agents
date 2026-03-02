"""
Vector Store Module
Manages vector database operations using ChromaDB.
Handles document storage, retrieval, and similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import uuid

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
settings = get_settings()
perf_monitor = get_performance_monitor()


class VectorStore:
    """
    Wrapper for ChromaDB vector database operations.
    
    Features:
    - Document storage with metadata
    - Semantic similarity search
    - CRUD operations on collections
    - Persistent storage
    - Filtering and metadata queries
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            embedding_function: Custom embedding function (uses default if None)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or str(
            Path(settings.vector_db_path)
        )
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {self.persist_directory}")
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"description": "RAG document collection"}
            )
            
            logger.info(
                f"Vector store initialized: {collection_name} "
                f"({self.collection.count()} documents)"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs (auto-generated if None)
            embeddings: Pre-computed embeddings (computed if None)
            
        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Ensure all metadata values are serializable
        metadatas = [self._sanitize_metadata(m) for m in metadatas]
        
        try:
            perf_monitor.start_timer("vector_store_add")
            try:
                # Add to collection
                if embeddings is not None:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                else:
                    # Let ChromaDB compute embeddings
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                perf_monitor.increment_counter("documents_added", len(documents))
                logger.info(f"Added {len(documents)} documents to {self.collection_name}")
            finally:
                perf_monitor.stop_timer("vector_store_add")
            
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_texts: List of query texts
            query_embeddings: List of query embeddings
            n_results: Number of results to return per query
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include: What to include in results
            
        Returns:
            Query results with documents, distances, and metadata
        """
        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        
        try:
            perf_monitor.start_timer("vector_store_query")
            try:
                results = self.collection.query(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include or ["documents", "metadatas", "distances"]
                )
                
                perf_monitor.increment_counter("vector_queries", 1)
                
                logger.debug(
                    f"Query returned {len(results.get('ids', [[]])[0])} results"
                )
            finally:
                perf_monitor.stop_timer("vector_store_query")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query_text: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a single query.
        
        Args:
            query_text: Query text
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of results with document, metadata, and distance
        """
        # Generate embedding for query text
        from .embedding_model import get_embedding_model
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(query_text).tolist()
        
        logger.info(f"[DEBUG] Similarity search: query='{query_text[:50]}...', k={k}, embedding_dim={len(query_embedding)}")
        logger.info(f"[DEBUG] Collection has {self.collection.count()} documents")
        
        results = self.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter
        )
        
        logger.info(f"[DEBUG] Query returned {len(results.get('ids', [[]])[0])} results")
        
        # Format results
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                # ChromaDB returns L2 distance - convert to similarity score (0-1 range)
                # Lower distance = higher similarity
                # Use inverse: similarity = 1 / (1 + distance)
                similarity = 1.0 / (1.0 + abs(distance))
                
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'similarity': similarity
                })
        
        return formatted_results
    
    def get_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents by IDs or filters.
        
        Args:
            ids: List of document IDs
            where: Metadata filter
            limit: Maximum number of documents
            offset: Number of documents to skip
            
        Returns:
            Documents with metadata
        """
        try:
            results = self.collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas", "embeddings"]
            )
            
            logger.debug(f"Retrieved {len(results['ids'])} documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get documents: {str(e)}")
            raise
    
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> None:
        """
        Update existing documents.
        
        Args:
            ids: List of document IDs to update
            documents: New document texts
            metadatas: New metadata
            embeddings: New embeddings
        """
        try:
            # Sanitize metadata
            if metadatas is not None:
                metadatas = [self._sanitize_metadata(m) for m in metadatas]
            
            self.collection.update(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Updated {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to update documents: {str(e)}")
            raise
    
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents by IDs or filters.
        
        Args:
            ids: List of document IDs
            where: Metadata filter
        """
        try:
            self.collection.delete(ids=ids, where=where)
            
            count = len(ids) if ids else "filtered"
            logger.info(f"Deleted {count} documents")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Document count
        """
        return self.collection.count()
    
    def reset(self) -> None:
        """
        Delete all documents from the collection.
        """
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"}
            )
            
            logger.warning(f"Reset collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection metadata and statistics
        """
        return {
            "name": self.collection_name,
            "count": self.count(),
            "persist_directory": self.persist_directory,
            "metadata": self.collection.metadata
        }
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to ensure ChromaDB compatibility.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Sanitized metadata
        """
        sanitized = {}
        
        for key, value in metadata.items():
            # Convert types that ChromaDB doesn't support
            if isinstance(value, (list, tuple)):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, dict):
                # Convert dicts to JSON strings
                import json
                sanitized[key] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                sanitized[key] = value
            elif value is None:
                # Skip None values
                continue
            else:
                # Convert other types to strings
                sanitized[key] = str(value)
        
        return sanitized
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [c.name for c in collections]


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store(
    collection_name: Optional[str] = None,
    force_reload: bool = False
) -> VectorStore:
    """
    Get or create the vector store singleton.
    
    Args:
        collection_name: Collection name (uses default if None)
        force_reload: Whether to force reload
        
    Returns:
        VectorStore instance
    """
    global _vector_store
    
    if _vector_store is None or force_reload:
        _vector_store = VectorStore(collection_name=collection_name or "documents")
    
    return _vector_store


if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store\n" + "="*50)
    
    # Initialize vector store
    store = VectorStore(collection_name="test_collection")
    print(f"\nCollection Info:")
    info = store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Add documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers."
    ]
    metadatas = [
        {"topic": "ml", "category": "tech"},
        {"topic": "nlp", "category": "tech"},
        {"topic": "dl", "category": "tech"}
    ]
    
    print(f"\nAdding {len(documents)} documents...")
    ids = store.add_documents(documents, metadatas)
    print(f"Added documents with IDs: {ids}")
    print(f"Total documents: {store.count()}")
    
    # Search
    query = "What is machine learning?"
    print(f"\nSearching for: '{query}'")
    results = store.similarity_search(query, k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Document: {result['document']}")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Metadata: {result['metadata']}")
    
    # Cleanup
    print(f"\nCleaning up test collection...")
    store.reset()
    print(f"Documents after reset: {store.count()}")
    
    print("\nVector store test completed!")
