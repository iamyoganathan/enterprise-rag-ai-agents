"""
Indexing Pipeline Module
Converts document chunks to embeddings and stores them in the vector database.
Orchestrates the complete indexing workflow.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from src.ingestion.chunker import Chunk
from src.embeddings.vector_store import get_vector_store
from src.utils.logger import get_logger
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
perf_monitor = get_performance_monitor()


class IndexingPipeline:
    """
    Pipeline for indexing document chunks into the vector database.
    
    Workflow:
    1. Generate embeddings for chunks
    2. Prepare metadata
    3. Store in vector database
    4. Track statistics
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model_name: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the indexing pipeline.
        
        Args:
            collection_name: Name of the vector store collection
            embedding_model_name: Name of embedding model to use
            batch_size: Batch size for embedding generation
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Initialize components (ChromaDB handles embeddings via built-in ONNX)
        self.vector_store = get_vector_store(collection_name)
        
        # Statistics
        self.stats = {
            "chunks_indexed": 0,
            "documents_indexed": 0,
            "embeddings_generated": 0,
            "indexing_time": 0.0
        }
        
        logger.info(
            f"Indexing pipeline initialized: "
            f"collection={collection_name} (using ChromaDB built-in embeddings)"
        )
    
    def index_chunks(
        self,
        chunks: List[Chunk],
        source_metadata: Optional[Dict[str, Any]] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        Index a list of chunks.
        
        Args:
            chunks: List of Chunk objects to index
            source_metadata: Additional metadata to add to all chunks
            show_progress: Whether to show progress bar
            
        Returns:
            List of document IDs in the vector store
        """
        if not chunks:
            logger.warning("No chunks to index")
            return []
        
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        perf_monitor.start_timer("indexing_pipeline")
        try:
            # Extract texts (ChromaDB will generate embeddings via built-in ONNX)
            texts = [chunk.text for chunk in chunks]
            self.stats["embeddings_generated"] += len(texts)
            
            # Prepare metadata
            metadatas = []
            for chunk in chunks:
                # Combine chunk metadata with source metadata
                metadata = chunk.metadata.copy() if chunk.metadata else {}
                
                # Add chunk-specific fields
                metadata.update({
                    "chunk_id": chunk.chunk_id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "token_count": chunk.token_count,
                    "text_length": len(chunk.text)
                })
                
                # Add source metadata
                if source_metadata:
                    metadata.update(source_metadata)
                
                metadatas.append(metadata)
            
            # Generate unique IDs for chunks (must be strings)
            # Include source filename to avoid ID conflicts across documents
            source_file = chunks[0].metadata.get('source_file', 'unknown') if chunks else 'unknown'
            import uuid
            doc_id = str(uuid.uuid4())[:8]
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            
            # Store in vector database
            logger.debug(f"Storing {len(chunks)} chunks in vector database...")
            self.vector_store.add_documents(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update statistics
            self.stats["chunks_indexed"] += len(chunks)
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
        finally:
            perf_monitor.stop_timer("indexing_pipeline")
        
        return ids
    
    def index_document(
        self,
        chunks: List[Chunk],
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Index all chunks from a single document.
        
        Args:
            chunks: List of chunks from the document
            document_metadata: Metadata about the source document
            
        Returns:
            Indexing result with statistics
        """
        logger.info(
            f"Indexing document: {document_metadata.get('file_name', 'unknown')} "
            f"({len(chunks)} chunks)"
        )
        
        # Index the chunks
        ids = self.index_chunks(
            chunks,
            source_metadata=document_metadata,
            show_progress=False
        )
        
        # Update statistics
        self.stats["documents_indexed"] += 1
        
        return {
            "document": document_metadata.get("file_name", "unknown"),
            "chunks_indexed": len(ids),
            "chunk_ids": ids,
            "status": "success"
        }
    
    def index_documents(
        self,
        documents_with_chunks: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Index multiple documents with their chunks.
        
        Args:
            documents_with_chunks: List of dicts containing 'chunks' and 'metadata'
            show_progress: Whether to show progress
            
        Returns:
            List of indexing results
        """
        logger.info(f"Indexing {len(documents_with_chunks)} documents...")
        
        results = []
        for doc_data in documents_with_chunks:
            try:
                result = self.index_document(
                    chunks=doc_data["chunks"],
                    document_metadata=doc_data.get("metadata", {})
                )
                results.append(result)
                
            except Exception as e:
                logger.error(
                    f"Failed to index document "
                    f"{doc_data.get('metadata', {}).get('file_name', 'unknown')}: "
                    f"{str(e)}"
                )
                results.append({
                    "document": doc_data.get("metadata", {}).get("file_name", "unknown"),
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(
            f"Indexed {len(results)} documents "
            f"({sum(1 for r in results if r['status'] == 'success')} successful)"
        )
        
        return results
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the indexed documents.
        
        Args:
            query: Search query
            k: Number of results
            filter: Metadata filter
            
        Returns:
            Search results
        """
        logger.debug(f"Searching for: '{query[:50]}...' (k={k})")
        
        results = self.vector_store.similarity_search(
            query_text=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def delete_document(
        self,
        document_id: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: ID of the document
            filter: Metadata filter to identify documents
        """
        if document_id:
            filter = {"document_id": document_id}
        
        self.vector_store.delete_documents(where=filter)
        logger.info(f"Deleted document with filter: {filter}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get indexing statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        stats["vector_store_count"] = self.vector_store.count()
        stats["collection_name"] = self.collection_name
        stats["embedding_model"] = self.embedding_model.model_name
        
        return stats
    
    def reset(self) -> None:
        """
        Clear all indexed documents.
        """
        self.vector_store.reset()
        self.stats = {
            "chunks_indexed": 0,
            "documents_indexed": 0,
            "embeddings_generated": 0,
            "indexing_time": 0.0
        }
        logger.warning("Indexing pipeline reset - all documents deleted")


def index_file(
    file_path: str,
    chunks: List[Chunk],
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: str = "documents"
) -> Dict[str, Any]:
    """
    Convenience function to index a single file.
    
    Args:
        file_path: Path to the source file
        chunks: List of chunks from the file
        metadata: Additional metadata
        collection_name: Vector store collection name
        
    Returns:
        Indexing result
    """
    pipeline = IndexingPipeline(collection_name=collection_name)
    
    # Prepare document metadata
    doc_metadata = metadata or {}
    doc_metadata["file_path"] = str(file_path)
    doc_metadata["file_name"] = Path(file_path).name
    
    return pipeline.index_document(chunks, doc_metadata)


def search_index(
    query: str,
    k: int = 5,
    collection_name: str = "documents",
    filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to search the index.
    
    Args:
        query: Search query
        k: Number of results
        collection_name: Collection to search
        filter: Metadata filter
        
    Returns:
        Search results
    """
    pipeline = IndexingPipeline(collection_name=collection_name)
    return pipeline.search(query, k, filter)


if __name__ == "__main__":
    # Test the indexing pipeline
    print("Testing Indexing Pipeline\n" + "="*50)
    
    # Create sample chunks
    from src.ingestion.chunker import Chunk
    
    chunks = [
        Chunk(
            text="Machine learning is a method of data analysis.",
            chunk_id="chunk_1",
            start_char=0,
            end_char=46,
            metadata={"source": "test"}
        ),
        Chunk(
            text="It automates analytical model building.",
            chunk_id="chunk_2",
            start_char=47,
            end_char=86,
            metadata={"source": "test"}
        ),
        Chunk(
            text="Deep learning is a subset of machine learning.",
            chunk_id="chunk_3",
            start_char=87,
            end_char=134,
            metadata={"source": "test"}
        )
    ]
    
    # Initialize pipeline
    pipeline = IndexingPipeline(collection_name="test_indexing")
    print(f"\nPipeline Stats:")
    for key, value in pipeline.get_stats().items():
        print(f"  {key}: {value}")
    
    # Index chunks
    print(f"\nIndexing {len(chunks)} chunks...")
    doc_metadata = {
        "file_name": "test_doc.txt",
        "category": "technical"
    }
    result = pipeline.index_document(chunks, doc_metadata)
    
    print(f"\nIndexing Result:")
    print(f"  Document: {result['document']}")
    print(f"  Chunks indexed: {result['chunks_indexed']}")
    print(f"  Status: {result['status']}")
    
    # Search
    query = "What is machine learning?"
    print(f"\nSearching for: '{query}'")
    search_results = pipeline.search(query, k=2)
    
    for i, result in enumerate(search_results, 1):
        print(f"\nResult {i}:")
        print(f"  Text: {result['document'][:80]}...")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Metadata: {result['metadata']}")
    
    # Statistics
    print(f"\nFinal Statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    print(f"\nCleaning up...")
    pipeline.reset()
    
    print("\nIndexing pipeline test completed!")
