"""
Complete Document Ingestion Pipeline
Orchestrates loading, extraction, chunking, and metadata enrichment.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from .document_loader import DocumentLoaderFactory
from .text_extractor import TextExtractor
from .chunker import TextChunker, Chunk
from .metadata_extractor import MetadataExtractor
from ..utils.logger import get_logger
from ..utils.config import get_settings
from ..utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
settings = get_settings()
monitor = get_performance_monitor()


class IngestionPipeline:
    """Complete pipeline for document ingestion."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: str = "recursive"
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking
        """
        self.loader_factory = DocumentLoaderFactory()
        self.text_extractor = TextExtractor()
        self.chunker = TextChunker(chunk_size, chunk_overlap, chunking_strategy)
        self.metadata_extractor = MetadataExtractor()
        
        logger.info("Ingestion pipeline initialized")
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Processed document with chunks and metadata
        """
        monitor.start_timer("ingestion_pipeline")
        monitor.increment_counter("documents_processed")
        
        try:
            # Step 1: Load document
            logger.info(f"Processing file: {file_path}")
            document = self.loader_factory.load_document(file_path)
            
            # Step 2: Extract text and structure
            extracted = self.text_extractor.extract(document)
            document['extracted'] = extracted
            
            # Step 3: Extract and enrich metadata
            metadata = self.metadata_extractor.extract(document)
            document['metadata'] = metadata
            
            # Step 4: Chunk the document
            chunks = self.chunker.chunk(document)
            
            # Step 5: Enrich chunk metadata
            chunks = self.metadata_extractor.enrich_chunks_metadata(chunks, metadata)
            
            result = {
                'document': document,
                'chunks': chunks,
                'metadata': metadata,
                'stats': {
                    'num_chunks': len(chunks),
                    'avg_chunk_size': sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
                    'total_chars': metadata.get('char_count', 0),
                    'total_words': metadata.get('word_count', 0),
                }
            }
            
            duration = monitor.stop_timer("ingestion_pipeline")
            logger.info(f"Successfully processed {file_path} in {duration:.2f}s ({len(chunks)} chunks)")
            monitor.increment_counter("documents_success")
            
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing {file_path}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            monitor.increment_counter("documents_failed")
            raise
    
    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of processed documents
        """
        directory = Path(directory)
        logger.info(f"Processing directory: {directory} (recursive={recursive})")
        
        # Load all documents
        documents = self.loader_factory.load_directory(directory, recursive)
        
        results = []
        for doc in documents:
            try:
                result = self.process_document(doc)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {doc.get('file_name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(results)} documents from {directory}")
        return results
    
    def process_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an already-loaded document.
        
        Args:
            document: Loaded document
            
        Returns:
            Processed document with chunks
        """
        monitor.start_timer("document_processing")
        
        try:
            # Extract text and structure
            extracted = self.text_extractor.extract(document)
            document['extracted'] = extracted
            
            # Extract and enrich metadata
            metadata = self.metadata_extractor.extract(document)
            document['metadata'] = metadata
            
            # Chunk the document
            chunks = self.chunker.chunk(document)
            
            # Enrich chunk metadata
            chunks = self.metadata_extractor.enrich_chunks_metadata(chunks, metadata)
            
            result = {
                'document': document,
                'chunks': chunks,
                'metadata': metadata,
                'stats': {
                    'num_chunks': len(chunks),
                    'avg_chunk_size': sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
                }
            }
            
            duration = monitor.stop_timer("document_processing")
            logger.debug(f"Processed document in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'counters': monitor.get_counters(),
            'metrics': monitor.get_summary()
        }


def ingest_file(
    file_path: Union[str, Path],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to ingest a single file.
    
    Args:
        file_path: Path to the file
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        
    Returns:
        Processed document
    """
    pipeline = IngestionPipeline(chunk_size, chunk_overlap)
    return pipeline.process_file(file_path)


def ingest_directory(
    directory: Union[str, Path],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    recursive: bool = True
) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest all files in a directory.
    
    Args:
        directory: Path to directory
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        recursive: Search subdirectories
        
    Returns:
        List of processed documents
    """
    pipeline = IngestionPipeline(chunk_size, chunk_overlap)
    return pipeline.process_directory(directory, recursive)


if __name__ == "__main__":
    # Test the pipeline
    import sys
    
    if len(sys.argv) > 1:
        # Process file/directory from command line
        path = Path(sys.argv[1])
        
        pipeline = IngestionPipeline()
        
        if path.is_file():
            result = pipeline.process_file(path)
            print(f"\nProcessed: {result['document']['file_name']}")
            print(f"Chunks: {result['stats']['num_chunks']}")
            print(f"Category: {result['metadata'].get('category', 'unknown')}")
        elif path.is_dir():
            results = pipeline.process_directory(path)
            print(f"\nProcessed {len(results)} documents")
            for r in results:
                print(f"  - {r['document']['file_name']}: {r['stats']['num_chunks']} chunks")
        
        print(f"\nPipeline stats:")
        stats = pipeline.get_stats()
        print(f"  Documents processed: {stats['counters'].get('documents_processed', 0)}")
        print(f"  Success: {stats['counters'].get('documents_success', 0)}")
        print(f"  Failed: {stats['counters'].get('documents_failed', 0)}")
    else:
        print("Usage: python -m src.ingestion.pipeline <file_or_directory>")
        print("\nExample:")
        print("  python -m src.ingestion.pipeline data/sample_documents/")
