"""
Ingestion Module
Handles document loading, text extraction, chunking, and metadata extraction.
"""

from src.ingestion.document_loader import (
    DocumentLoaderFactory,
    BaseDocumentLoader,
    PDFLoader,
    DOCXLoader,
    TextLoader,
    MarkdownLoader
)
from src.ingestion.text_extractor import TextExtractor
from src.ingestion.chunker import Chunk, TextChunker
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.pipeline import (
    IngestionPipeline,
    ingest_file,
    ingest_directory
)

__all__ = [
    "DocumentLoaderFactory",
    "BaseDocumentLoader",
    "PDFLoader",
    "DOCXLoader",
    "TextLoader",
    "MarkdownLoader",
    "TextExtractor",
    "Chunk",
    "TextChunker",
    "MetadataExtractor",
    "IngestionPipeline",
    "ingest_file",
    "ingest_directory"
]

__version__ = "1.0.0"
