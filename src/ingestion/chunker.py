"""
Chunker Module
Intelligent text chunking with overlap for optimal retrieval.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from ..utils.logger import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """Represents a text chunk."""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None


class TextChunker:
    """
    Intelligent text chunking with multiple strategies.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        strategy: str = "recursive"
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ("recursive", "sentence", "paragraph", "semantic")
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy
        
        logger.info(f"Initialized TextChunker: size={self.chunk_size}, overlap={self.chunk_overlap}, strategy={self.strategy}")
    
    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of chunks
        """
        content = document.get('content', '')
        
        if not content:
            logger.warning(f"Empty document: {document.get('file_name', 'Unknown')}")
            return []
        
        if self.strategy == "recursive":
            chunks = self._recursive_chunk(content)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunk(content)
        elif self.strategy == "paragraph":
            chunks = self._paragraph_chunk(content)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunk(content)
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using recursive")
            chunks = self._recursive_chunk(content)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i
            chunk.metadata.update({
                "source_file": document.get('file_name', 'Unknown'),
                "source_path": document.get('file_path', ''),
                "total_chunks": len(chunks),
                "format": document.get('metadata', {}).get('format', 'unknown')
            })
        
        logger.info(f"Created {len(chunks)} chunks from {document.get('file_name', 'Unknown')}")
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[Chunk]:
        """
        Recursive chunking strategy.
        Tries to split on paragraph, then sentence, then character boundaries.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        separators = [
            "\n\n\n",  # Multiple newlines
            "\n\n",    # Double newline (paragraph)
            "\n",      # Single newline
            ". ",      # Sentence ending
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",      # Comma
            " ",       # Space
            ""         # Character
        ]
        
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[Chunk]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_start = 0
        
        if not separators:
            # No more separators, split by character
            return self._split_by_length(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator) if separator else list(text)
        
        current_text = ""
        current_start = 0
        
        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            if separator and i < len(splits) - 1:
                split += separator
            
            # Check if adding this split exceeds chunk size
            if len(current_text) + len(split) > self.chunk_size:
                if current_text:
                    # Save current chunk
                    chunk = Chunk(
                        text=current_text.strip(),
                        chunk_id=len(chunks),
                        start_char=current_start,
                        end_char=current_start + len(current_text),
                        metadata={}
                    )
                    chunks.append(chunk)
                    
                    # Handle overlap
                    overlap_text = current_text[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_start = current_start + len(current_text) - len(overlap_text)
                    current_text = overlap_text + split
                else:
                    # Split is too large, try next separator
                    if remaining_separators:
                        sub_chunks = self._recursive_split(split, remaining_separators)
                        chunks.extend(sub_chunks)
                        current_start += len(split)
                    else:
                        # No more separators, force split
                        force_chunks = self._split_by_length(split)
                        chunks.extend(force_chunks)
                        current_start += len(split)
            else:
                current_text += split
        
        # Add remaining text
        if current_text.strip():
            chunk = Chunk(
                text=current_text.strip(),
                chunk_id=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_text),
                metadata={}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sentence_chunk(self, text: str) -> List[Chunk]:
        """
        Chunk by sentences.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+[\s\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = Chunk(
                    text=current_chunk.strip(),
                    chunk_id=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={"chunking_method": "sentence"}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk.split(". ")[-2:] if ". " in current_chunk else []
                overlap_text = ". ".join(overlap_sentences)
                current_start = current_start + len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining chunk
        if current_chunk.strip():
            chunk = Chunk(
                text=current_chunk.strip(),
                chunk_id=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"chunking_method": "sentence"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_chunk(self, text: str) -> List[Chunk]:
        """
        Chunk by paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = Chunk(
                    text=current_chunk.strip(),
                    chunk_id=len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={"chunking_method": "paragraph"}
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_start += len(current_chunk)
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add remaining chunk
        if current_chunk.strip():
            chunk = Chunk(
                text=current_chunk.strip(),
                chunk_id=len(chunks),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={"chunking_method": "paragraph"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[Chunk]:
        """
        Semantic chunking based on topic coherence (simplified version).
        Falls back to sentence chunking.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # For now, use sentence chunking as semantic chunking requires embeddings
        # This can be enhanced later with actual semantic similarity
        logger.info("Semantic chunking not fully implemented, using sentence chunking")
        chunks = self._sentence_chunk(text)
        for chunk in chunks:
            chunk.metadata["chunking_method"] = "semantic"
        return chunks
    
    def _split_by_length(self, text: str) -> List[Chunk]:
        """
        Force split by character length.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=len(chunks),
                start_char=start,
                end_char=end,
                metadata={"chunking_method": "force_split"}
            )
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


def chunk_document(
    document: Dict[str, Any],
    chunk_size: int = None,
    chunk_overlap: int = None,
    strategy: str = "recursive"
) -> List[Chunk]:
    """
    Chunk a document into smaller pieces.
    
    Args:
        document: Document to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy
        
    Returns:
        List of chunks
    """
    chunker = TextChunker(chunk_size, chunk_overlap, strategy)
    return chunker.chunk(document)


if __name__ == "__main__":
    # Test the chunker
    from .document_loader import load_document
    
    sample_text = """
    This is the first paragraph. It contains several sentences.
    This helps demonstrate the chunking functionality.
    
    This is the second paragraph. It's separated by a double newline.
    The chunker should handle this appropriately.
    
    This is the third paragraph with more content to ensure we have
    enough text to create multiple chunks and test the overlap
    functionality properly.
    """
    
    # Test different strategies
    strategies = ["recursive", "sentence", "paragraph"]
    
    for strategy in strategies:
        print(f"\n=== Testing {strategy} strategy ===")
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy=strategy)
        
        doc = {
            "content": sample_text,
            "file_name": "test.txt",
            "metadata": {"format": "txt"}
        }
        
        chunks = chunker.chunk(doc)
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i}:")
            print(f"  Length: {len(chunk.text)}")
            print(f"  Text: {chunk.text[:80]}...")
