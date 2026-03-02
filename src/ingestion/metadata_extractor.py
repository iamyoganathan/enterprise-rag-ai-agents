"""
Metadata Extractor Module
Extracts and enriches document metadata.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import hashlib

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract and enrich document metadata."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        pass
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a document.
        
        Args:
            document: Loaded document
            
        Returns:
            Enhanced metadata dictionary
        """
        file_path = Path(document['file_path'])
        
        # Start with existing metadata
        metadata = document.get('metadata', {}).copy()
        
        # Add file system metadata
        metadata.update(self._extract_file_metadata(file_path))
        
        # Add content statistics
        metadata.update(self._extract_content_stats(document))
        
        # Add document hash for deduplication
        content = document.get('content', '') or ''
        metadata['content_hash'] = self._calculate_content_hash(content)
        
        # Add timestamp
        metadata['processed_at'] = datetime.now().isoformat()
        
        # Extract keywords (simplified)
        metadata['keywords'] = self._extract_keywords(content)
        
        # Document classification (simplified)
        metadata['category'] = self._classify_document(document)
        
        return metadata
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract file system metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File metadata
        """
        try:
            stat = file_path.stat()
            
            return {
                "file_name": file_path.name,
                "file_stem": file_path.stem,
                "file_extension": file_path.suffix,
                "file_size_bytes": stat.st_size,
                "file_size_kb": round(stat.st_size / 1024, 2),
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "absolute_path": str(file_path.absolute()),
            }
        except Exception as e:
            logger.error(f"Error extracting file metadata: {e}")
            return {}
    
    def _extract_content_stats(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract statistics about document content.
        
        Args:
            document: Document data
            
        Returns:
            Content statistics
        """
        content = document.get('content', '') or ''  # Ensure content is never None
        
        # Basic statistics
        char_count = len(content)
        word_count = len(content.split())
        line_count = content.count('\n') + 1
        
        # Calculate averages
        avg_word_length = char_count / word_count if word_count > 0 else 0
        avg_sentence_length = self._estimate_avg_sentence_length(content)
        
        # Complexity metrics
        stats = {
            "char_count": char_count,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": content.count('\n\n') + 1,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "estimated_reading_time_minutes": round(word_count / 200, 1),  # 200 words per minute
        }
        
        # Add format-specific stats
        doc_format = document.get('metadata', {}).get('format', '')
        
        if doc_format == 'pdf':
            stats['num_pages'] = document.get('metadata', {}).get('num_pages', 0)
        elif doc_format == 'docx':
            stats['num_paragraphs'] = document.get('metadata', {}).get('num_paragraphs', 0)
            stats['num_tables'] = document.get('metadata', {}).get('num_tables', 0)
        elif doc_format == 'markdown':
            stats['num_headers'] = document.get('metadata', {}).get('num_headers', 0)
            stats['num_code_blocks'] = document.get('metadata', {}).get('num_code_blocks', 0)
        
        return stats
    
    def _estimate_avg_sentence_length(self, text: str) -> float:
        """
        Estimate average sentence length in words.
        
        Args:
            text: Input text
            
        Returns:
            Average sentence length
        """
        import re
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _calculate_content_hash(self, content: str) -> str:
        """
        Calculate MD5 hash of content for deduplication.
        
        Args:
            content: Document content
            
        Returns:
            MD5 hash string
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text (simplified TF-IDF approach).
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        import re
        from collections import Counter
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what',
            'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'from',
            'up', 'down', 'out', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there'
        }
        
        # Extract words (alphanumeric, 3+ characters)
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Filter stop words
        words = [w for w in words if w not in stop_words]
        
        # Count frequencies
        word_freq = Counter(words)
        
        # Get top N keywords
        keywords = [word for word, _ in word_freq.most_common(top_n)]
        
        return keywords
    
    def _classify_document(self, document: Dict[str, Any]) -> str:
        """
        Classify document into a category (simplified).
        
        Args:
            document: Document data
            
        Returns:
            Document category
        """
        content = document.get('content', '').lower()
        
        # Simple keyword-based classification
        categories = {
            'technical': ['code', 'function', 'algorithm', 'data', 'system', 'software', 'api'],
            'business': ['revenue', 'profit', 'market', 'customer', 'sales', 'business', 'strategy'],
            'legal': ['law', 'legal', 'contract', 'agreement', 'terms', 'rights', 'liability'],
            'academic': ['research', 'study', 'analysis', 'methodology', 'hypothesis', 'conclusion'],
            'medical': ['patient', 'treatment', 'diagnosis', 'medical', 'clinical', 'health'],
        }
        
        category_scores = {}
        
        for category, keywords in categories.items():
            score = sum(content.count(keyword) for keyword in keywords)
            category_scores[category] = score
        
        # Return category with highest score, or 'general' if all scores are 0
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'
    
    def enrich_chunks_metadata(
        self,
        chunks: List[Any],
        document_metadata: Dict[str, Any]
    ) -> List[Any]:
        """
        Enrich chunk metadata with document-level information.
        
        Args:
            chunks: List of chunks
            document_metadata: Document metadata
            
        Returns:
            Chunks with enriched metadata
        """
        for chunk in chunks:
            # Add relevant document metadata to each chunk
            chunk.metadata.update({
                'doc_title': document_metadata.get('title', ''),
                'doc_author': document_metadata.get('author', ''),
                'doc_created': document_metadata.get('created_at', ''),
                'doc_category': document_metadata.get('category', ''),
                'doc_hash': document_metadata.get('content_hash', ''),
            })
        
        return chunks
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text (simplified version).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and values
        """
        import re
        
        entities = {
            'emails': [],
            'urls': [],
            'dates': [],
            'numbers': [],
        }
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities['urls'] = re.findall(url_pattern, text)
        
        # Extract dates (simple patterns)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        entities['dates'] = re.findall(date_pattern, text)
        
        # Extract numbers
        number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        entities['numbers'] = re.findall(number_pattern, text)[:10]  # Limit to avoid too many
        
        return entities


def extract_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from a document.
    
    Args:
        document: Document to process
        
    Returns:
        Enhanced metadata
    """
    extractor = MetadataExtractor()
    return extractor.extract(document)


if __name__ == "__main__":
    # Test the metadata extractor
    from .document_loader import load_document
    
    # Create a test document
    test_doc = {
        'file_path': 'test.txt',
        'file_name': 'test.txt',
        'content': """This is a test document for metadata extraction.
        It contains multiple sentences and paragraphs.
        
        The document discusses technical topics like algorithms and data structures.
        It also mentions business concepts like revenue and market analysis.
        
        Contact us at info@example.com or visit https://example.com for more information.
        The date is 01/28/2026 and the price is $1,234.56.""",
        'metadata': {'format': 'txt'}
    }
    
    extractor = MetadataExtractor()
    metadata = extractor.extract(test_doc)
    
    print("Extracted Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print("\nExtracted Entities:")
    entities = extractor.extract_entities(test_doc['content'])
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {values}")
