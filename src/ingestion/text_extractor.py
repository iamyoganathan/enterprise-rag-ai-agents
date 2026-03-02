"""
Text Extractor Module
Extracts text, tables, and metadata from loaded documents.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextExtractor:
    """Extract structured text from documents."""
    
    def __init__(self):
        """Initialize the text extractor."""
        pass
    
    def extract(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured content from a document.
        
        Args:
            document: Document loaded by DocumentLoader
            
        Returns:
            Dictionary with extracted content
        """
        doc_format = document['metadata']['format']
        
        if doc_format == 'pdf':
            return self._extract_from_pdf(document)
        elif doc_format == 'docx':
            return self._extract_from_docx(document)
        elif doc_format == 'txt':
            return self._extract_from_text(document)
        elif doc_format == 'markdown':
            return self._extract_from_markdown(document)
        else:
            return self._extract_generic(document)
    
    def _extract_from_pdf(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from PDF document."""
        content = document.get('content', '') or ''
        
        # Extract sections based on common patterns
        sections = self._extract_sections(content)
        
        # Extract sentences
        sentences = self._extract_sentences(content)
        
        # Identify potential headings (lines in all caps or followed by newlines)
        headings = self._extract_headings_from_text(content)
        
        return {
            "main_content": content,
            "sections": sections,
            "sentences": sentences,
            "headings": headings,
            "has_tables": False,  # Basic extraction, can be enhanced
            "page_count": document['metadata'].get('num_pages', 0)
        }
    
    def _extract_from_docx(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from DOCX document."""
        content = document.get('content', '') or ''
        paragraphs = document.get('paragraphs', [])
        tables = document.get('tables', [])
        
        # Extract sections
        sections = self._extract_sections(content)
        
        # Extract sentences
        sentences = self._extract_sentences(content)
        
        # Identify headings from paragraph styles (simplified)
        headings = []
        for para in paragraphs:
            if self._is_likely_heading(para):
                headings.append(para)
        
        # Format tables
        formatted_tables = []
        for table in tables:
            formatted_tables.append(self._format_table(table))
        
        return {
            "main_content": content,
            "sections": sections,
            "sentences": sentences,
            "headings": headings,
            "paragraphs": paragraphs,
            "tables": formatted_tables,
            "has_tables": len(tables) > 0,
            "paragraph_count": len(paragraphs)
        }
    
    def _extract_from_text(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from plain text document."""
        content = document.get('content', '') or ''
        lines = document.get('lines', [])
        
        # Extract sections
        sections = self._extract_sections(content)
        
        # Extract sentences
        sentences = self._extract_sentences(content)
        
        # Extract headings
        headings = self._extract_headings_from_text(content)
        
        return {
            "main_content": content,
            "sections": sections,
            "sentences": sentences,
            "headings": headings,
            "lines": lines,
            "has_tables": False,
            "line_count": len(lines)
        }
    
    def _extract_from_markdown(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content from Markdown document."""
        content = document.get('content', '') or ''
        headers = document.get('headers', [])
        raw_markdown = document.get('raw_markdown', content)
        
        # Extract sections based on headers
        sections = self._extract_markdown_sections(raw_markdown, headers)
        
        # Extract sentences
        sentences = self._extract_sentences(content)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(raw_markdown)
        
        return {
            "main_content": content,
            "sections": sections,
            "sentences": sentences,
            "headings": [h['text'] for h in headers],
            "headers": headers,
            "code_blocks": code_blocks,
            "has_tables": document['metadata'].get('has_tables', False)
        }
    
    def _extract_generic(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Generic extraction for unknown formats."""
        content = document.get('content', '') or ''
        
        return {
            "main_content": content,
            "sections": self._extract_sections(content),
            "sentences": self._extract_sentences(content),
            "headings": [],
            "has_tables": False
        }
    
    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract sections from text based on common patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of sections with titles and content
        """
        sections = []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_section = {"title": "Introduction", "content": ""}
        
        for para in paragraphs:
            # Check if paragraph might be a heading
            if self._is_likely_heading(para):
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                # Start new section
                current_section = {"title": para, "content": ""}
            else:
                current_section["content"] += para + "\n\n"
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with nltk)
        # Split on period, exclamation, question mark followed by space/newline
        sentences = re.split(r'[.!?]+[\s\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def _extract_headings_from_text(self, text: str) -> List[str]:
        """
        Extract likely headings from plain text.
        
        Args:
            text: Input text
            
        Returns:
            List of headings
        """
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if self._is_likely_heading(line):
                headings.append(line)
        
        return headings
    
    def _extract_markdown_sections(
        self,
        markdown_text: str,
        headers: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract sections from Markdown based on headers.
        
        Args:
            markdown_text: Raw markdown text
            headers: List of headers from markdown
            
        Returns:
            List of sections
        """
        if not headers:
            return [{"title": "Main Content", "content": markdown_text}]
        
        sections = []
        lines = markdown_text.split('\n')
        
        current_section = None
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    "title": title,
                    "level": level,
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _extract_code_blocks(self, markdown_text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from Markdown.
        
        Args:
            markdown_text: Raw markdown text
            
        Returns:
            List of code blocks with language and content
        """
        code_blocks = []
        
        # Match fenced code blocks
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.finditer(pattern, markdown_text, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append({
                "language": language,
                "code": code
            })
        
        return code_blocks
    
    def _is_likely_heading(self, text: str) -> bool:
        """
        Determine if text is likely a heading.
        
        Args:
            text: Text to check
            
        Returns:
            True if likely a heading
        """
        if not text or len(text) > 200:
            return False
        
        # Check for common heading patterns
        patterns = [
            text.isupper(),  # All caps
            bool(re.match(r'^\d+\.?\s+[A-Z]', text)),  # Starts with number
            bool(re.match(r'^[A-Z][^.!?]{5,100}$', text)),  # Capitalized, no punctuation
            len(text.split()) <= 10,  # Short
        ]
        
        return sum(patterns) >= 2
    
    def _format_table(self, table: List[List[str]]) -> str:
        """
        Format table data as string.
        
        Args:
            table: 2D list of table cells
            
        Returns:
            Formatted table string
        """
        if not table:
            return ""
        
        # Simple pipe-separated format
        formatted_rows = []
        for row in table:
            formatted_rows.append(" | ".join(row))
        
        return "\n".join(formatted_rows)
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract key phrases from text (simplified version).
        
        Args:
            text: Input text
            top_n: Number of phrases to return
            
        Returns:
            List of key phrases
        """
        # Simple frequency-based extraction
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                       'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                       'can', 'could', 'may', 'might', 'must', 'this', 'that', 'these', 'those'}
        
        # Extract words
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Filter common words
        words = [w for w in words if w not in common_words]
        
        # Count frequencies
        from collections import Counter
        word_freq = Counter(words)
        
        # Get top N
        key_phrases = [word for word, _ in word_freq.most_common(top_n)]
        
        return key_phrases


# Convenience function
def extract_text(document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured text from a document.
    
    Args:
        document: Document loaded by DocumentLoader
        
    Returns:
        Extracted content
    """
    extractor = TextExtractor()
    return extractor.extract(document)


if __name__ == "__main__":
    # Test the text extractor
    from .document_loader import load_document
    
    # Example: Load and extract from a document
    sample_file = Path("data/sample_documents/example.pdf")
    
    if sample_file.exists():
        doc = load_document(sample_file)
        extractor = TextExtractor()
        extracted = extractor.extract(doc)
        
        print(f"Extracted content from {doc['file_name']}:")
        print(f"  Sections: {len(extracted['sections'])}")
        print(f"  Sentences: {len(extracted['sentences'])}")
        print(f"  Headings: {len(extracted['headings'])}")
    else:
        print("Sample document not found. Add a document to test extraction.")
