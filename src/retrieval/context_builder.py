"""
Context Builder Module
Constructs optimal context windows for LLM from retrieved documents.
Handles context size limits and formatting.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken

from src.retrieval.reranker import RankedResult
from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Context:
    """Built context for LLM."""
    text: str
    chunks: List[Dict[str, Any]]
    num_chunks: int
    total_tokens: int
    truncated: bool
    metadata: Dict[str, Any]


class ContextBuilder:
    """
    Builds optimal context from retrieved documents.
    
    Features:
    - Token counting and limit enforcement
    - Context formatting with citations
    - Chunk deduplication
    - Priority-based selection
    - Metadata preservation
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        model: str = "gpt-3.5-turbo",
        include_metadata: bool = True,
        add_citations: bool = True,
        citation_format: str = "number"
    ):
        """
        Initialize the context builder.
        
        Args:
            max_tokens: Maximum context length in tokens
            model: Model name for token counting
            include_metadata: Whether to include metadata in context
            add_citations: Whether to add citation markers
            citation_format: Citation format ('number', 'bracket', 'inline')
        """
        self.max_tokens = max_tokens
        self.model = model
        self.include_metadata = include_metadata
        self.add_citations = add_citations
        self.citation_format = citation_format
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Unknown model {model}, using cl100k_base tokenizer")
        
        logger.info(f"Context builder initialized: max_tokens={max_tokens}")
    
    def build_context(
        self,
        retrieved_docs: List[Any],
        query: Optional[str] = None,
        include_query: bool = True
    ) -> Context:
        """
        Build context from retrieved documents.
        
        Args:
            retrieved_docs: Retrieved documents (can be dicts or RankedResults)
            query: Original query
            include_query: Whether to include query in context
            
        Returns:
            Context object
        """
        # Convert to standard format
        chunks = self._normalize_docs(retrieved_docs)
        
        # Remove duplicates
        chunks = self._deduplicate_chunks(chunks)
        
        # Build context text
        context_parts = []
        current_tokens = 0
        used_chunks = []
        truncated = False
        
        # Add query if requested
        if include_query and query:
            query_text = f"Query: {query}\n\n"
            query_tokens = self._count_tokens(query_text)
            if current_tokens + query_tokens < self.max_tokens:
                context_parts.append(query_text)
                current_tokens += query_tokens
        
        # Add instruction
        instruction = "Context information is below:\n\n"
        instruction_tokens = self._count_tokens(instruction)
        if current_tokens + instruction_tokens < self.max_tokens:
            context_parts.append(instruction)
            current_tokens += instruction_tokens
        
        # Add chunks
        for i, chunk in enumerate(chunks, 1):
            # Format chunk
            chunk_text = self._format_chunk(chunk, i)
            chunk_tokens = self._count_tokens(chunk_text)
            
            # Check if it fits
            if current_tokens + chunk_tokens > self.max_tokens:
                truncated = True
                logger.debug(
                    f"Context truncated: {len(used_chunks)}/{len(chunks)} chunks fit"
                )
                break
            
            context_parts.append(chunk_text)
            used_chunks.append(chunk)
            current_tokens += chunk_tokens
        
        # Build final context
        context_text = ''.join(context_parts)
        
        # Add closing
        if used_chunks:
            closing = "\n\nUse the above context to answer the query.\n"
            closing_tokens = self._count_tokens(closing)
            if current_tokens + closing_tokens <= self.max_tokens:
                context_text += closing
                current_tokens += closing_tokens
        
        # Build metadata
        metadata = {
            'sources': [c.get('metadata', {}).get('file_name', 'unknown') 
                       for c in used_chunks],
            'categories': list(set(
                c.get('metadata', {}).get('category', 'unknown') 
                for c in used_chunks
            ))
        }
        
        return Context(
            text=context_text,
            chunks=used_chunks,
            num_chunks=len(used_chunks),
            total_tokens=current_tokens,
            truncated=truncated,
            metadata=metadata
        )
    
    def _normalize_docs(
        self,
        docs: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Normalize documents to standard format.
        
        Args:
            docs: Documents in various formats
            
        Returns:
            Normalized document list
        """
        normalized = []
        
        for doc in docs:
            if isinstance(doc, RankedResult):
                normalized.append({
                    'document': doc.document,
                    'metadata': doc.metadata,
                    'score': doc.final_score,
                    'id': doc.id
                })
            elif isinstance(doc, dict):
                normalized.append({
                    'document': doc.get('document', ''),
                    'metadata': doc.get('metadata', {}),
                    'score': doc.get('similarity', doc.get('final_score', 0)),
                    'id': doc.get('id', '')
                })
            else:
                logger.warning(f"Unknown document format: {type(doc)}")
        
        return normalized
    
    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Deduplicated chunks
        """
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            text = chunk['document'].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        
        if len(unique_chunks) < len(chunks):
            logger.debug(
                f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks"
            )
        
        return unique_chunks
    
    def _format_chunk(self, chunk: Dict[str, Any], index: int) -> str:
        """
        Format a chunk for context.
        
        Args:
            chunk: Chunk data
            index: Chunk index
            
        Returns:
            Formatted chunk text
        """
        parts = []
        
        # Add citation marker
        if self.add_citations:
            if self.citation_format == "number":
                parts.append(f"[{index}] ")
            elif self.citation_format == "bracket":
                parts.append(f"<doc{index}> ")
        
        # Add document text
        parts.append(chunk['document'])
        
        # Add metadata if requested
        if self.include_metadata:
            metadata = chunk.get('metadata', {})
            meta_parts = []
            
            # File name
            if 'file_name' in metadata:
                meta_parts.append(f"Source: {metadata['file_name']}")
            
            # Page number (for PDFs)
            if 'page_number' in metadata:
                meta_parts.append(f"Page: {metadata['page_number']}")
            elif 'page' in metadata:
                meta_parts.append(f"Page: {metadata['page']}")
            
            # Section information
            if 'section' in metadata:
                meta_parts.append(f"Section: {metadata['section']}")
            elif 'heading' in metadata:
                meta_parts.append(f"Section: {metadata['heading']}")
            
            # Chunk position (for context)
            if 'chunk_index' in metadata:
                meta_parts.append(f"Chunk: {metadata['chunk_index']}")
            
            # Category
            if 'category' in metadata:
                meta_parts.append(f"Category: {metadata['category']}")
            
            if meta_parts:
                parts.append(f" ({', '.join(meta_parts)})")
        
        parts.append("\n\n")
        
        return ''.join(parts)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using approximation")
            # Approximate: ~4 characters per token
            return len(text) // 4
    
    def build_context_with_query(
        self,
        query: str,
        retrieved_docs: List[Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build complete prompt with system message, context, and query.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with 'system' and 'user' messages
        """
        # Build context
        context = self.build_context(retrieved_docs, query, include_query=False)
        
        # Build system prompt
        if not system_prompt:
            system_prompt = (
                "You are a helpful AI assistant. Use the provided context "
                "to answer the user's question accurately. If the context "
                "doesn't contain enough information, say so clearly."
            )
        
        # Build user message with context and query
        user_message = f"{context.text}\n\nQuestion: {query}"
        
        return {
            'system': system_prompt,
            'user': user_message,
            'metadata': {
                'num_chunks': context.num_chunks,
                'total_tokens': context.total_tokens,
                'truncated': context.truncated,
                'sources': context.metadata.get('sources', [])
            }
        }
    
    def get_chunk_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics about chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Summary statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_chars': 0,
                'avg_chunk_size': 0,
                'sources': []
            }
        
        total_chars = sum(len(c['document']) for c in chunks)
        sources = list(set(
            c.get('metadata', {}).get('file_name', 'unknown')
            for c in chunks
        ))
        
        return {
            'total_chunks': len(chunks),
            'total_chars': total_chars,
            'avg_chunk_size': total_chars // len(chunks),
            'sources': sources
        }


def build_context(
    retrieved_docs: List[Any],
    query: Optional[str] = None,
    max_tokens: int = 4000
) -> Context:
    """
    Convenience function to build context.
    
    Args:
        retrieved_docs: Retrieved documents
        query: Optional query
        max_tokens: Maximum context tokens
        
    Returns:
        Context object
    """
    builder = ContextBuilder(max_tokens=max_tokens)
    return builder.build_context(retrieved_docs, query)


if __name__ == "__main__":
    # Test the context builder
    print("Testing Context Builder\n" + "="*60)
    
    # Sample retrieved documents
    sample_docs = [
        {
            'document': 'Machine learning is a method of data analysis that automates analytical model building.',
            'metadata': {'file_name': 'ml_intro.txt', 'category': 'technical'},
            'similarity': 0.92,
            'id': 'doc1'
        },
        {
            'document': 'It is a branch of artificial intelligence based on the idea that systems can learn from data.',
            'metadata': {'file_name': 'ml_intro.txt', 'category': 'technical'},
            'similarity': 0.88,
            'id': 'doc2'
        },
        {
            'document': 'Machine learning algorithms build a model based on sample data, known as training data.',
            'metadata': {'file_name': 'ml_guide.txt', 'category': 'educational'},
            'similarity': 0.85,
            'id': 'doc3'
        }
    ]
    
    # Initialize builder
    builder = ContextBuilder(max_tokens=500, add_citations=True)
    
    # Build context
    query = "What is machine learning?"
    context = builder.build_context(sample_docs, query)
    
    print(f"Context Statistics:")
    print(f"  Chunks used: {context.num_chunks}/{len(sample_docs)}")
    print(f"  Total tokens: {context.total_tokens}")
    print(f"  Truncated: {context.truncated}")
    print(f"  Sources: {', '.join(context.metadata['sources'])}")
    
    print(f"\nBuilt Context:")
    print("-" * 60)
    print(context.text)
    
    # Build complete prompt
    print(f"\n" + "="*60)
    print("Complete Prompt")
    print("="*60)
    
    prompt = builder.build_context_with_query(query, sample_docs)
    print(f"\nSystem: {prompt['system'][:100]}...")
    print(f"\nUser: {prompt['user'][:200]}...")
    print(f"\nMetadata: {prompt['metadata']}")
    
    print("\nContext builder test completed!")
