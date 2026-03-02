"""
Query Processor Module
Handles query transformation, expansion, and multi-query generation.
Improves retrieval by creating better search queries.
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ProcessedQuery:
    """A processed query with variations."""
    original: str
    cleaned: str
    expanded: List[str]
    keywords: List[str]
    intent: str


class QueryProcessor:
    """
    Processes and enhances search queries.
    
    Features:
    - Query cleaning and normalization
    - Keyword extraction
    - Query expansion
    - Multi-query generation
    - Intent detection
    """
    
    def __init__(
        self,
        expand_queries: bool = True,
        max_variations: int = 3,
        extract_keywords: bool = True
    ):
        """
        Initialize the query processor.
        
        Args:
            expand_queries: Whether to generate query expansions
            max_variations: Maximum number of query variations
            extract_keywords: Whether to extract keywords
        """
        self.expand_queries = expand_queries
        self.max_variations = max_variations
        self.extract_keywords = extract_keywords
        
        # Common stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with'
        }
        
        # Intent patterns
        self.intent_patterns = {
            'definition': r'\b(what is|define|explain|describe)\b',
            'how_to': r'\b(how to|how do|how can)\b',
            'comparison': r'\b(difference|compare|versus|vs)\b',
            'list': r'\b(list|enumerate|types of|kinds of)\b',
            'reasoning': r'\b(why|reason|cause)\b',
            'example': r'\b(example|instance|sample)\b'
        }
        
        logger.info("Query processor initialized")
    
    def process(self, query: str) -> ProcessedQuery:
        """
        Process a query and generate variations.
        
        Args:
            query: Input query
            
        Returns:
            ProcessedQuery with all variations
        """
        # Clean query
        cleaned = self._clean_query(query)
        
        # Extract keywords
        keywords = []
        if self.extract_keywords:
            keywords = self._extract_keywords(cleaned)
        
        # Detect intent
        intent = self._detect_intent(cleaned)
        
        # Generate expansions
        expanded = [cleaned]
        if self.expand_queries:
            expanded.extend(self._expand_query(cleaned, intent))
            expanded = expanded[:self.max_variations]
        
        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            expanded=expanded,
            keywords=keywords,
            intent=intent
        )
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize the query.
        
        Args:
            query: Raw query
            
        Returns:
            Cleaned query
        """
        # Remove extra whitespace
        cleaned = ' '.join(query.split())
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        cleaned = re.sub(r'[^\w\s\?\.\,\-]', '', cleaned)
        
        # Normalize case (keep original for now, as it may contain important info)
        # cleaned = cleaned.lower()
        
        return cleaned.strip()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from the query.
        
        Args:
            query: Cleaned query
            
        Returns:
            List of keywords
        """
        # Tokenize
        words = query.lower().split()
        
        # Remove stop words and short words
        keywords = [
            w for w in words
            if w not in self.stop_words and len(w) > 2
        ]
        
        # Keep unique keywords in order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _detect_intent(self, query: str) -> str:
        """
        Detect the intent of the query.
        
        Args:
            query: Cleaned query
            
        Returns:
            Intent type
        """
        query_lower = query.lower()
        
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                return intent
        
        return 'general'
    
    def _expand_query(self, query: str, intent: str) -> List[str]:
        """
        Generate query expansions based on intent.
        
        Args:
            query: Cleaned query
            intent: Detected intent
            
        Returns:
            List of expanded queries
        """
        expansions = []
        query_lower = query.lower()
        
        # Intent-specific expansions
        if intent == 'definition':
            # Add more specific variations
            expansions.append(query.replace('what is', 'explanation of'))
            expansions.append(query.replace('define', 'definition of'))
        
        elif intent == 'how_to':
            # Add tutorial-style variations
            expansions.append(query.replace('how to', 'steps to'))
            expansions.append(query.replace('how do', 'method for'))
        
        elif intent == 'comparison':
            # Add comparison variations
            if 'difference' in query_lower:
                expansions.append(query.replace('difference', 'comparison'))
        
        elif intent == 'list':
            # Add enumeration variations
            expansions.append(query.replace('list', 'types of'))
        
        elif intent == 'reasoning':
            # Add explanation variations
            expansions.append(query.replace('why', 'reason for'))
        
        # Generic expansions
        if '?' in query:
            # Remove question mark for statement form
            expansions.append(query.replace('?', ''))
        else:
            # Add question mark for question form
            expansions.append(f"{query}?")
        
        # Add keyword-only version
        keywords = self._extract_keywords(query)
        if len(keywords) >= 2:
            expansions.append(' '.join(keywords[:3]))
        
        # Remove duplicates and original query
        unique_expansions = []
        seen = {query.lower()}
        
        for exp in expansions:
            exp_lower = exp.lower()
            if exp_lower not in seen and exp_lower != query.lower():
                seen.add(exp_lower)
                unique_expansions.append(exp)
        
        return unique_expansions
    
    def generate_multi_queries(
        self,
        query: str,
        num_queries: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations for multi-query retrieval.
        
        Args:
            query: Original query
            num_queries: Number of variations to generate
            
        Returns:
            List of query variations
        """
        processed = self.process(query)
        
        # Start with expanded queries
        variations = [query] + processed.expanded
        
        # Add keyword-based queries
        if len(processed.keywords) >= 2:
            # Different keyword combinations
            if len(processed.keywords) >= 3:
                variations.append(' '.join(processed.keywords[:2]))
                variations.append(' '.join(processed.keywords[1:3]))
        
        # Add intent-specific variations
        if processed.intent == 'definition':
            variations.append(f"information about {' '.join(processed.keywords)}")
        elif processed.intent == 'how_to':
            variations.append(f"tutorial for {' '.join(processed.keywords)}")
        
        # Remove duplicates
        unique = []
        seen = set()
        for v in variations:
            v_lower = v.lower()
            if v_lower not in seen:
                seen.add(v_lower)
                unique.append(v)
        
        return unique[:num_queries]
    
    def decompose_complex_query(self, query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-queries
        """
        sub_queries = []
        
        # Split by common conjunctions
        conjunctions = [' and ', ' or ', ', ']
        
        parts = [query]
        for conj in conjunctions:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(conj))
            parts = new_parts
        
        # Clean and validate sub-queries
        for part in parts:
            cleaned = part.strip()
            if len(cleaned.split()) >= 2:  # At least 2 words
                sub_queries.append(cleaned)
        
        # If no valid decomposition, return original
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    def enhance_query_with_context(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance query with additional context.
        
        Args:
            query: Original query
            context: Additional context (domain, user preferences, etc.)
            
        Returns:
            Enhanced query
        """
        if not context:
            return query
        
        enhanced = query
        
        # Add domain context
        if 'domain' in context:
            domain = context['domain']
            if domain and domain not in query.lower():
                enhanced = f"{query} in {domain}"
        
        # Add temporal context
        if 'time_period' in context:
            period = context['time_period']
            if period:
                enhanced = f"{enhanced} {period}"
        
        return enhanced


def process_query(
    query: str,
    expand: bool = True,
    max_variations: int = 3
) -> ProcessedQuery:
    """
    Convenience function to process a query.
    
    Args:
        query: Input query
        expand: Whether to expand the query
        max_variations: Maximum variations
        
    Returns:
        ProcessedQuery
    """
    processor = QueryProcessor(
        expand_queries=expand,
        max_variations=max_variations
    )
    return processor.process(query)


if __name__ == "__main__":
    # Test the query processor
    print("Testing Query Processor\n" + "="*60)
    
    processor = QueryProcessor()
    
    test_queries = [
        "What is machine learning?",
        "How to implement neural networks in Python?",
        "Difference between supervised and unsupervised learning",
        "List the types of deep learning architectures",
        "Why is data preprocessing important?"
    ]
    
    for query in test_queries:
        print(f"\nOriginal Query: '{query}'")
        print("-" * 60)
        
        processed = processor.process(query)
        
        print(f"Cleaned: {processed.cleaned}")
        print(f"Intent: {processed.intent}")
        print(f"Keywords: {', '.join(processed.keywords)}")
        print(f"\nExpanded Queries:")
        for i, exp in enumerate(processed.expanded, 1):
            print(f"  {i}. {exp}")
        
        # Multi-query generation
        print(f"\nMulti-Query Variations:")
        multi = processor.generate_multi_queries(query, num_queries=3)
        for i, q in enumerate(multi, 1):
            print(f"  {i}. {q}")
    
    # Test decomposition
    print(f"\n" + "="*60)
    print("Complex Query Decomposition")
    print("="*60)
    
    complex_query = "What is machine learning and how does it differ from traditional programming?"
    print(f"\nComplex Query: '{complex_query}'")
    sub_queries = processor.decompose_complex_query(complex_query)
    print("Sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")
    
    print("\nQuery processor test completed!")
