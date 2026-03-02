"""
Test script for document ingestion module.
Creates sample documents and tests the complete pipeline.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_sample_documents():
    """Create sample documents for testing."""
    sample_dir = Path("data/sample_documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample 1: Technical document
    tech_doc = """# Technical Documentation

## Introduction
This document provides an overview of the RAG (Retrieval Augmented Generation) system architecture.

## System Architecture
The RAG system consists of several key components:

1. **Document Ingestion**: Processes and chunks documents
2. **Embedding Generation**: Converts text to vector embeddings
3. **Vector Storage**: Stores embeddings in ChromaDB
4. **Retrieval**: Searches for relevant documents
5. **Generation**: Uses LLM to generate responses

## Implementation Details
The system uses Python 3.10+ and the following technologies:
- LangChain for LLM orchestration
- ChromaDB for vector storage
- FastAPI for REST API
- Streamlit for user interface

## Code Example
```python
def retrieve_context(query: str, top_k: int = 5):
    # Embed query
    query_embedding = embed_text(query)
    
    # Search vector database
    results = vector_db.search(query_embedding, top_k)
    
    return results
```

## Conclusion
This architecture provides a scalable and efficient RAG implementation.
"""
    
    with open(sample_dir / "technical_doc.md", "w", encoding="utf-8") as f:
        f.write(tech_doc)
    
    # Sample 2: Business document
    business_doc = """Business Analysis Report

Executive Summary
This report analyzes the market trends and revenue projections for Q4 2026.

Market Overview
The current market shows strong growth potential with increasing customer demand.
Our revenue has grown by 25% compared to the previous quarter.

Key Findings:
- Customer acquisition cost decreased by 15%
- Customer retention rate improved to 92%
- Average revenue per user increased by $12

Financial Projections
Based on current trends, we project:
- Q4 Revenue: $2.5M
- Annual Revenue: $8.7M
- Profit Margin: 32%

Recommendations
1. Increase marketing budget by 20%
2. Expand to new markets
3. Invest in customer success team
4. Develop new product features

Conclusion
The business is well-positioned for continued growth in 2027.
"""
    
    with open(sample_dir / "business_report.txt", "w", encoding="utf-8") as f:
        f.write(business_doc)
    
    # Sample 3: Research paper
    research_doc = """Abstract

This paper presents a novel approach to document retrieval using semantic embeddings
and large language models. The proposed method achieves 87% accuracy on benchmark datasets.

1. Introduction

Information retrieval has evolved significantly with the advent of neural networks
and transformer-based models. Traditional keyword-based search systems often fail
to capture semantic meaning and context.

2. Related Work

Previous research in this area includes:
- BM25 algorithm for text ranking
- TF-IDF for feature extraction
- BERT for semantic understanding
- Dense Passage Retrieval (DPR)

3. Methodology

Our approach combines the following techniques:
1. Document preprocessing and chunking
2. Embedding generation using sentence-transformers
3. Vector similarity search
4. Re-ranking with cross-encoders
5. Response generation using LLMs

4. Experimental Results

We evaluated our system on three datasets:
- MS MARCO: 85% accuracy
- Natural Questions: 87% accuracy
- SQuAD: 89% accuracy

The results demonstrate significant improvement over baseline methods.

5. Conclusion

This research shows that combining retrieval with generation yields superior results
compared to either approach alone. Future work will explore multi-modal retrieval.

References
[1] Vaswani et al. "Attention is All You Need" (2017)
[2] Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
[3] Lewis et al. "Retrieval-Augmented Generation" (2020)
"""
    
    with open(sample_dir / "research_paper.txt", "w", encoding="utf-8") as f:
        f.write(research_doc)
    
    logger.info(f"Created 3 sample documents in {sample_dir}")
    return sample_dir


def test_ingestion():
    """Test the complete ingestion pipeline."""
    print("=" * 60)
    print("Testing Document Ingestion Module")
    print("=" * 60)
    
    # Create sample documents
    print("\n1. Creating sample documents...")
    sample_dir = create_sample_documents()
    print(f"   ✓ Created documents in {sample_dir}")
    
    # Initialize pipeline
    print("\n2. Initializing ingestion pipeline...")
    pipeline = IngestionPipeline(
        chunk_size=500,
        chunk_overlap=50,
        chunking_strategy="recursive"
    )
    print("   ✓ Pipeline initialized")
    
    # Process directory
    print(f"\n3. Processing documents from {sample_dir}...")
    try:
        results = pipeline.process_directory(sample_dir)
        print(f"   ✓ Successfully processed {len(results)} documents")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Display results
    print("\n4. Processing Results:")
    print("-" * 60)
    
    total_chunks = 0
    for result in results:
        doc = result['document']
        metadata = result['metadata']
        stats = result['stats']
        
        print(f"\n   Document: {doc['file_name']}")
        print(f"   Format: {metadata['format']}")
        print(f"   Category: {metadata.get('category', 'unknown')}")
        print(f"   Words: {doc['word_count']}")
        print(f"   Chunks: {stats['num_chunks']}")
        print(f"   Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
        print(f"   Keywords: {', '.join(metadata.get('keywords', [])[:5])}")
        print(f"   Reading time: {metadata.get('estimated_reading_time_minutes', 0)} min")
        
        total_chunks += stats['num_chunks']
        
        # Show first chunk
        if result['chunks']:
            first_chunk = result['chunks'][0]
            print(f"\n   First chunk preview:")
            print(f"   {first_chunk.text[:150]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total documents processed: {len(results)}")
    print(f"  Total chunks created: {total_chunks}")
    
    pipeline_stats = pipeline.get_stats()
    print(f"\n  Pipeline Statistics:")
    counters = pipeline_stats['counters']
    for key, value in counters.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_ingestion()
