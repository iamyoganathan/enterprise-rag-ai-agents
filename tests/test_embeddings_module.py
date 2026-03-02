"""
Test Module 2: Embeddings
Tests embedding generation, vector storage, and indexing pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from src.embeddings.vector_store import VectorStore
from src.embeddings.indexing import IndexingPipeline
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunker import Chunk


def test_embedding_model():
    """Test the embedding model."""
    print("\n" + "="*60)
    print("TEST 1: Embedding Model")
    print("="*60)
    
    # Initialize model
    model = EmbeddingModel()
    print(f"\nModel Info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test single text embedding
    text = "This is a test sentence for embedding."
    embedding = model.encode(text)
    print(f"\nSingle Text Embedding:")
    print(f"  Text: {text}")
    print(f"  Shape: {embedding.shape}")
    print(f"  Norm: {embedding.dot(embedding)**0.5:.4f}")
    
    # Test batch embedding
    texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning algorithms learn from data.",
        "Natural language processing handles human language."
    ]
    embeddings = model.encode_batch(texts, show_progress=False)
    print(f"\nBatch Embeddings:")
    print(f"  Texts: {len(texts)}")
    print(f"  Shape: {embeddings.shape}")
    
    # Test similarity
    sim_01 = model.similarity(embeddings[0], embeddings[1])
    sim_02 = model.similarity(embeddings[0], embeddings[2])
    print(f"\nSimilarity Scores:")
    print(f"  Text 0 vs Text 1: {sim_01:.4f}")
    print(f"  Text 0 vs Text 2: {sim_02:.4f}")
    
    print("\n✓ Embedding Model Test Passed!")
    return True


def test_vector_store():
    """Test the vector store."""
    print("\n" + "="*60)
    print("TEST 2: Vector Store")
    print("="*60)
    
    # Initialize vector store with test collection
    store = VectorStore(collection_name="test_embeddings")
    
    # Clear any existing data
    store.reset()
    
    print(f"\nCollection Info:")
    info = store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Add documents
    documents = [
        "Python is a high-level programming language.",
        "JavaScript is used for web development.",
        "SQL is used for database queries.",
        "Docker enables containerized applications.",
        "Kubernetes orchestrates container deployments."
    ]
    metadatas = [
        {"category": "programming", "language": "python"},
        {"category": "programming", "language": "javascript"},
        {"category": "database", "language": "sql"},
        {"category": "devops", "tool": "docker"},
        {"category": "devops", "tool": "kubernetes"}
    ]
    
    print(f"\nAdding {len(documents)} documents...")
    ids = store.add_documents(documents, metadatas)
    print(f"  Added with IDs: {[id[:8] for id in ids]}")
    print(f"  Total documents: {store.count()}")
    
    # Search
    query = "What programming languages are available?"
    print(f"\nSearching: '{query}'")
    results = store.similarity_search(query, k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Text: {result['document']}")
        print(f"    Similarity: {result['similarity']:.4f}")
        print(f"    Category: {result['metadata'].get('category', 'N/A')}")
    
    # Filtered search
    print(f"\nFiltered Search (category=devops):")
    results = store.similarity_search(
        "container management",
        k=2,
        filter={"category": "devops"}
    )
    
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['document']} (sim: {result['similarity']:.4f})")
    
    # Cleanup
    print(f"\nCleaning up test collection...")
    store.reset()
    print(f"  Documents after reset: {store.count()}")
    
    print("\n✓ Vector Store Test Passed!")
    return True


def test_indexing_pipeline():
    """Test the complete indexing pipeline."""
    print("\n" + "="*60)
    print("TEST 3: Indexing Pipeline")
    print("="*60)
    
    # Create sample document
    sample_text = """
# Machine Learning Guide

## Introduction
Machine learning is a subset of artificial intelligence that focuses on 
building systems that learn from and make decisions based on data.

## Types of Machine Learning

### Supervised Learning
In supervised learning, the algorithm learns from labeled training data.
Common applications include classification and regression tasks.

### Unsupervised Learning
Unsupervised learning works with unlabeled data to find hidden patterns.
Clustering and dimensionality reduction are typical use cases.

### Reinforcement Learning
Reinforcement learning trains agents to make decisions through rewards.
Applications include game playing and robotics control systems.
"""
    
    # Create sample file
    sample_dir = Path("data/sample_documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_dir / "ml_guide.md"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    print(f"\nCreated test document: {sample_file.name}")
    
    # Step 1: Ingest the document
    print("\nStep 1: Ingesting document...")
    ingestion_pipeline = IngestionPipeline(chunk_size=200, chunk_overlap=20)
    ingestion_result = ingestion_pipeline.process_file(sample_file)
    
    chunks = ingestion_result["chunks"]
    metadata = ingestion_result["metadata"]
    
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Document category: {metadata.get('category', 'unknown')}")
    
    # Step 2: Index the chunks
    print("\nStep 2: Indexing chunks...")
    indexing_pipeline = IndexingPipeline(collection_name="test_indexing")
    indexing_pipeline.reset()  # Clear any existing data
    
    doc_metadata = {
        "file_name": sample_file.name,
        "file_path": str(sample_file),
        "category": metadata.get("category", "unknown")
    }
    
    index_result = indexing_pipeline.index_document(chunks, doc_metadata)
    print(f"  Chunks indexed: {index_result['chunks_indexed']}")
    print(f"  Status: {index_result['status']}")
    
    # Step 3: Search the index
    print("\nStep 3: Searching indexed content...")
    
    queries = [
        "What is supervised learning?",
        "Explain reinforcement learning",
        "Types of machine learning"
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = indexing_pipeline.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"    {i}. Similarity: {result['similarity']:.4f}")
            print(f"       Text: {result['document'][:80]}...")
    
    # Step 4: Statistics
    print("\nPipeline Statistics:")
    stats = indexing_pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    print(f"\nCleaning up...")
    indexing_pipeline.reset()
    sample_file.unlink()
    
    print("\n✓ Indexing Pipeline Test Passed!")
    return True


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Workflow")
    print("="*60)
    
    # Create multiple test documents
    sample_dir = Path("data/sample_documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    documents = {
        "python_basics.md": """
# Python Programming

Python is a versatile programming language known for its simplicity.
It supports multiple programming paradigms including procedural and object-oriented.
Python has extensive libraries for data science and machine learning.
""",
        "web_development.txt": """
Web development involves creating websites and web applications.
Frontend technologies include HTML, CSS, and JavaScript.
Backend development uses languages like Python, Node.js, and Ruby.
Frameworks like React and Vue.js make frontend development easier.
""",
        "data_science.txt": """
Data science combines statistics, programming, and domain knowledge.
Key tools include Python, R, and SQL for data analysis.
Machine learning and AI are important components of data science.
Data visualization helps communicate insights effectively.
"""
    }
    
    # Write documents
    print(f"\nCreating {len(documents)} test documents...")
    for filename, content in documents.items():
        filepath = sample_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Created: {filename}")
    
    # Initialize pipelines
    ingestion_pipeline = IngestionPipeline(chunk_size=150, chunk_overlap=20)
    indexing_pipeline = IndexingPipeline(collection_name="test_e2e")
    indexing_pipeline.reset()
    
    # Process all documents
    print(f"\nProcessing and indexing documents...")
    for filename in documents.keys():
        filepath = sample_dir / filename
        
        # Ingest
        result = ingestion_pipeline.process_file(filepath)
        print(f"  Ingested: {filename} ({len(result['chunks'])} chunks)")
        
        # Index
        doc_meta = {
            "file_name": filename,
            "category": result["metadata"].get("category", "unknown")
        }
        indexing_pipeline.index_document(result["chunks"], doc_meta)
    
    # Search across all documents
    print(f"\nSearching across all indexed documents...")
    test_queries = [
        "Tell me about Python programming",
        "What is web development?",
        "Explain data science tools"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = indexing_pipeline.search(query, k=2)
        
        for i, result in enumerate(results, 1):
            print(f"    {i}. {result['metadata'].get('file_name', 'unknown')}")
            print(f"       Similarity: {result['similarity']:.4f}")
            print(f"       Snippet: {result['document'][:60]}...")
    
    # Final statistics
    print(f"\nFinal Statistics:")
    stats = indexing_pipeline.get_stats()
    print(f"  Total documents: {stats['documents_indexed']}")
    print(f"  Total chunks: {stats['chunks_indexed']}")
    print(f"  Vector store count: {stats['vector_store_count']}")
    
    # Cleanup
    print(f"\nCleaning up...")
    indexing_pipeline.reset()
    for filename in documents.keys():
        (sample_dir / filename).unlink()
    
    print("\n✓ End-to-End Test Passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MODULE 2: EMBEDDINGS - COMPREHENSIVE TEST")
    print("="*60)
    
    try:
        # Run all tests
        tests = [
            ("Embedding Model", test_embedding_model),
            ("Vector Store", test_vector_store),
            ("Indexing Pipeline", test_indexing_pipeline),
            ("End-to-End Workflow", test_end_to_end)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"\n✗ {test_name} Test Failed: {str(e)}")
                failed += 1
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  Tests Passed: {passed}/{len(tests)}")
        print(f"  Tests Failed: {failed}/{len(tests)}")
        
        if failed == 0:
            print("\n✓ ALL TESTS PASSED!")
            print("\nModule 2 (Embeddings) is working correctly!")
            print("Ready to proceed to Module 3 (Retrieval)")
        else:
            print(f"\n✗ {failed} test(s) failed. Please review errors above.")
        
    except Exception as e:
        print(f"\nTest suite error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
