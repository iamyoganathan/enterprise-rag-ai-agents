"""
RAG Chain Module
Combines retrieval and generation for question-answering.
This is the core RAG pipeline that ties everything together.
"""

from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import time

from src.retrieval import Retriever, RetrievalStrategy, Reranker, ContextBuilder
from src.retrieval.query_processor import QueryProcessor
from src.llm.llm_client import LLMClient, get_llm_client
from src.llm.prompt_templates import get_template_manager
from src.llm.conversation import ConversationManager, get_conversation_manager
from src.utils.logger import get_logger
from src.utils.monitoring import get_performance_monitor

logger = get_logger(__name__)
perf_monitor = get_performance_monitor()


@dataclass
class RAGResponse:
    """Response from RAG chain."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    tokens_used: Dict[str, int]
    latency: float
    retrieval_time: float
    generation_time: float
    context_used: str
    metadata: Dict[str, Any]


class RAGChain:
    """
    Complete RAG (Retrieval-Augmented Generation) chain.
    
    Workflow:
    1. Process query (expand, reformulate)
    2. Retrieve relevant documents
    3. Rerank results
    4. Build context
    5. Generate response with LLM
    6. Return answer with sources
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        retrieval_strategy: str = "semantic",
        rerank_strategy: str = "mmr",
        top_k: int = 5,
        temperature: float = 0.7,
        max_context_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        use_conversation: bool = False
    ):
        """
        Initialize RAG chain.
        
        Args:
            collection_name: Vector store collection
            retrieval_strategy: Retrieval strategy
            rerank_strategy: Reranking strategy
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            max_context_tokens: Maximum context size
            system_prompt: Optional system prompt
            use_conversation: Whether to use conversation history
        """
        # Initialize components
        self.retriever = Retriever(
            collection_name=collection_name,
            strategy=RetrievalStrategy(retrieval_strategy),
            top_k=top_k
        )
        self.reranker = Reranker(strategy=rerank_strategy)
        self.query_processor = QueryProcessor()
        self.context_builder = ContextBuilder(max_tokens=max_context_tokens)
        self.llm_client = get_llm_client()
        self.template_manager = get_template_manager()
        
        # Configuration
        self.temperature = temperature
        self.system_prompt = system_prompt or self.template_manager.get_system_prompt("rag")
        self.use_conversation = use_conversation
        
        if use_conversation:
            self.conversation_manager = get_conversation_manager()
        else:
            self.conversation_manager = None
        
        logger.info(
            f"RAG chain initialized: "
            f"collection={collection_name}, "
            f"strategy={retrieval_strategy}, "
            f"top_k={top_k}"
        )
    
    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> RAGResponse | Iterator[str]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for history
            stream: Whether to stream the response
            **kwargs: Additional arguments (filters, etc.)
            
        Returns:
            RAGResponse or stream iterator
        """
        logger.info(f"RAG query: '{query[:50]}...'")
        
        start_time = time.time()
        
        perf_monitor.start_timer("rag_query")
        try:
            # Step 1: Process query
            processed_query = self.query_processor.process(query)
            logger.debug(f"Query intent: {processed_query.intent}")
            
            # Step 2: Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = self.retriever.retrieve(
                processed_query.cleaned,
                filter=kwargs.get('filter')
            )
            retrieval_time = time.time() - retrieval_start
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
            
            if not retrieved_docs:
                # No documents found
                no_docs_response = self._handle_no_documents(query)
                return RAGResponse(
                    answer=no_docs_response,
                    sources=[],
                    query=query,
                    tokens_used={"prompt": 0, "completion": 0, "total": 0},
                    latency=time.time() - start_time,
                    retrieval_time=retrieval_time,
                    generation_time=0,
                    context_used="",
                    metadata={"status": "no_documents"}
                )
            
            # Step 3: Rerank results
            reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.retriever.top_k)
            
            # Step 4: Build context
            context = self.context_builder.build_context(reranked_docs, query)
            
            logger.debug(
                f"Context built: {context.num_chunks} chunks, "
                f"{context.total_tokens} tokens"
            )
            
            # Step 5: Get conversation history if enabled
            conversation_history = []
            if self.use_conversation and conversation_id:
                conversation_history = self.conversation_manager.get_history_for_llm(
                    conversation_id,
                    include_system=False
                )
            
            # Step 6: Build prompt
            prompt = self._build_prompt(query, context.text, conversation_history)
            
            # Step 7: Generate response
            generation_start = time.time()
            
            if stream:
                # Return streaming generator
                return self._generate_stream(
                    query,
                    prompt,
                    conversation_id,
                    reranked_docs,
                    context,
                    start_time,
                    retrieval_time
                )
            else:
                # Generate complete response
                llm_response = self.llm_client.generate(
                    messages=prompt,
                    temperature=self.temperature
                )
                generation_time = time.time() - generation_start
                
                # Step 8: Save to conversation if enabled
                if self.use_conversation and conversation_id:
                    self.conversation_manager.add_message(
                        conversation_id, "user", query
                    )
                    self.conversation_manager.add_message(
                        conversation_id, "assistant", llm_response.content
                    )
                
                # Step 9: Build response
                total_latency = time.time() - start_time
                
                # Extract sources
                sources = [
                    {
                        "document": doc.document,
                        "score": doc.final_score,
                        "source": doc.metadata.get("file_name", "unknown"),
                        "metadata": doc.metadata
                    }
                    for doc in reranked_docs
                ]
                
                logger.info(
                    f"RAG query completed: {total_latency:.2f}s "
                    f"({retrieval_time:.2f}s retrieval, {generation_time:.2f}s generation)"
                )
                
                perf_monitor.increment_counter("rag_queries", 1)
                
                return RAGResponse(
                    answer=llm_response.content,
                    sources=sources,
                    query=query,
                    tokens_used=llm_response.tokens_used,
                    latency=total_latency,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    context_used=context.text,
                    metadata={
                        "num_sources": len(sources),
                        "context_tokens": context.total_tokens,
                        "truncated": context.truncated,
                        "intent": processed_query.intent
                    }
                )
        finally:
            perf_monitor.stop_timer("rag_query")
    
    def _generate_stream(
        self,
        query: str,
        prompt: List[Dict[str, str]],
        conversation_id: Optional[str],
        sources: List[Any],
        context: Any,
        start_time: float,
        retrieval_time: float
    ) -> Iterator[str]:
        """Generate streaming response."""
        
        generation_start = time.time()
        
        # Stream the response
        stream = self.llm_client.generate(
            messages=prompt,
            temperature=self.temperature,
            stream=True
        )
        
        # Collect chunks for conversation history
        full_response = []
        
        for chunk in stream:
            full_response.append(chunk)
            yield chunk
        
        generation_time = time.time() - generation_start
        
        # Save to conversation if enabled
        if self.use_conversation and conversation_id:
            complete_response = "".join(full_response)
            self.conversation_manager.add_message(
                conversation_id, "user", query
            )
            self.conversation_manager.add_message(
                conversation_id, "assistant", complete_response
            )
        
        logger.info(
            f"Streaming completed: {time.time() - start_time:.2f}s total "
            f"({retrieval_time:.2f}s retrieval, {generation_time:.2f}s generation)"
        )
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Build prompt messages for LLM."""
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history (if not too long)
        if history:
            # Limit history to avoid token overflow
            messages.extend(history[-6:])  # Last 3 exchanges
        
        # Add context and query
        user_prompt = self.template_manager.format_rag_prompt(
            query=query,
            context=context,
            template_name="qa_with_citations"
        )
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def _handle_no_documents(self, query: str) -> str:
        """Handle case when no documents are found."""
        
        return (
            "I couldn't find any relevant information in the knowledge base "
            "to answer your question. The query might be too specific, or "
            "the required information hasn't been indexed yet."
        )
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        create_conversation: bool = True,
        stream: bool = False
    ) -> RAGResponse | Iterator[str]:
        """
        Conversational interface for RAG.
        
        Args:
            message: User message
            conversation_id: Conversation ID
            create_conversation: Create conversation if not exists
            stream: Stream response
            
        Returns:
            RAGResponse or stream iterator
        """
        # Create or get conversation
        if self.use_conversation:
            if conversation_id is None and create_conversation:
                conv = self.conversation_manager.create_conversation(
                    system_prompt=self.system_prompt
                )
                conversation_id = conv.id
            elif conversation_id and not self.conversation_manager.get_conversation(conversation_id):
                if create_conversation:
                    self.conversation_manager.create_conversation(
                        conversation_id=conversation_id,
                        system_prompt=self.system_prompt
                    )
        
        return self.query(message, conversation_id=conversation_id, stream=stream)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        
        return {
            "retriever": self.retriever.get_stats(),
            "llm": self.llm_client.get_stats(),
            "conversation_manager": (
                len(self.conversation_manager.list_conversations())
                if self.conversation_manager else 0
            )
        }


# Convenience function

def ask(
    query: str,
    collection_name: str = "documents",
    stream: bool = False
) -> RAGResponse | Iterator[str]:
    """
    Quick RAG query.
    
    Args:
        query: User query
        collection_name: Collection to search
        stream: Stream response
        
    Returns:
        RAGResponse or stream iterator
    """
    chain = RAGChain(collection_name=collection_name)
    return chain.query(query, stream=stream)


if __name__ == "__main__":
    # Test RAG chain
    print("Testing RAG Chain\n" + "="*60)
    
    # Note: Requires indexed documents
    from src.embeddings import IndexingPipeline
    from src.ingestion import Chunk
    
    # Setup test data
    print("Setting up test data...")
    indexing = IndexingPipeline(collection_name="test_rag")
    indexing.reset()
    
    test_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing (NLP) helps computers understand and generate human language.",
        "Data science combines statistics, programming, and domain expertise."
    ]
    
    chunks = [
        Chunk(text, f"chunk_{i}", 0, len(text), {"topic": "tech"})
        for i, text in enumerate(test_docs)
    ]
    
    indexing.index_chunks(chunks, show_progress=False)
    print(f"Indexed {len(chunks)} documents\n")
    
    # Initialize RAG chain
    rag = RAGChain(
        collection_name="test_rag",
        retrieval_strategy="semantic",
        top_k=3
    )
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Explain machine learning",
        "What is deep learning?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        response = rag.query(query)
        
        print(f"\nAnswer: {response.answer}")
        print(f"\nSources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. {source['document'][:60]}... (score: {source['score']:.3f})")
        
        print(f"\nMetrics:")
        print(f"  Total time: {response.latency:.2f}s")
        print(f"  Retrieval: {response.retrieval_time:.2f}s")
        print(f"  Generation: {response.generation_time:.2f}s")
        print(f"  Tokens: {response.tokens_used['total']}")
    
    # Test streaming
    print(f"\n" + "="*60)
    print("Test: Streaming Response")
    print("="*60)
    
    print(f"\nQuery: How does NLP work?")
    print("Answer (streaming): ", end="", flush=True)
    
    stream = rag.query("How does NLP work?", stream=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
    print("\n")
    
    # Statistics
    print("="*60)
    print("RAG Statistics")
    print("="*60)
    stats = rag.get_stats()
    print(f"Retriever queries: {stats['retriever']['total_queries']}")
    print(f"LLM requests: {stats['llm']['total_requests']}")
    print(f"Total tokens: {stats['llm']['total_tokens']}")
    
    # Cleanup
    indexing.reset()
    
    print("\nRAG chain test completed!")
