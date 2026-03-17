"""
RAG Chain Module - LangChain LCEL Edition
Combines retrieval and generation using LangChain Expression Language.
"""

from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import time

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.retrieval import RetrievalStrategy, Reranker, ContextBuilder
from src.retrieval import get_retriever, get_reranker, get_query_processor, get_context_builder
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


# LangChain LCEL prompt for RAG
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", """Context information is below:

{context}

Based on the above context, answer the following question:
Question: {query}

Instructions:
1. Structure your response with clear markdown formatting
2. Use ## for main headings, ### for subheadings
3. Use **bold** for key terms, bullet points for lists
4. **Cite sources using [1], [2], etc. at the END of sentences**
5. When available, mention PAGE NUMBERS and SECTIONS in citations
6. If the context doesn't contain enough information, say so clearly
7. Be confident and comprehensive when the context supports it"""),
])


class RAGChain:
    """
    LangChain LCEL RAG Chain.

    Workflow:
    1. Process query (expand, reformulate)
    2. Retrieve relevant documents
    3. Rerank results
    4. Build context
    5. Generate response with LangChain LCEL chain
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
        # Initialize components (reuse singletons to save memory)
        self.retriever = get_retriever(
            collection_name=collection_name,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k
        )
        self.reranker = get_reranker(strategy=rerank_strategy or "score")
        self.query_processor = get_query_processor()
        self.context_builder = get_context_builder()
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

        # Build LCEL chain: prompt | llm | output_parser
        self._chain = RAG_PROMPT | self.llm_client.get_chat_model() | StrOutputParser()

        logger.info(
            f"LangChain LCEL RAG chain initialized: "
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
    ) -> 'RAGResponse | Iterator[str]':
        """
        Query the RAG system using LangChain LCEL.
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

            # Step 6: Generate response using LCEL chain
            generation_start = time.time()

            if stream:
                return self._generate_stream(
                    query, context.text, conversation_history,
                    conversation_id, reranked_docs, context,
                    start_time, retrieval_time
                )
            else:
                # Invoke the LCEL chain
                chain_input = {
                    "system_prompt": self.system_prompt,
                    "context": context.text,
                    "query": query,
                    "history": self._format_history(conversation_history),
                }
                answer = self._chain.invoke(chain_input)
                generation_time = time.time() - generation_start

                # Save to conversation if enabled
                if self.use_conversation and conversation_id:
                    self.conversation_manager.add_message(
                        conversation_id, "user", query
                    )
                    self.conversation_manager.add_message(
                        conversation_id, "assistant", answer
                    )

                # Build response
                total_latency = time.time() - start_time

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
                    f"LCEL RAG query completed: {total_latency:.2f}s "
                    f"({retrieval_time:.2f}s retrieval, {generation_time:.2f}s generation)"
                )

                perf_monitor.increment_counter("rag_queries", 1)

                return RAGResponse(
                    answer=answer,
                    sources=sources,
                    query=query,
                    tokens_used={"prompt": 0, "completion": 0, "total": 0},
                    latency=total_latency,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    context_used=context.text,
                    metadata={
                        "num_sources": len(sources),
                        "context_tokens": context.total_tokens,
                        "truncated": context.truncated,
                        "intent": processed_query.intent,
                        "chain": "langchain-lcel"
                    }
                )
        finally:
            perf_monitor.stop_timer("rag_query")

    def _generate_stream(
        self,
        query: str,
        context_text: str,
        conversation_history: List[Dict[str, str]],
        conversation_id: Optional[str],
        sources: List[Any],
        context: Any,
        start_time: float,
        retrieval_time: float
    ) -> Iterator[str]:
        """Generate streaming response using LCEL chain.stream()."""
        generation_start = time.time()

        chain_input = {
            "system_prompt": self.system_prompt,
            "context": context_text,
            "query": query,
            "history": self._format_history(conversation_history),
        }

        full_response = []
        for chunk in self._chain.stream(chain_input):
            full_response.append(chunk)
            yield chunk

        generation_time = time.time() - generation_start

        if self.use_conversation and conversation_id:
            complete_response = "".join(full_response)
            self.conversation_manager.add_message(
                conversation_id, "user", query
            )
            self.conversation_manager.add_message(
                conversation_id, "assistant", complete_response
            )

        logger.info(
            f"LCEL streaming completed: {time.time() - start_time:.2f}s total "
            f"({retrieval_time:.2f}s retrieval, {generation_time:.2f}s generation)"
        )

    def _format_history(self, history: List[Dict[str, str]]) -> List:
        """Convert dict history to LangChain message tuples for MessagesPlaceholder."""
        if not history:
            return []
        from langchain_core.messages import HumanMessage, AIMessage
        messages = []
        for msg in history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
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
    ) -> 'RAGResponse | Iterator[str]':
        """Conversational interface for RAG."""
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
            ),
            "chain": "langchain-lcel"
        }


# Convenience function
def ask(
    query: str,
    collection_name: str = "documents",
    stream: bool = False
) -> 'RAGResponse | Iterator[str]':
    """Quick RAG query."""
    chain = RAGChain(collection_name=collection_name)
    return chain.query(query, stream=stream)
