"""
Chat API Endpoints
Handle RAG-based chat and question answering.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import asyncio
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: Optional[str] = Field(default=None, description="User message (alias for query)")
    query: Optional[str] = Field(default=None, description="User question")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for multi-turn chat")
    stream: bool = Field(default=False, description="Enable streaming response")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    use_reranking: bool = Field(default=True, description="Use reranking for better results")
    use_agents: bool = Field(default=False, description="Enable multi-agent orchestration for complex queries")
    force_agents: bool = Field(default=False, description="Force agent usage regardless of complexity")
    
    @property
    def user_query(self) -> str:
        """Get the user query from either message or query field."""
        return self.message or self.query or ""


class Source(BaseModel):
    """Source document info."""
    document: str
    score: float
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    sources: List[Source]
    conversation_id: Optional[str] = None
    tokens_used: int
    latency: float
    retrieval_time: float
    generation_time: float


class ConversationMessage(BaseModel):
    """Conversation message."""
    role: str
    content: str
    timestamp: float


class ConversationHistory(BaseModel):
    """Conversation history response."""
    conversation_id: str
    messages: List[ConversationMessage]
    created_at: float
    updated_at: float


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a question and get an AI-generated answer with sources.
    
    Supports:
    - Single-turn Q&A
    - Multi-turn conversations (provide conversation_id)
    - Agent orchestration for complex queries (use_agents=true)
    """
    
    try:
        logger.info(
            f"Chat query: '{request.user_query}' | "
            f"Conversation: {request.conversation_id} | "
            f"Top K: {request.top_k} | "
            f"Agents: {request.use_agents}"
        )
        
        from src.llm import RAGChain, get_conversation_manager
        from src.agents.agent_rag_chain import get_agent_rag_chain
        
        # Choose RAG implementation
        if request.use_agents:
            # Use agent-enhanced RAG
            rag = get_agent_rag_chain(
                collection_name="documents",
                enable_agents=True,
                agent_threshold="moderate"
            )
            
            response = rag.query(
                query=request.user_query,
                force_agents=request.force_agents
            )
        else:
            # Use basic RAG chain
            rag = RAGChain(
                collection_name="documents",
                retrieval_strategy="semantic",
                rerank_strategy="mmr" if request.use_reranking else None,
                top_k=request.top_k,
                use_conversation=request.conversation_id is not None
            )
            
            # Generate response
            if request.conversation_id:
                response = rag.chat(
                    message=request.user_query,
                    conversation_id=request.conversation_id,
                    create_conversation=True,
                    temperature=request.temperature
                )
            else:
                response = rag.query(
                    query=request.user_query,
                    temperature=request.temperature
                )
        
        # Format sources
        sources = [
            Source(
                document=src["document"],
                score=src["score"],
                metadata=src["metadata"]
            )
            for src in response.sources
        ]
        
        # Get total tokens
        if isinstance(response.tokens_used, dict):
            total_tokens = response.tokens_used.get("total", 0)
        else:
            total_tokens = response.tokens_used
        
        logger.info(
            f"Chat completed: {len(sources)} sources, "
            f"{total_tokens} tokens, "
            f"{response.latency:.2f}s, "
            f"mode={response.metadata.get('mode', 'basic')}"
        )
        
        return ChatResponse(
            answer=response.answer,
            sources=sources,
            conversation_id=response.metadata.get("conversation_id"),
            tokens_used=total_tokens,
            latency=response.latency,
            retrieval_time=response.retrieval_time,
            generation_time=response.generation_time
        )
    
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Stream chat responses chunk by chunk in real-time.
    """
    return StreamingResponse(
        _stream_generator(request),
        media_type="text/event-stream"
    )


async def _stream_generator(request: ChatRequest):
    """
    Internal generator for streaming chat responses.
    """
    try:
        # Initialize RAG chain
        from src.llm import RAGChain
        
        rag = RAGChain(
            collection_name="documents",
            retrieval_strategy="semantic",
            rerank_strategy="mmr" if request.use_reranking else None,
            top_k=request.top_k,
            use_conversation=request.conversation_id is not None
        )
        
        # Stream response
        if request.conversation_id:
            response_gen = rag.chat(
                message=request.user_query,
                conversation_id=request.conversation_id,
                create_conversation=True,
                stream=True
            )
        else:
            response_gen = rag.query(
                query=request.user_query,
                temperature=request.temperature,
                stream=True
            )
        
        # Check if response is a RAGResponse object (no documents found) or iterator
        if hasattr(response_gen, 'answer'):
            # It's a RAGResponse object, not a stream
            yield f"data: {json.dumps({'type': 'chunk', 'content': response_gen.answer})}\n\n"
            response_dict = {
                'answer': response_gen.answer,
                'sources': response_gen.sources,
                'query': response_gen.query,
                'tokens_used': response_gen.tokens_used,
                'latency': response_gen.latency,
                'retrieval_time': response_gen.retrieval_time,
                'generation_time': response_gen.generation_time,
                'metadata': response_gen.metadata
            }
            yield f"data: {json.dumps({'type': 'complete', 'response': response_dict})}\n\n"
        else:
            # Stream chunks
            for chunk in response_gen:
                if isinstance(chunk, str):
                    # Text chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                else:
                    # Final response with metadata
                    yield f"data: {json.dumps({'type': 'complete', 'response': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        logger.error(f"Stream failed: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@router.get("/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history.
    """
    try:
        from src.llm import get_conversation_manager
        
        conv_manager = get_conversation_manager()
        conversation = conv_manager.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        messages = [
            ConversationMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp
            )
            for msg in conversation.messages
        ]
        
        return ConversationHistory(
            conversation_id=conversation.id,
            messages=messages,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def list_conversations():
    """
    List all conversations.
    """
    try:
        from src.llm import get_conversation_manager
        
        conv_manager = get_conversation_manager()
        conversations = conv_manager.list_conversations()
        
        return {
            "conversations": [
                {
                    "conversation_id": conv_id,
                    "created_at": conv_info.get("created_at", 0),
                    "message_count": conv_info.get("message_count", 0)
                }
                for conv_id, conv_info in conversations.items()
            ],
            "total": len(conversations)
        }
    
    except Exception as e:
        logger.error(f"Failed to list conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation.
    """
    try:
        from src.llm import get_conversation_manager
        
        conv_manager = get_conversation_manager()
        conv_manager.delete_conversation(conversation_id)
        
        return {
            "status": "success",
            "message": f"Conversation {conversation_id} deleted"
        }
    
    except Exception as e:
        logger.error(f"Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quick-ask")
async def quick_ask(
    question: str,
    top_k: int = 3
):
    """
    Quick Q&A endpoint - simplified interface for single questions.
    """
    try:
        from src.llm import RAGChain
        
        rag = RAGChain(
            collection_name="documents",
            retrieval_strategy="semantic",
            top_k=top_k
        )
        
        response = rag.query(question)
        
        return {
            "question": question,
            "answer": response.answer,
            "sources_count": len(response.sources),
            "confidence": "high" if response.sources else "low"
        }
    
    except Exception as e:
        logger.error(f"Quick ask failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
