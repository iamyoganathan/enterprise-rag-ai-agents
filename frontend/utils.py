"""
Utility functions for the Streamlit frontend.
API client wrapper for backend communication.
"""

import requests
from typing import Dict, List, Optional, Any, Iterator
import streamlit as st
from pathlib import Path
import os


class APIClient:
    """Client for interacting with the FastAPI backend."""
    
    def __init__(self, base_url: str = None):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the FastAPI backend (defaults to localhost or env variable)
        """
        # Priority: 1. Passed argument, 2. Streamlit secrets, 3. Environment variable, 4. Localhost
        if base_url is None:
            # Check Streamlit secrets (for Streamlit Cloud)
            if hasattr(st, 'secrets') and 'API_BASE_URL' in st.secrets:
                base_url = st.secrets['API_BASE_URL']
            # Check environment variable
            elif 'API_BASE_URL' in os.environ:
                base_url = os.environ['API_BASE_URL']
            # Default to localhost
            else:
                base_url = "http://localhost:8000"
        
        self.base_url = base_url.rstrip("/")
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the backend is healthy.
        
        Returns:
            Health check response
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            System statistics
        """
        response = requests.get(f"{self.base_url}/api/stats")
        response.raise_for_status()
        return response.json()
    
    def upload_document(self, file_path: Path) -> Dict[str, Any]:
        """Upload a document for processing.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Upload response with document ID and status
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            response = requests.post(
                f"{self.base_url}/api/documents/upload",
                files=files,
                timeout=300  # 5 minutes for large documents
            )
        response.raise_for_status()
        return response.json()
    
    def upload_file_object(self, file_obj, filename: str) -> Dict[str, Any]:
        """Upload a file object (from Streamlit file uploader).
        
        Args:
            file_obj: File object from st.file_uploader
            filename: Original filename
            
        Returns:
            Upload response
        """
        files = {"file": (filename, file_obj, "application/octet-stream")}
        response = requests.post(
            f"{self.base_url}/api/documents/upload",
            files=files,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    
    def list_documents(self) -> Dict[str, Any]:
        """List all indexed documents.
        
        Returns:
            List of documents with metadata
        """
        response = requests.get(f"{self.base_url}/api/documents/list")
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Delete confirmation
        """
        response = requests.delete(f"{self.base_url}/api/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        stream: bool = False,
        use_agents: bool = False,
        force_agents: bool = False
    ) -> Dict[str, Any]:
        """Send a chat message.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID for multi-turn chat
            stream: Whether to stream the response
            use_agents: Enable multi-agent orchestration
            force_agents: Force agent usage regardless of complexity
            
        Returns:
            Chat response with answer and sources
        """
        data = {
            "message": message,
            "conversation_id": conversation_id,
            "stream": stream,
            "use_agents": use_agents,
            "force_agents": force_agents
        }
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    
    def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        use_agents: bool = False,
        force_agents: bool = False
    ) -> Iterator[str]:
        """Send a chat message with streaming response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            use_agents: Enable multi-agent orchestration (not supported with streaming)
            force_agents: Force agent usage (not supported with streaming)
            
        Yields:
            Response chunks as they arrive
        """
        import json
        
        data = {
            "message": message,
            "conversation_id": conversation_id,
            "stream": True,
            "use_agents": use_agents,
            "force_agents": force_agents
        }
        
        with requests.post(
            f"{self.base_url}/api/chat/stream",
            json=data,
            stream=True,
            timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data_obj = json.loads(data_str)
                        if data_obj.get("type") == "chunk":
                            yield data_obj.get("content", "")
                    except json.JSONDecodeError:
                        continue
    
    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation history with messages
        """
        response = requests.get(
            f"{self.base_url}/api/chat/history/{conversation_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Search results with documents and scores
        """
        data = {
            "query": query,
            "top_k": top_k,
            "filter": filter_dict
        }
        
        response = requests.post(f"{self.base_url}/api/search", json=data)
        response.raise_for_status()
        return response.json()


@st.cache_resource
def get_api_client() -> APIClient:
    """Get a cached API client instance.
    
    Returns:
        APIClient instance
    """
    return APIClient()


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format source documents for display with detailed metadata.
    
    Args:
        sources: List of source documents
        
    Returns:
        Formatted string with sources including page/section info
    """
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        # Get basic info
        # source.get("source") = filename, source.get("document") = content
        doc_name = source.get("source", source.get("document", "Unknown"))
        score = source.get("score", 0)
        content = source.get("document", "")
        metadata = source.get("metadata", {})
        
        # Build detailed reference info
        ref_parts = []
        
        # Page number
        if "page_number" in metadata:
            ref_parts.append(f"📄 Page {metadata['page_number']}")
        elif "page" in metadata:
            ref_parts.append(f"📄 Page {metadata['page']}")
        
        # Section
        if "section" in metadata:
            ref_parts.append(f"📑 {metadata['section']}")
        elif "heading" in metadata:
            ref_parts.append(f"📑 {metadata['heading']}")
        
        # Chunk info
        if "chunk_index" in metadata:
            ref_parts.append(f"🔢 Chunk {metadata['chunk_index']}")
        elif "chunk_id" in metadata:
            chunk_id = metadata['chunk_id']
            # Extract number from chunk_id if it's like 'chunk_5'
            if isinstance(chunk_id, str) and '_' in chunk_id:
                chunk_num = chunk_id.split('_')[-1]
                ref_parts.append(f"🔢 Chunk {chunk_num}")
        
        # Format reference line
        ref_info = " | ".join(ref_parts) if ref_parts else "No location info"
        
        # Truncate content if too long
        if len(content) > 200:
            content = content[:200] + "..."
        
        # Build formatted source
        formatted.append(
            f"**[{i}] {doc_name}**\n"
            f"📍 {ref_info} | ⭐ Score: {score:.3f}\n"
            f"> {content}"
        )
    
    return "\n\n".join(formatted)


def format_metrics(response: Dict[str, Any]) -> str:
    """Format response metrics for display.
    
    Args:
        response: Chat response with metrics
        
    Returns:
        Formatted metrics string
    """
    tokens = response.get("tokens_used", 0)
    latency = response.get("latency", 0)
    retrieval_time = response.get("retrieval_time", 0)
    generation_time = response.get("generation_time", 0)
    
    return (
        f"⏱️ **Latency:** {latency:.2f}s | "
        f"🔍 **Retrieval:** {retrieval_time:.2f}s | "
        f"🤖 **Generation:** {generation_time:.2f}s | "
        f"📊 **Tokens:** {tokens}"
    )


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    
    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = False
    
    if "stream_responses" not in st.session_state:
        st.session_state.stream_responses = True
    
    if "use_agents" not in st.session_state:
        st.session_state.use_agents = False
    
    if "force_agents" not in st.session_state:
        st.session_state.force_agents = False
