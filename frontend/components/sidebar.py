"""
Sidebar component for document upload and settings.
"""

import streamlit as st
from typing import Optional
from frontend.utils import get_api_client, initialize_session_state


def render_sidebar():
    """Render the sidebar with document upload and settings."""
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Backend health check
        st.subheader("🏥 System Status")
        api_client = get_api_client()
        
        try:
            health = api_client.health_check()
            if health.get("status") == "healthy":
                st.success("✅ Backend is healthy")
                
                # Show document count
                doc_count = health.get("components", {}).get("document_count", 0)
                st.metric("📚 Documents Indexed", doc_count)
            else:
                st.error(f"❌ Backend unhealthy: {health.get('error', 'Unknown error')}")
                st.info("Make sure the API server is running:\n```bash\nuvicorn src.api.main:app --reload\n```")
                return
                
        except Exception as e:
            st.error(f"❌ Cannot connect to backend: {str(e)}")
            st.info("Start the server:\n```bash\ncd D:\\GenAI\\enterprise-rag-ai-agents\npython -m uvicorn src.api.main:app --reload\n```")
            return
        
        st.divider()
        
        # Document Upload Section
        st.subheader("📤 Upload Documents")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md"],
            help="Upload PDF, DOCX, TXT, or Markdown files"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Wake up backend if on free tier (cold start takes ~30s)
                        health = api_client.health_check()
                        if health.get("status") != "healthy":
                            st.info("⏳ Waking up backend server (free tier)...")
                            import time
                            for _ in range(12):  # Retry up to 60s
                                time.sleep(5)
                                health = api_client.health_check()
                                if health.get("status") == "healthy":
                                    break
                        
                        # Upload to backend
                        result = api_client.upload_file_object(
                            uploaded_file,
                            uploaded_file.name
                        )
                        
                        st.success(f"✅ Uploaded: {result['filename']}")
                        st.info(f"📊 Status: {result.get('status', 'processing')}")
                        st.info(f"🆔 ID: {result['id']}")
                        
                        # Add to session state
                        st.session_state.uploaded_docs.append(result)
                        
                        # Clear file uploader
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Upload failed: {str(e)}")
        
        st.divider()
        
        # Document Management
        st.subheader("📁 Manage Documents")
        
        if st.button("🔄 Refresh Document List", use_container_width=True):
            try:
                docs = api_client.list_documents()
                st.session_state.uploaded_docs = docs.get("documents", [])
                st.success(f"Found {docs.get('total', 0)} documents")
            except Exception as e:
                st.error(f"Failed to load documents: {str(e)}")
        
        # Show uploaded documents
        if st.session_state.uploaded_docs:
            st.write(f"**{len(st.session_state.uploaded_docs)} documents indexed:**")
            
            for doc in st.session_state.uploaded_docs[:5]:  # Show first 5
                filename = doc.get("filename", doc.get("file_name", "Unknown"))
                doc_id = doc.get("id", doc.get("document_id", ""))
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📄 {filename}")
                with col2:
                    if st.button("🗑️", key=f"delete_{doc_id}", help="Delete document"):
                        try:
                            api_client.delete_document(doc_id)
                            st.success("Deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {str(e)}")
            
            if len(st.session_state.uploaded_docs) > 5:
                st.caption(f"... and {len(st.session_state.uploaded_docs) - 5} more")
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # Chat Settings
        st.subheader("💬 Chat Settings")
        
        st.session_state.show_sources = st.checkbox(
            "Show sources",
            value=st.session_state.get("show_sources", True),
            help="Display source documents with each answer"
        )
        
        st.session_state.show_metrics = st.checkbox(
            "Show metrics",
            value=st.session_state.get("show_metrics", False),
            help="Display performance metrics (latency, tokens)"
        )
        
        st.session_state.stream_responses = st.checkbox(
            "Stream responses",
            value=st.session_state.get("stream_responses", True),
            help="Stream responses in real-time"
        )
        
        st.divider()
        
        # Agent Settings
        st.subheader("🤖 AI Agent Mode")
        
        st.session_state.use_agents = st.checkbox(
            "Enable Multi-Agent System",
            value=st.session_state.get("use_agents", False),
            help="Use intelligent multi-agent orchestration for complex queries. Automatically activates for complex questions."
        )
        
        if st.session_state.use_agents:
            st.session_state.force_agents = st.checkbox(
                "Force Agent Mode",
                value=st.session_state.get("force_agents", False),
                help="Always use agents, even for simple queries (slower but more thorough)"
            )
            
            st.info(
                "🧠 **Agent System Active**\n\n"
                "Agents will automatically analyze query complexity and deploy:\n"
                "- 🔍 **SearchAgent** - Enhanced retrieval\n"
                "- 🎯 **AnalysisAgent** - Deep analysis\n"
                "- 🔬 **SynthesisAgent** - Comprehensive synthesis\n\n"
                "⚠️ Note: Agent mode disables streaming and takes longer (5-15s)"
            )
        else:
            st.caption("💡 Enable for complex queries requiring multi-step reasoning")
        
        st.divider()
        
        # Conversation Management
        st.subheader("🗨️ Conversation")
        
        if st.session_state.conversation_id:
            st.info(f"Active: {st.session_state.conversation_id[:8]}...")
        
        if st.button("🆕 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.success("Started new conversation!")
            st.rerun()
        
        if st.session_state.messages:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        
        try:
            stats = api_client.get_stats()
            
            st.metric(
                "Total Documents",
                stats.get("documents", {}).get("total_indexed", 0)
            )
            
            llm_stats = stats.get("llm", {})
            st.metric(
                "Total Queries",
                llm_stats.get("total_requests", 0)
            )
            
            if llm_stats.get("total_requests", 0) > 0:
                avg_latency = llm_stats.get("avg_latency", 0)
                st.metric(
                    "Avg Latency",
                    f"{avg_latency:.2f}s"
                )
                
        except Exception as e:
            st.caption(f"Stats unavailable: {str(e)}")
        
        st.divider()
        
        # Footer
        st.caption("🤖 Enterprise RAG System")
        st.caption("v1.0.0 | Built with ❤️")
