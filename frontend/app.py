"""
Streamlit Frontend Application for Enterprise RAG System.

This is the main entry point for the chat interface.
Run with: streamlit run frontend/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from frontend.utils import initialize_session_state
from frontend.components.sidebar import render_sidebar
from frontend.components.chat import (
    render_chat_interface,
    handle_user_input,
    render_welcome_message
)


# Page configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/enterprise-rag-system',
        'Report a bug': 'https://github.com/yourusername/enterprise-rag-system/issues',
        'About': """
        # Enterprise RAG System
        
        A production-ready Retrieval-Augmented Generation (RAG) system built with:
        - **FastAPI** for backend services
        - **LangChain** for LLM orchestration
        - **ChromaDB** for vector storage
        - **Groq** for fast inference
        - **Streamlit** for the frontend
        
        Version 1.0.0
        """
    }
)


def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("💬 Enterprise RAG Chat")
    
    # Show welcome message if no conversation
    if not st.session_state.messages:
        render_welcome_message()
    
    # Render chat interface
    render_chat_interface()
    
    # Handle user input (must be at the end for st.chat_input to work properly)
    handle_user_input()


if __name__ == "__main__":
    main()
