"""
Chat interface component for the Streamlit application.
"""

import streamlit as st
from typing import Optional
import uuid
from frontend.utils import (
    get_api_client,
    format_sources,
    format_metrics,
    initialize_session_state
)


def render_chat_interface():
    """Render the main chat interface."""
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and st.session_state.show_sources:
                sources = message.get("sources", [])
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        st.markdown(format_sources(sources))
            
            # Show metrics if enabled
            if message["role"] == "assistant" and st.session_state.show_metrics:
                metrics = message.get("metrics")
                if metrics:
                    st.caption(metrics)


def handle_user_input():
    """Handle user input and generate response."""
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            api_client = get_api_client()
            
            try:
                # Get or create conversation ID
                if not st.session_state.conversation_id:
                    st.session_state.conversation_id = str(uuid.uuid4())
                
                # Check if agent mode is enabled
                use_agents = st.session_state.get("use_agents", False)
                force_agents = st.session_state.get("force_agents", False)
                
                # Show agent mode indicator
                if use_agents:
                    mode_indicator = "🤖 Agent Mode" if not force_agents else "🤖 Forced Agent Mode"
                    st.caption(f"{mode_indicator} | This may take longer but provides deeper analysis")
                
                # Stream or non-stream response (agents disable streaming)
                if st.session_state.stream_responses and not use_agents:
                    # Streaming response (not supported with agents)
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in api_client.chat_stream(
                        prompt,
                        st.session_state.conversation_id,
                        use_agents=False,
                        force_agents=False
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Store response (sources not available in streaming mode)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": [],
                        "metrics": None
                    })
                    
                else:
                    # Non-streaming response (or agent mode)
                    spinner_text = "🤖 Agents analyzing..." if use_agents else "Thinking..."
                    
                    with st.spinner(spinner_text):
                        response = api_client.chat(
                            prompt,
                            st.session_state.conversation_id,
                            stream=False,
                            use_agents=use_agents,
                            force_agents=force_agents
                        )
                    
                    answer = response.get("answer", "Sorry, I couldn't generate a response.")
                    sources = response.get("sources", [])
                    metadata = response.get("metadata", {})
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show agent info if agents were used
                    if use_agents and metadata.get("orchestrated"):
                        agent_info = metadata.get("agents", [])
                        intent_info = metadata.get("intent", {})
                        
                        with st.expander("🤖 Agent Execution Details", expanded=False):
                            st.write(f"**Query Complexity:** {intent_info.get('complexity', 'unknown').title()}")
                            st.write(f"**Primary Intent:** {intent_info.get('primary', 'unknown').title()}")
                            st.write(f"**Confidence:** {intent_info.get('confidence', 0):.2%}")
                            
                            st.write(f"\\n**Agents Executed ({len(agent_info)}):**")
                            for agent in agent_info:
                                status_emoji = "✅" if agent.get("status") == "completed" else "❌"
                                st.write(
                                    f"{status_emoji} **{agent.get('agent')}** "
                                    f"({agent.get('type')}) - {agent.get('time', 0):.2f}s"
                                )
                    
                    # Display sources
                    if sources and st.session_state.show_sources:
                        with st.expander("📚 Sources", expanded=False):
                            st.markdown(format_sources(sources))
                    
                    # Display metrics
                    metrics_str = None
                    if st.session_state.show_metrics:
                        metrics_str = format_metrics(response)
                        # Add agent mode indicator to metrics
                        if metadata.get("mode") == "agent":
                            metrics_str = f"🤖 Agent Mode | {metrics_str}"
                        st.caption(metrics_str)
                    
                    # Store response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metrics": metrics_str
                    })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "metrics": None
                })


def render_welcome_message():
    """Render welcome message when chat is empty."""
    
    if not st.session_state.messages:
        st.markdown("""
        # 🤖 Welcome to Enterprise RAG System
        
        ### Get started:
        
        1. **📤 Upload documents** using the sidebar
        2. **💬 Ask questions** about your documents
        3. **📚 View sources** for each answer
        
        ### Example questions:
        
        - "What are the main topics in the documents?"
        - "Summarize the key points"
        - "What is mentioned about [topic]?"
        
        ---
        
        **Tip:** Enable "Stream responses" in settings for real-time answers!
        """)
