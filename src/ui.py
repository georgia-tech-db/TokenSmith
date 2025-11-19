"""
Streamlit UI for TokenSmith Chat Interface
Provides a modern web interface for interacting with TokenSmith.
"""

import streamlit as st
import sys
import pathlib
import yaml
from typing import Dict, List, Optional

# Add parent directory to path to import src modules
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.config import QueryPlanConfig
from src.main import get_answer
from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever
from src.ranking.ranker import EnsembleRanker
from src.instrumentation.logging import init_logger, get_logger
from src.planning.heuristics import HeuristicQueryPlanner


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "artifacts" not in st.session_state:
        st.session_state.artifacts = None
    if "cfg" not in st.session_state:
        st.session_state.cfg = None
    if "args" not in st.session_state:
        st.session_state.args = None
    if "logger" not in st.session_state:
        st.session_state.logger = None
    if "planner" not in st.session_state:
        st.session_state.planner = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # Dict of {chat_id: [messages]}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = "default"
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def load_config_and_artifacts():
    """Load configuration and artifacts once."""
    if st.session_state.initialized:
        return True
    
    try:
        # Load config
        config_path = pathlib.Path("config/config.yaml")
        if not config_path.exists():
            st.error("Config file not found at config/config.yaml")
            return False
        
        cfg = QueryPlanConfig.from_yaml(config_path)
        st.session_state.cfg = cfg
        
        # Initialize logger
        init_logger(cfg)
        logger = get_logger()
        st.session_state.logger = logger
        
        # Create args namespace
        class Args:
            def __init__(self):
                self.index_prefix = "textbook_index"
                self.model_path = None
                self.system_prompt_mode = "tutor"
        
        args = Args()
        st.session_state.args = args
        
        # Load artifacts
        with st.spinner("Loading index and models..."):
            artifacts_dir = cfg.make_artifacts_directory()
            faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
                artifacts_dir=artifacts_dir,
                index_prefix=args.index_prefix
            )
            
            retrievers = [
                FAISSRetriever(faiss_index, cfg.embed_model),
                BM25Retriever(bm25_index)
            ]
            ranker = EnsembleRanker(
                ensemble_method=cfg.ensemble_method,
                weights=cfg.ranker_weights,
                rrf_k=int(cfg.rrf_k)
            )
            
            artifacts = {
                "chunks": chunks,
                "sources": sources,
                "retrievers": retrievers,
                "ranker": ranker,
                "metadata": metadata
            }
            st.session_state.artifacts = artifacts
            
            # Store metadata status for display
            if metadata and len(metadata) > 0:
                st.session_state.metadata_loaded = True
                st.session_state.metadata_count = len(metadata)
            else:
                st.session_state.metadata_loaded = False
        
        # Initialize query planner if enabled
        raw_config = yaml.safe_load(open(config_path))
        use_query_planner = raw_config.get("use_query_planner", True)
        if use_query_planner:
            planner = HeuristicQueryPlanner(cfg)
            st.session_state.planner = planner
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.exception(e)
        return False


def format_citations(chunk_metadata: List[Dict]) -> str:
    """Format citations as a markdown string."""
    if not chunk_metadata:
        return ""
    
    citations = []
    for i, meta in enumerate(chunk_metadata, 1):
        citation_parts = []
        if meta.get('page_number'):
            citation_parts.append(f"Page {meta['page_number']}")
        if meta.get('chapter', 0) > 0:
            citation_parts.append(f"Chapter {meta['chapter']}")
        section_hierarchy = meta.get('section_hierarchy', {})
        if section_hierarchy.get('section', 0) > 0:
            section_str = f"{section_hierarchy['section']}"
            if section_hierarchy.get('subsection', 0) > 0:
                section_str += f".{section_hierarchy['subsection']}"
            citation_parts.append(f"Section {section_str}")
        
        if citation_parts:
            citation = f"**[{i}]** {', '.join(citation_parts)}"
            section = meta.get('section', '')
            if section:
                section_display = section[:80] + "..." if len(section) > 80 else section
                citation += f"\n  *{section_display}*"
            citations.append(citation)
    
    return "\n\n".join(citations) if citations else ""


def create_new_chat():
    """Create a new chat session."""
    import uuid
    new_chat_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[new_chat_id] = []
    st.session_state.current_chat_id = new_chat_id
    st.rerun()


def main():
    st.set_page_config(
        page_title="TokenSmith - Database Textbook Q&A",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("TokenSmith")
        st.markdown("**Database Textbook Q&A System**")
        st.markdown("---")
        
        # New Chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            create_new_chat()
        
        # Chat history selector
        st.subheader("Chat History")
        chat_ids = list(st.session_state.conversations.keys())
        if chat_ids:
            # Show chat previews
            for chat_id in reversed(chat_ids):  # Most recent first
                chat_messages = st.session_state.conversations[chat_id]
                if chat_messages:
                    first_msg = chat_messages[0]["content"][:50] + "..." if len(chat_messages[0]["content"]) > 50 else chat_messages[0]["content"]
                    is_active = chat_id == st.session_state.current_chat_id
                    label = f"{'▶ ' if is_active else ''}{chat_id}: {first_msg}"
                    if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                else:
                    if st.button(f"{chat_id} (empty)", key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
        else:
            st.caption("No chat history yet. Start a conversation!")
        
        st.markdown("---")
        
        # Status
        if st.session_state.get("initialized", False):
            st.markdown("---")
            st.caption("**Status**")
            if st.session_state.get("metadata_loaded", False):
                st.success(f"Index loaded ({st.session_state.get('metadata_count', 0)} chunks)")
            else:
                st.warning("Citations unavailable")
        
        st.markdown("---")
        
        # Settings
        with st.expander("Settings"):
            system_prompt_mode = st.selectbox(
                "Answer Style:",
                ["tutor", "concise", "detailed", "baseline"],
                index=0
            )
            if st.session_state.args:
                st.session_state.args.system_prompt_mode = system_prompt_mode
            
            show_citations = st.checkbox("Show Citations", value=True)
            st.session_state.show_citations = show_citations
    
    # Main content area
    st.title("Chat with TokenSmith")
    st.caption("Ask questions about database systems from the Silberschatz textbook.")
    
    # Initialize if not done
    if not load_config_and_artifacts():
        st.error("Failed to initialize. Please check your configuration and ensure the index has been built.")
        st.stop()
    
    # Get current conversation
    if st.session_state.current_chat_id not in st.session_state.conversations:
        st.session_state.conversations[st.session_state.current_chat_id] = []
    
    messages = st.session_state.conversations[st.session_state.current_chat_id]
    
    # Display chat history
    if messages:
        for msg in messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    if "citations" in msg and msg["citations"] and st.session_state.get("show_citations", True):
                            with st.expander("References", expanded=False):
                            st.markdown(msg["citations"])
    else:
        # Welcome message for empty chat
        with st.chat_message("assistant"):
            st.markdown("👋 **Welcome to TokenSmith!**")
            st.markdown("""
            I can help you understand database systems concepts from the Silberschatz textbook.
            
            **Try asking:**
            - "What is DDL?"
            - "Explain ACID properties"
            - "How does indexing work?"
            - "What are the differences between B+ trees and hash indexes?"
            """)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about database systems..."):
        # Add user message
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get conversation history for context
                    conversation_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in messages[:-1]  # Exclude current question
                    ][-10:]  # Last 5 turns (10 messages)
                    
                    # Use query planner if available
                    query_cfg = st.session_state.cfg
                    if st.session_state.planner:
                        query_cfg = st.session_state.planner.plan(prompt)
                        # Update ranker if weights changed
                        if query_cfg.ranker_weights != st.session_state.cfg.ranker_weights:
                            st.session_state.artifacts["ranker"] = EnsembleRanker(
                                ensemble_method=query_cfg.ensemble_method,
                                weights=query_cfg.ranker_weights,
                                rrf_k=int(query_cfg.rrf_k)
                            )
                    
                    # Get answer
                    result = get_answer(
                        prompt,
                        query_cfg,
                        st.session_state.args,
                        st.session_state.logger,
                        artifacts=st.session_state.artifacts,
                        conversation_history=conversation_history
                    )
                    
                    # Handle return value
                    if isinstance(result, tuple):
                        answer, chunk_metadata = result
                    else:
                        answer = result
                        chunk_metadata = []
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Format and display citations
                    citations_text = ""
                    if chunk_metadata and st.session_state.get("show_citations", True):
                        citations_text = format_citations(chunk_metadata)
                        if citations_text:
                            with st.expander("References", expanded=True):
                                st.markdown(citations_text)
                    
                    # Store in conversation
                    messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations_text
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)
                    messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
        
        # Rerun to update display
        st.rerun()


if __name__ == "__main__":
    main()

