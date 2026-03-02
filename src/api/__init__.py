"""
API module for the RAG system.
"""

__version__ = "1.0.0"

from . import documents, search, chat

__all__ = [
    "documents",
    "search",
    "chat",
]
