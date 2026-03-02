"""
Conversation Module
Manages conversation history and context for chat interactions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """A conversation with history."""
    id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {})
        )


class ConversationManager:
    """
    Manages conversation history and context.
    
    Features:
    - Message history management
    - Token-aware context windowing
    - Conversation persistence
    - Multi-conversation support
    - Summary generation for long conversations
    """
    
    def __init__(
        self,
        max_history_tokens: int = 4000,
        max_messages: int = 50,
        persist_dir: Optional[str] = None
    ):
        """
        Initialize conversation manager.
        
        Args:
            max_history_tokens: Maximum tokens in history
            max_messages: Maximum number of messages to keep
            persist_dir: Directory to save conversations
        """
        self.max_history_tokens = max_history_tokens
        self.max_messages = max_messages
        self.persist_dir = persist_dir or str(Path("data") / "conversations")
        
        # Create persist directory
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Active conversations
        self.conversations: Dict[str, Conversation] = {}
        
        logger.info(
            f"Conversation manager initialized: "
            f"max_tokens={max_history_tokens}, max_messages={max_messages}"
        )
    
    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional conversation ID
            system_prompt: Optional system prompt
            
        Returns:
            New Conversation object
        """
        # Generate ID if not provided
        if conversation_id is None:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conversation = Conversation(id=conversation_id)
        
        # Add system prompt if provided
        if system_prompt:
            conversation.messages.append(
                Message(role="system", content=system_prompt)
            )
        
        self.conversations[conversation_id] = conversation
        
        logger.info(f"Created conversation: {conversation_id}")
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None
        """
        # Check active conversations
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        # Try loading from disk
        return self.load_conversation(conversation_id)
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created Message
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        
        # Trim if needed
        self._trim_conversation(conversation)
        
        logger.debug(f"Added {role} message to {conversation_id}")
        return message
    
    def get_messages(
        self,
        conversation_id: str,
        role: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages from conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Optional role filter
            limit: Optional message limit
            
        Returns:
            List of messages
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages
        
        # Filter by role
        if role:
            messages = [m for m in messages if m.role == role]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_history_for_llm(
        self,
        conversation_id: str,
        include_system: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM.
        
        Args:
            conversation_id: Conversation ID
            include_system: Whether to include system messages
            
        Returns:
            List of message dicts
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = []
        for msg in conversation.messages:
            if not include_system and msg.role == "system":
                continue
            
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    def _trim_conversation(self, conversation: Conversation) -> None:
        """
        Trim conversation to stay within limits.
        
        Args:
            conversation: Conversation to trim
        """
        # Keep system message if present
        system_messages = [m for m in conversation.messages if m.role == "system"]
        other_messages = [m for m in conversation.messages if m.role != "system"]
        
        # Trim by message count
        if len(other_messages) > self.max_messages:
            other_messages = other_messages[-self.max_messages:]
            logger.debug(f"Trimmed conversation to {self.max_messages} messages")
        
        # Trim by token count (approximate)
        total_tokens = sum(len(m.content) // 4 for m in other_messages)
        
        while total_tokens > self.max_history_tokens and len(other_messages) > 1:
            # Remove oldest non-system message
            removed = other_messages.pop(0)
            total_tokens -= len(removed.content) // 4
            logger.debug("Removed message to stay within token limit")
        
        # Reconstruct message list
        conversation.messages = system_messages + other_messages
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear all messages from conversation except system.
        
        Args:
            conversation_id: Conversation ID
        """
        conversation = self.get_conversation(conversation_id)
        if conversation:
            system_messages = [m for m in conversation.messages if m.role == "system"]
            conversation.messages = system_messages
            conversation.updated_at = datetime.now()
            logger.info(f"Cleared conversation: {conversation_id}")
    
    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation.
        
        Args:
            conversation_id: Conversation ID
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        
        # Delete from disk
        filepath = Path(self.persist_dir) / f"{conversation_id}.json"
        if filepath.exists():
            filepath.unlink()
        
        logger.info(f"Deleted conversation: {conversation_id}")
    
    def save_conversation(self, conversation_id: str) -> None:
        """
        Save conversation to disk.
        
        Args:
            conversation_id: Conversation ID
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            logger.warning(f"Cannot save non-existent conversation: {conversation_id}")
            return
        
        filepath = Path(self.persist_dir) / f"{conversation_id}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, indent=2)
            
            logger.debug(f"Saved conversation: {conversation_id}")
        
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Load conversation from disk.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None
        """
        filepath = Path(self.persist_dir) / f"{conversation_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversation = Conversation.from_dict(data)
            self.conversations[conversation_id] = conversation
            
            logger.debug(f"Loaded conversation: {conversation_id}")
            return conversation
        
        except Exception as e:
            logger.error(f"Failed to load conversation: {str(e)}")
            return None
    
    def list_conversations(self) -> List[str]:
        """
        List all conversation IDs.
        
        Returns:
            List of conversation IDs
        """
        # Active conversations
        active = set(self.conversations.keys())
        
        # Saved conversations
        saved = set()
        persist_path = Path(self.persist_dir)
        if persist_path.exists():
            for file in persist_path.glob("*.json"):
                saved.add(file.stem)
        
        return list(active | saved)
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get summary of conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Summary dictionary
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return {}
        
        messages_by_role = {}
        for msg in conversation.messages:
            messages_by_role[msg.role] = messages_by_role.get(msg.role, 0) + 1
        
        return {
            "id": conversation.id,
            "message_count": len(conversation.messages),
            "messages_by_role": messages_by_role,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat()
        }


# Singleton instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get or create conversation manager singleton."""
    global _conversation_manager
    
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    
    return _conversation_manager


if __name__ == "__main__":
    # Test conversation manager
    print("Testing Conversation Manager\n" + "="*60)
    
    manager = ConversationManager()
    
    # Create conversation
    conv = manager.create_conversation(
        system_prompt="You are a helpful AI assistant."
    )
    print(f"\nCreated conversation: {conv.id}")
    
    # Add messages
    manager.add_message(conv.id, "user", "What is machine learning?")
    manager.add_message(
        conv.id,
        "assistant",
        "Machine learning is a method of data analysis that automates model building."
    )
    manager.add_message(conv.id, "user", "Can you give an example?")
    
    # Get history
    print(f"\nConversation history:")
    history = manager.get_history_for_llm(conv.id)
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Summary
    print(f"\nConversation summary:")
    summary = manager.get_conversation_summary(conv.id)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save and load
    print(f"\nSaving conversation...")
    manager.save_conversation(conv.id)
    
    # Clear and reload
    manager.delete_conversation(conv.id)
    loaded = manager.load_conversation(conv.id)
    print(f"Loaded conversation with {len(loaded.messages)} messages")
    
    # Cleanup
    manager.delete_conversation(conv.id)
    
    print("\nConversation manager test completed!")
