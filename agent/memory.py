# Vector memory retrieval
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from services.azure_openai import openai_service
from services.database import db_service
import logging
from enum import Enum

# Create logger for this module
logger = logging.getLogger(__name__)

class MemoryType(str, Enum):
    CHAT = "chat"
    DOCUMENT = "document"
    SUMMARY = "summary"
    TASK = "task"
    CONTEXT = "context"
    PLANNER = "planner"
    CRITIC = "critic"
    RESPONDER = "responder"

    @classmethod
    def from_role(cls, role: str) -> "MemoryType":
        """Convert an agent role to a memory type."""
        try:
            return cls(role)
        except ValueError:
            # If direct conversion fails, try to map common agent roles
            role_map = {
                "context": cls.CONTEXT,
                "planner": cls.PLANNER,
                "critic": cls.CRITIC,
                "responder": cls.RESPONDER,
                # Add more mappings if needed
            }
            if role in role_map:
                return role_map[role]
            # Default to CHAT type if no mapping exists
            logger.warning(f"Unknown role type '{role}', defaulting to CHAT")
            return cls.CHAT

class Memory(BaseModel):
    """Base class for memory entries."""
    id: Optional[int] = None
    chat_id: int
    content: str
    memory_type: MemoryType
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0

class MemoryStore:
    """Long-term memory store for the agent."""
    
    def __init__(self):
        self.recent_memories: Dict[int, List[Memory]] = {}  # Cache for recent memories by chat_id
        
    async def add_memory(
        self,
        chat_id: int,
        content: str,
        memory_type: MemoryType,
        metadata: Dict[str, Any] = None,
        relevance_score: float = 1.0
    ) -> Memory:
        """Add a new memory to the store."""
        try:
            # Generate embedding for the content
            embedding = await openai_service.get_embedding(content)
            
            # Create memory object
            memory = Memory(
                chat_id=chat_id,
                content=content,
                memory_type=memory_type,
                embedding=embedding,
                metadata=metadata or {},
                relevance_score=relevance_score
            )
            
            # Save to database
            saved_memory = await db_service.save_agent_memory(
                role=memory_type.value,
                chat_id=chat_id,
                context=content,
                embedding=embedding,
                relevance_score=relevance_score,
                metadata=metadata
            )
            
            if saved_memory:
                memory.id = saved_memory["id"]
                # Update cache
                if chat_id not in self.recent_memories:
                    self.recent_memories[chat_id] = []
                self.recent_memories[chat_id].append(memory)
                
                logger.info(f"Added new {memory_type.value} memory for chat {chat_id}")
                return memory
            
            logger.error("Failed to save memory to database")
            return None
            
        except Exception as e:
            logger.error(f"Error adding memory: {str(e)}", exc_info=True)
            raise

    async def get_relevant_memories(
        self,
        chat_id: int,
        query: str,
        memory_type: Optional[MemoryType] = None,
        threshold: float = 0.3,
        limit: int = 10
    ) -> List[Memory]:
        """Retrieve relevant memories based on query."""
        try:
            # Generate embedding for the query
            query_embedding = await openai_service.get_embedding(query)
            
            # Get memories from database
            memories = await db_service.get_agent_memories(
                embedding=query_embedding,
                role=memory_type.value if memory_type else None,
                chat_id=chat_id,
                threshold=threshold,
                limit=limit
            )
            
            # Convert to Memory objects
            memory_objects = []
            for mem in memories:
                try:
                    # Log the memory data for debugging
                    logger.debug(f"Processing memory: {mem}")
                    
                    # Extract and validate required fields
                    memory_id = mem.get("id")
                    if not memory_id:
                        logger.error(f"Memory missing ID: {mem}")
                        continue
                        
                    memory_role = mem.get("memory_role")
                    if not memory_role:
                        logger.error(f"Memory missing role: {mem}")
                        continue
                    
                    # Create memory object with validated data
                    memory = Memory(
                        id=memory_id,
                        chat_id=chat_id,
                        content=mem["context"],
                        memory_type=MemoryType.from_role(memory_role),
                        embedding=mem["embedding"],
                        metadata=mem.get("metadata", {}),
                        relevance_score=mem.get("relevance_score", 1.0),
                        created_at=mem.get("created_at", datetime.utcnow()),
                        similarity=mem.get("similarity", 0.0)
                    )
                    memory_objects.append(memory)
                except Exception as e:
                    logger.error(f"Error creating memory object: {str(e)}, Memory data: {mem}", exc_info=True)
                    continue
            
            logger.info(f"Retrieved {len(memory_objects)} relevant memories for chat {chat_id}")
            return memory_objects
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}", exc_info=True)
            return []

    async def add_document_memory(
        self,
        chat_id: int,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: int = 1000
    ) -> List[Memory]:
        """Add document content as chunked memories."""
        try:
            memories = []
            # Split content into chunks
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                memory = await self.add_memory(
                    chat_id=chat_id,
                    content=chunk,
                    memory_type=MemoryType.DOCUMENT,
                    metadata=chunk_metadata,
                    relevance_score=1.0
                )
                
                if memory:
                    memories.append(memory)
            
            logger.info(f"Added document with {len(memories)} chunks for chat {chat_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error adding document memory: {str(e)}", exc_info=True)
            return []

    async def add_chat_memory(
        self,
        chat_id: int,
        message: Dict[str, Any]
    ) -> Optional[Memory]:
        """Add a chat message to memory."""
        try:
            metadata = {
                "message_id": message.get("message_id"),
                "user_id": message.get("user_id"),
                "username": message.get("username"),
                "is_bot": message.get("is_bot", False)
            }
            
            content = message.get("text", "")
            if message.get("file_content"):
                # If message has file content, add it as a document memory instead
                await self.add_document_memory(
                    chat_id=chat_id,
                    content=message["file_content"],
                    metadata={
                        **metadata,
                        "file_url": message.get("file_url"),
                        "file_name": message.get("file_name")
                    }
                )
                
            # Add the message text as chat memory
            if content:
                memory = await self.add_memory(
                    chat_id=chat_id,
                    content=content,
                    memory_type=MemoryType.CHAT,
                    metadata=metadata
                )
                return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Error adding chat memory: {str(e)}", exc_info=True)
            return None

    async def get_chat_summary(
        self,
        chat_id: int,
        time_window: timedelta = timedelta(days=7)
    ) -> Optional[str]:
        """Get or generate a summary of recent chat history."""
        try:
            # First, try to get an existing recent summary
            summaries = await db_service.get_agent_memories(
                role=MemoryType.SUMMARY.value,
                chat_id=chat_id,
                limit=1
            )
            
            if summaries:
                latest_summary = summaries[0]
                summary_time = latest_summary["created_at"]
                if datetime.utcnow() - summary_time < timedelta(hours=24):
                    return latest_summary["context"]
            
            # If no recent summary, generate a new one
            start_time = datetime.utcnow() - time_window
            memories = await db_service.get_agent_memories(
                role=MemoryType.CHAT.value,
                chat_id=chat_id,
                start_time=start_time
            )
            
            if not memories:
                return None
            
            # Prepare messages for summary generation
            messages = [
                openai_service.create_system_message(
                    "You are a summarization agent. Create a concise summary of the chat history "
                    "that captures key points, decisions, and important information."
                ),
                openai_service.create_user_message(
                    "Chat History:\n" + "\n\n".join([
                        f"{mem['created_at'].strftime('%Y-%m-%d %H:%M:%S')}: {mem['context']}"
                        for mem in memories
                    ]) + "\n\nCreate a comprehensive summary:"
                )
            ]
            
            summary = await openai_service.get_completion(messages)
            
            # Save the summary as a new memory
            await self.add_memory(
                chat_id=chat_id,
                content=summary,
                memory_type=MemoryType.SUMMARY,
                metadata={
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "message_count": len(memories)
                }
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating chat summary: {str(e)}", exc_info=True)
            return None

# Create global memory store instance
memory_store = MemoryStore()