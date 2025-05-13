from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from services.azure_openai import openai_service
from services.database import db_service
import logging

# Create logger for this module
logger = logging.getLogger(__name__)

class AgentMemory(BaseModel):
    """Base class for agent memory."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    current_message: Dict[str, Any] = Field(default_factory=dict)
    chat_id: int = Field(default=0)
    role: str = Field(default="")

class BaseAgent:
    """Base class for all agent roles."""
    def __init__(self, role: str):
        self.role = role
    
    async def get_context(self, query: str, chat_id: int) -> List[Dict[str, Any]]:
        """Get relevant context from agent memory."""
        try:
            # Get embedding from OpenAI (already compressed to 2000D)
            embedding = await openai_service.get_embedding(query)
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            
            # Get memories using the compressed embedding
            memories = await db_service.get_agent_memories(
                embedding=embedding,
                role=self.role,
                chat_id=chat_id
            )
            return memories
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}", exc_info=True)
            return []
    
    async def save_memory(
        self,
        chat_id: int,
        context: str,
        relevance_score: float = 1.0
    ) -> Dict[str, Any]:
        """Save new context to agent memory."""
        try:
            # Get embedding from OpenAI (already compressed to 2000D)
            embedding = await openai_service.get_embedding(context)
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            
            # Save memory with the compressed embedding
            memory = await db_service.save_agent_memory(
                role=self.role,
                chat_id=chat_id,
                context=context,
                embedding=embedding,
                relevance_score=relevance_score
            )
            return memory
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}", exc_info=True)
            return None
    
    async def process(self, memory: AgentMemory) -> str:
        """Process the current state and return a response."""
        raise NotImplementedError("Subclasses must implement process()") 