from supabase import create_client, Client
from typing import List, Dict, Optional, Any
import numpy as np
from config.config import config

class DatabaseService:
    def __init__(self):
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
    
    async def save_message(
        self,
        chat_id: int,
        message_id: int,
        user_id: int,
        username: str,
        text: Optional[str] = None,
        file_url: Optional[str] = None,
        file_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save a message to the database."""
        message_data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "user_id": user_id,
            "username": username,
            "text": text,
            "file_url": file_url,
            "file_content": file_content
        }
        
        result = self.client.table("messages").insert(message_data).execute()
        return result.data[0] if result.data else None

    async def save_embedding(
        self,
        message_id: int,
        chat_id: int,
        text: str,
        embedding: List[float]
    ) -> Dict[str, Any]:
        """Save a message embedding to the database."""
        embedding_data = {
            "message_id": message_id,
            "chat_id": chat_id,
            "text": text,
            "embedding": embedding
        }
        
        result = self.client.table("message_embeddings").insert(embedding_data).execute()
        return result.data[0] if result.data else None

    async def get_similar_messages(
        self,
        embedding: List[float],
        chat_id: int,
        threshold: float = 0.7,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar messages based on embedding similarity."""
        result = self.client.rpc(
            "match_messages",
            {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": limit
            }
        ).execute()
        
        return result.data if result.data else []

    async def get_chat_history(
        self,
        chat_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent chat history."""
        result = self.client.table("messages")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data if result.data else []

db_service = DatabaseService() 