from supabase import create_client, Client
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
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
        
        result = self.client.table("inna_messages").insert(message_data).execute()
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
        
        result = self.client.table("inna_message_embeddings").insert(embedding_data).execute()
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
        result = self.client.table("inna_messages")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return result.data if result.data else []

    async def get_chat_history_by_date_range(
        self,
        chat_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get chat history within a date range."""
        result = self.client.table("inna_messages")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .gte("created_at", start_date.isoformat())\
            .lte("created_at", end_date.isoformat())\
            .order("created_at", desc=True)\
            .execute()
        
        return result.data if result.data else []

    async def get_active_chats(self) -> List[Dict[str, Any]]:
        """Get all active chat IDs from the messages table."""
        # Get unique chat_ids from messages within the last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        result = self.client.table("inna_messages")\
            .select("chat_id")\
            .gte("created_at", thirty_days_ago.isoformat())\
            .execute()
        
        # Get unique chat_ids
        chat_ids = set()
        chats = []
        for row in result.data:
            if row["chat_id"] not in chat_ids:
                chat_ids.add(row["chat_id"])
                chats.append({"chat_id": row["chat_id"]})
        
        return chats

    async def save_summary(
        self,
        chat_id: int,
        summary_type: str,
        content: str,
        period_start: datetime,
        period_end: datetime,
        gdoc_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save a summary to the database."""
        summary_data = {
            "chat_id": chat_id,
            "summary_type": summary_type,
            "content": content,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "gdoc_url": gdoc_url
        }
        
        result = self.client.table("inna_summaries").insert(summary_data).execute()
        return result.data[0] if result.data else None

    async def get_latest_summary(
        self,
        chat_id: int,
        summary_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest summary of a specific type for a chat."""
        result = self.client.table("inna_summaries")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .eq("summary_type", summary_type)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        return result.data[0] if result.data else None

    async def get_summaries_by_date_range(
        self,
        chat_id: int,
        summary_type: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get summaries within a date range."""
        result = self.client.table("inna_summaries")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .eq("summary_type", summary_type)\
            .gte("period_start", start_date.isoformat())\
            .lte("period_end", end_date.isoformat())\
            .order("created_at", desc=True)\
            .execute()
        
        return result.data if result.data else []

    async def save_gdoc_sync(
        self,
        doc_id: str,
        doc_type: str,
        reference_id: int
    ) -> Dict[str, Any]:
        """Save a Google Doc sync record."""
        sync_data = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "reference_id": reference_id,
            "last_synced_at": datetime.utcnow().isoformat(),
            "sync_status": "completed"
        }
        
        result = self.client.table("inna_gdoc_sync").insert(sync_data).execute()
        return result.data[0] if result.data else None

    async def get_gdoc_sync(
        self,
        reference_id: int,
        doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get a Google Doc sync record."""
        result = self.client.table("inna_gdoc_sync")\
            .select("*")\
            .eq("reference_id", reference_id)\
            .eq("doc_type", doc_type)\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
        
        return result.data[0] if result.data else None

    async def create_task(
        self,
        chat_id: int,
        title: str,
        description: Optional[str] = None,
        priority: int = 1,
        due_date: Optional[datetime] = None,
        assigned_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new task."""
        task_data = {
            "chat_id": chat_id,
            "title": title,
            "description": description,
            "status": "pending",
            "priority": priority,
            "due_date": due_date.isoformat() if due_date else None,
            "assigned_to": assigned_to
        }
        
        result = self.client.table("inna_tasks").insert(task_data).execute()
        return result.data[0] if result.data else None

    async def update_task(
        self,
        task_id: int,
        **updates
    ) -> Dict[str, Any]:
        """Update a task."""
        updates["updated_at"] = datetime.utcnow().isoformat()
        result = self.client.table("inna_tasks")\
            .update(updates)\
            .eq("id", task_id)\
            .execute()
        
        return result.data[0] if result.data else None

    async def get_tasks(
        self,
        chat_id: int
    ) -> List[Dict[str, Any]]:
        """Get all tasks for a chat."""
        result = self.client.table("inna_tasks")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .order("created_at", desc=True)\
            .execute()
        
        return result.data if result.data else []

    async def get_tasks_by_date_range(
        self,
        chat_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get tasks updated within a date range."""
        result = self.client.table("inna_tasks")\
            .select("*")\
            .eq("chat_id", chat_id)\
            .gte("updated_at", start_date.isoformat())\
            .lte("updated_at", end_date.isoformat())\
            .order("updated_at", desc=True)\
            .execute()
        
        return result.data if result.data else []

    async def count_messages(self, chat_id: int) -> int:
        """Count total messages for a chat."""
        result = self.client.table("inna_messages")\
            .select("id", count="exact")\
            .eq("chat_id", chat_id)\
            .execute()
        
        return result.count if result.count is not None else 0

    async def count_summaries(self, chat_id: int) -> int:
        """Count total summaries for a chat."""
        result = self.client.table("inna_summaries")\
            .select("id", count="exact")\
            .eq("chat_id", chat_id)\
            .execute()
        
        return result.count if result.count is not None else 0

db_service = DatabaseService() 