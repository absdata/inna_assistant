from supabase import create_client, Client
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from config.config import config
import logging

# Create logger for this module
logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        logger.info("Initializing database service...")
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
        # Set logging level to DEBUG to see all logs
        logger.setLevel(logging.DEBUG)
        logger.info("Database service initialized successfully")
    
    def _log_query(self, operation: str, table: str, params: Dict[str, Any]) -> None:
        """Log database query details."""
        logger.debug(
            f"\n{'='*80}\n"
            f"DB Operation: {operation}\n"
            f"Table: {table}\n"
            f"Parameters: {params}\n"
            f"{'='*80}"
        )
    
    def _log_result(self, operation: str, result: Any) -> None:
        """Log database query results."""
        logger.debug(
            f"\n{'-'*80}\n"
            f"Operation Result:\n"
            f"Status: {'Success' if result and (result.data or result.count is not None) else 'No Data'}\n"
            f"Data: {result.data if result and hasattr(result, 'data') else result}\n"
            f"{'-'*80}"
        )

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
        
        self._log_query("INSERT", "inna_messages", message_data)
        
        try:
            result = self.client.table("inna_messages").insert(message_data).execute()
            self._log_result("INSERT", result)
            
            if result.data:
                logger.info(f"Message saved successfully with ID: {result.data[0]['id']}")
                return result.data[0]
            else:
                logger.error("Failed to save message: no data returned")
                return None
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}", exc_info=True)
            raise

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
        logger.debug(f"Finding similar messages for chat {chat_id} with threshold {threshold}")
        try:
            result = self.client.rpc(
                "match_messages",
                {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "match_count": limit
                }
            ).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} similar messages")
                logger.debug(f"Similarity scores: {[msg['similarity'] for msg in result.data]}")
            else:
                logger.info("No similar messages found")
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error finding similar messages: {str(e)}", exc_info=True)
            raise

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
        logger.debug("Fetching active chats...")
        try:
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
            
            logger.info(f"Found {len(chats)} active chats")
            logger.debug(f"Active chat IDs: {[chat['chat_id'] for chat in chats]}")
            return chats
            
        except Exception as e:
            logger.error(f"Error fetching active chats: {str(e)}", exc_info=True)
            raise

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

    async def search_messages_with_content(
        self,
        chat_id: int,
        query_embedding: List[float],
        text_search: Optional[str] = None,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search messages using both vector similarity and text content."""
        logger.debug(
            f"\n{'='*80}\n"
            f"Starting Combined Search:\n"
            f"Chat ID: {chat_id}\n"
            f"Text Search: {text_search}\n"
            f"Threshold: {threshold}\n"
            f"Limit: {limit}\n"
            f"{'='*80}"
        )
        
        try:
            # First, get similar messages by embedding
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit * 2
            }
            self._log_query("RPC", "match_messages", rpc_params)
            
            similar_messages = self.client.rpc(
                "match_messages",
                rpc_params
            ).execute()
            
            self._log_result("Vector Search", similar_messages)
            logger.debug(f"Vector search found {len(similar_messages.data) if similar_messages.data else 0} matches")
            
            message_ids = [msg["id"] for msg in similar_messages.data] if similar_messages.data else []
            
            # If we have a text search, use it to filter messages
            if text_search and message_ids:
                logger.debug(f"Performing text search filtering for {len(message_ids)} messages")
                query_params = {
                    "chat_id": chat_id,
                    "message_ids": message_ids
                }
                self._log_query("SELECT", "inna_messages", query_params)
                
                # Build the query to search in both text and file_content
                query = self.client.table("inna_messages")\
                    .select("*")\
                    .eq("chat_id", chat_id)\
                    .in_("id", message_ids)
                
                result = query.execute()
                self._log_result("Text Search", result)
                
                # Filter messages that contain the search text
                text_lower = text_search.lower()
                filtered_messages = []
                for msg in result.data:
                    text_content = (msg.get("text") or "").lower()
                    file_content = (msg.get("file_content") or "").lower()
                    
                    if text_lower in text_content or text_lower in file_content:
                        similarity = next(
                            (m["similarity"] for m in similar_messages.data if m["id"] == msg["id"]),
                            0.0
                        )
                        msg["similarity"] = similarity
                        filtered_messages.append(msg)
                        logger.debug(
                            f"Match found:\n"
                            f"ID: {msg['id']}\n"
                            f"Similarity: {similarity}\n"
                            f"Has file content: {'Yes' if msg.get('file_content') else 'No'}"
                        )
                
                filtered_messages.sort(key=lambda x: x["similarity"], reverse=True)
                result_messages = filtered_messages[:limit]
                logger.debug(f"Final filtered results: {len(result_messages)} messages")
                return result_messages
            
            # If no text search, just get the full messages
            if message_ids:
                logger.debug("Retrieving full message content for vector matches")
                query_params = {
                    "message_ids": message_ids
                }
                self._log_query("SELECT", "inna_messages", query_params)
                
                result = self.client.table("inna_messages")\
                    .select("*")\
                    .in_("id", message_ids)\
                    .execute()
                
                self._log_result("Full Message Retrieval", result)
                
                # Add similarity scores
                full_messages = []
                for msg in result.data:
                    similarity = next(
                        (m["similarity"] for m in similar_messages.data if m["id"] == msg["id"]),
                        0.0
                    )
                    msg["similarity"] = similarity
                    full_messages.append(msg)
                    logger.debug(
                        f"Processing message:\n"
                        f"ID: {msg['id']}\n"
                        f"Similarity: {similarity}\n"
                        f"Has file content: {'Yes' if msg.get('file_content') else 'No'}"
                    )
                
                full_messages.sort(key=lambda x: x["similarity"], reverse=True)
                result_messages = full_messages[:limit]
                logger.debug(f"Final results: {len(result_messages)} messages")
                return result_messages
            
            logger.debug("No matching messages found")
            return []
            
        except Exception as e:
            logger.error(f"Error searching messages: {str(e)}", exc_info=True)
            raise

db_service = DatabaseService() 