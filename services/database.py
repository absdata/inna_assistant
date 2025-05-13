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

    async def save_file_chunks(
        self,
        message_id: int,
        content: str,
        chunk_size: int = 100000  # Default chunk size ~100KB of text
    ) -> List[Dict[str, Any]]:
        """Save file content in chunks."""
        logger.info(f"Saving file content in chunks for message {message_id}")
        chunks = []
        
        # Split content into chunks
        for i in range(0, len(content), chunk_size):
            chunk_data = {
                "message_id": message_id,
                "chunk_index": i // chunk_size,
                "chunk_content": content[i:i + chunk_size]
            }
            chunks.append(chunk_data)
        
        try:
            # Save all chunks
            self._log_query("INSERT", "inna_file_chunks", {"chunk_count": len(chunks)})
            result = self.client.table("inna_file_chunks").insert(chunks).execute()
            self._log_result("INSERT", result)
            
            if result.data:
                logger.info(f"Saved {len(result.data)} chunks for message {message_id}")
                return result.data
            else:
                logger.error("Failed to save file chunks: no data returned")
                return []
        except Exception as e:
            logger.error(f"Error saving file chunks: {str(e)}", exc_info=True)
            raise

    async def get_file_content(
        self,
        message_id: int
    ) -> Optional[str]:
        """Retrieve complete file content from chunks."""
        logger.info(f"Retrieving file content for message {message_id}")
        try:
            # Get all chunks for the message, ordered by chunk_index
            result = self.client.table("inna_file_chunks")\
                .select("*")\
                .eq("message_id", message_id)\
                .order("chunk_index")\
                .execute()
            
            if not result.data:
                logger.info(f"No chunks found for message {message_id}")
                return None
            
            # Combine chunks in order
            content = "".join(chunk["chunk_content"] for chunk in result.data)
            logger.info(f"Retrieved {len(result.data)} chunks for message {message_id}")
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving file chunks: {str(e)}", exc_info=True)
            raise

    async def save_embedding(
        self,
        message_id: int,
        chat_id: int,
        text: str,
        embedding: List[float],
        chunk_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Save a message embedding to the database."""
        embedding_data = {
            "message_id": message_id,
            "chat_id": chat_id,
            "text": text,
            "embedding": embedding,
            "chunk_index": chunk_index
        }
        
        self._log_query("INSERT", "inna_message_embeddings", embedding_data)
        result = self.client.table("inna_message_embeddings").insert(embedding_data).execute()
        self._log_result("INSERT", result)
        return result.data[0] if result.data else None

    async def get_embeddings_for_message(
        self,
        message_id: int
    ) -> List[Dict[str, Any]]:
        """Get all embeddings for a message, including chunks."""
        result = self.client.table("inna_message_embeddings")\
            .select("*")\
            .eq("message_id", message_id)\
            .order("chunk_index")\
            .execute()
        return result.data if result.data else []

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

    async def setup_match_messages_function(self):
        """Set up or update the match_messages database function."""
        function_sql = """
        create or replace function match_messages(
            query_embedding vector(2000),
            match_threshold float,
            match_count int
        )
        returns table (
            id bigint,
            chat_id bigint,
            text text,
            chunk_index int,
            similarity float
        )
        language sql stable
        as $$
            select
                inna_message_embeddings.id,
                inna_message_embeddings.chat_id,
                inna_message_embeddings.text,
                inna_message_embeddings.chunk_index,
                1 - (inna_message_embeddings.embedding <=> query_embedding) as similarity
            from inna_message_embeddings
            where 1 - (inna_message_embeddings.embedding <=> query_embedding) > match_threshold
            order by inna_message_embeddings.embedding <=> query_embedding
            limit match_count;
        $$;
        """
        try:
            await self.client.rpc("match_messages", {}).execute()
        except Exception:
            # Function doesn't exist or needs updating
            self.client.query(function_sql).execute()

    async def search_messages_with_content(
        self,
        chat_id: int,
        query_embedding: List[float],
        text_search: Optional[str] = None,
        threshold: float = 0.3,  # Lower threshold for better recall
        limit: int = 20  # Increased limit to find more potential matches
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
            # First, get similar messages by embedding with a higher limit for initial matching
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit * 3  # Get more initial matches to ensure we don't miss relevant chunks
            }
            self._log_query("RPC", "match_messages", rpc_params)
            
            similar_messages = self.client.rpc(
                "match_messages",
                rpc_params
            ).execute()
            
            self._log_result("Vector Search", similar_messages)
            logger.debug(f"Vector search found {len(similar_messages.data) if similar_messages.data else 0} matches")
            
            # Group results by message_id to combine chunks
            message_groups = {}
            for msg in similar_messages.data:
                message_id = msg["id"]
                if message_id not in message_groups:
                    message_groups[message_id] = {
                        "chunks": [],
                        "max_similarity": msg["similarity"],
                        "total_similarity": 0,  # Track total similarity for better ranking
                        "matching_chunk_count": 0  # Count matching chunks for better document relevance
                    }
                message_groups[message_id]["chunks"].append(msg)
                message_groups[message_id]["total_similarity"] += msg["similarity"]
                message_groups[message_id]["matching_chunk_count"] += 1
                message_groups[message_id]["max_similarity"] = max(
                    message_groups[message_id]["max_similarity"],
                    msg["similarity"]
                )
            
            # Get full message content for each group
            result_messages = []
            for message_id, group in message_groups.items():
                # Get the base message
                message = self.client.table("inna_messages")\
                    .select("*")\
                    .eq("id", message_id)\
                    .single()\
                    .execute()
                
                if message.data:
                    # Calculate average similarity for better document ranking
                    avg_similarity = group["total_similarity"] / group["matching_chunk_count"]
                    
                    # Get file content if needed
                    file_content = None
                    file_chunks = self.client.table("inna_file_chunks")\
                        .select("*")\
                        .eq("message_id", message_id)\
                        .order("chunk_index")\
                        .execute()
                    
                    if file_chunks.data:
                        file_content = "".join(chunk["chunk_content"] for chunk in file_chunks.data)
                        message.data["file_content"] = file_content
                        
                        # Boost similarity score for messages with file content
                        # This helps prioritize document content over chat messages
                        if group["matching_chunk_count"] > 1:
                            avg_similarity *= 1.2  # 20% boost for multi-chunk matches
                    
                    # Add similarity scores and chunk information
                    message.data["max_similarity"] = group["max_similarity"]
                    message.data["avg_similarity"] = avg_similarity
                    message.data["matching_chunk_count"] = group["matching_chunk_count"]
                    message.data["matching_chunks"] = sorted([
                        {
                            "chunk_index": chunk["chunk_index"],
                            "similarity": chunk["similarity"],
                            "text": chunk["text"]
                        }
                        for chunk in group["chunks"]
                    ], key=lambda x: x["similarity"], reverse=True)
                    
                    # If we have a text search, check content
                    if text_search:
                        text_lower = text_search.lower()
                        message_text = (message.data.get("text") or "").lower()
                        file_content_lower = (file_content or "").lower()
                        
                        # If text is found in content, boost the similarity score
                        if text_lower in message_text or text_lower in file_content_lower:
                            avg_similarity *= 1.3  # 30% boost for text match
                    
                    message.data["final_similarity"] = avg_similarity
                    result_messages.append(message.data)
                    
                    logger.debug(
                        f"Processed message {message_id}:\n"
                        f"Max Similarity: {group['max_similarity']:.3f}\n"
                        f"Avg Similarity: {avg_similarity:.3f}\n"
                        f"Matching Chunks: {group['matching_chunk_count']}\n"
                        f"Has File Content: {'Yes' if file_content else 'No'}"
                    )
            
            # Sort by final similarity score
            result_messages.sort(key=lambda x: x["final_similarity"], reverse=True)
            final_results = result_messages[:limit]
            
            logger.debug(
                f"Search Results Summary:\n"
                f"Total Messages Found: {len(result_messages)}\n"
                f"Returned Results: {len(final_results)}\n"
                f"Top Similarity Scores: {[f'{msg.get('final_similarity', 0):.3f}' for msg in final_results[:3]]}"
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching messages: {str(e)}", exc_info=True)
            raise

    async def save_agent_memory(
        self,
        role: str,
        chat_id: int,
        context: str,
        embedding: List[float],
        relevance_score: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Save an agent memory to the database."""
        memory_data = {
            "agent_role": role,
            "chat_id": chat_id,
            "context": context,
            "embedding": embedding,
            "relevance_score": relevance_score,
            "metadata": metadata or {}
        }
        
        self._log_query("INSERT", "inna_agent_memory", memory_data)
        
        try:
            result = self.client.table("inna_agent_memory").insert(memory_data).execute()
            self._log_result("INSERT", result)
            
            if result.data:
                logger.info(f"Memory saved successfully with ID: {result.data[0]['id']}")
                return result.data[0]
            else:
                logger.error("Failed to save memory: no data returned")
                return None
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}", exc_info=True)
            raise

    async def get_agent_memories(
        self,
        embedding: Optional[List[float]] = None,
        role: Optional[str] = None,
        chat_id: Optional[int] = None,
        threshold: float = 0.3,
        limit: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant agent memories."""
        try:
            if embedding:
                # Use vector similarity search
                result = self.client.rpc(
                    "match_agent_memories",
                    {
                        "query_embedding": embedding,
                        "agent_role": role,
                        "match_threshold": threshold,
                        "match_count": limit,
                        "start_time": start_time.isoformat() if start_time else None,
                        "end_time": end_time.isoformat() if end_time else None
                    }
                ).execute()
            else:
                # Use regular query
                query = self.client.table("inna_agent_memory").select("*")
                
                if role:
                    query = query.eq("agent_role", role)
                if chat_id:
                    query = query.eq("chat_id", chat_id)
                if start_time:
                    query = query.gte("created_at", start_time.isoformat())
                if end_time:
                    query = query.lte("created_at", end_time.isoformat())
                
                query = query.order("created_at", desc=True).limit(limit)
                result = query.execute()
            
            self._log_result("SELECT", result)
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}", exc_info=True)
            raise

    async def update_memory_access(
        self,
        memory_id: int
    ) -> None:
        """Update last_accessed and access_count for a memory."""
        try:
            self.client.table("inna_agent_memory")\
                .update({
                    "last_accessed": datetime.utcnow().isoformat(),
                    "access_count": self.client.raw("access_count + 1")
                })\
                .eq("id", memory_id)\
                .execute()
        except Exception as e:
            logger.error(f"Error updating memory access: {str(e)}", exc_info=True)
            # Don't raise the error as this is not critical

db_service = DatabaseService() 