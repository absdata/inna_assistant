from supabase import create_client, Client
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from config.config import config
import logging
from services.document_processor import document_processor

# Create logger for this module
logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        logger.info("Initializing database service...")
        self.client: Client = create_client(config.supabase_url, config.supabase_key)
        logger.info("Database service initialized successfully")
    
    def _log_query(self, operation: str, table: str, params: Dict[str, Any]) -> None:
        """Log database query details."""
        # Create a copy of params to avoid modifying the original
        filtered_params = params.copy()
        
        # If the table is related to embeddings, simplify the output
        if 'embedding' in table.lower():
            if 'embedding' in filtered_params:
                filtered_params['embedding'] = f'[{len(filtered_params["embedding"])} dimensions]'
        
        logger.debug(
            f"\n{'='*80}\n"
            f"DB Operation: {operation}\n"
            f"Table: {table}\n"
            f"Parameters: {filtered_params}\n"
            f"{'='*80}"
        )
    
    def _log_result(self, operation: str, result: Any) -> None:
        """Log database query results."""
        # Create a simplified version of the result for logging
        if hasattr(result, 'data') and result.data:
            # If dealing with embeddings, simplify the output
            if isinstance(result.data, list):
                simplified_data = []
                for item in result.data:
                    if isinstance(item, dict):
                        item_copy = item.copy()
                        if 'embedding' in item_copy:
                            item_copy['embedding'] = f'[{len(item_copy["embedding"])} dimensions]'
                        simplified_data.append(item_copy)
                    else:
                        simplified_data.append(item)
            else:
                simplified_data = result.data
        else:
            simplified_data = result
        
        logger.debug(
            f"\n{'-'*80}\n"
            f"Operation Result:\n"
            f"Status: {'Success' if result and (result.data or result.count is not None) else 'No Data'}\n"
            f"Data: {simplified_data}\n"
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
            "file_url": file_url
        }
        
        self._log_query("INSERT", "inna_messages", message_data)
        
        try:
            # Save the message
            result = self.client.table("inna_messages").insert(message_data).execute()
            self._log_result("INSERT", result)
            
            if not result.data:
                logger.error("Failed to save message: no data returned")
                return None
            
            saved_message = result.data[0]
            logger.info(f"Message saved successfully with ID: {saved_message['id']}")
            
            # Process text content if available
            if text:
                from services.azure_openai import openai_service
                embedding = await openai_service.get_embedding(text)
                await self.save_embedding(
                    message_id=saved_message['id'],
                    chat_id=chat_id,
                    text=text,
                    embedding=embedding
                )
            
            # Process file content if available
            if file_content:
                # Save file content in chunks
                chunks = await document_processor.process_document(
                    content=file_content,
                    source_id=str(saved_message['id']),
                    source_type="file",
                    title=file_url
                )
                
                # Prepare all file chunks first
                file_chunks = []
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk['content']
                    file_chunks.append({
                        "message_id": saved_message['id'],
                        "chunk_index": i,
                        "chunk_content": chunk_text
                    })
                
                # Save all file chunks in one operation
                if file_chunks:
                    await self.save_file_chunks(
                        message_id=saved_message['id'],
                        chunks=file_chunks
                    )
                
                # Save embeddings for each chunk
                from services.azure_openai import openai_service
                for chunk in chunks:
                    chunk_text = chunk['content']
                    chunk_metadata = chunk['metadata']
                    
                    # Get embedding for the chunk
                    embedding = await openai_service.get_embedding(chunk_text)
                    
                    # Save chunk embedding
                    await self.save_embedding(
                        message_id=saved_message['id'],
                        chat_id=chat_id,
                        text=chunk_text,
                        embedding=embedding,
                        chunk_index=chunk_metadata.get('chunk_index'),
                        section_title=chunk_metadata.get('section_title')
                    )
            
            return saved_message
            
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}", exc_info=True)
            raise

    async def save_file_chunks(
        self,
        message_id: int,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Save file content chunks."""
        logger.info(f"Saving {len(chunks)} file chunks for message {message_id}")
        
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
        chunk_index: Optional[int] = None,
        section_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save a message embedding to the database."""
        try:
            embedding_data = {
                "message_id": message_id,
                "chat_id": chat_id,
                "text": text,
                "embedding": embedding,
                "chunk_index": chunk_index,
                "section_title": section_title
            }
            
            self._log_query("INSERT", "inna_message_embeddings", embedding_data)
            result = self.client.table("inna_message_embeddings").insert(embedding_data).execute()
            self._log_result("INSERT", result)
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error saving embedding: {str(e)}", exc_info=True)
            raise

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

    async def get_agent_memories(
        self,
        embedding: List[float],
        role: Optional[str] = None,
        chat_id: Optional[int] = None,
        threshold: float = 0.3,
        limit: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant agent memories based on embedding similarity."""
        logger.debug(f"Retrieving memories for role: {role}, chat_id: {chat_id}")
        try:
            # Build parameters dictionary
            params = {
                "query_embedding": embedding,
                "agent_role": role,
                "match_threshold": threshold,
                "match_count": limit
            }
            
            # Only add time parameters if they are provided
            if start_time is not None:
                params["start_time"] = start_time.isoformat()
            if end_time is not None:
                params["end_time"] = end_time.isoformat()
            
            logger.debug(f"Calling match_agent_memories with params: {params}")
            
            # Call the match_agent_memories function
            result = self.client.rpc(
                "match_agent_memories",
                params
            ).execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} relevant memories")
                logger.debug(f"Memory similarity scores: {[mem['similarity'] for mem in result.data]}")
            else:
                logger.info("No relevant memories found")
            
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

    async def search_messages_with_content(
        self,
        chat_id: int,
        query_embedding: List[float],
        text_search: Optional[str] = None,
        section_title: Optional[str] = None,
        threshold: float = 0.3,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search messages using both vector similarity and text content."""
        logger.info(f"Searching messages for chat {chat_id}")
        
        try:
            # Build RPC parameters
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit * 3,
                "section_filter": section_title
            }
            
            self._log_query("RPC", "match_messages", rpc_params)
            similar_messages = self.client.rpc("match_messages", rpc_params).execute()
            self._log_result("Vector Search", similar_messages)
            
            # Group results by message_id to combine chunks
            message_groups = {}
            for msg in similar_messages.data:
                message_id = msg["id"]
                if message_id not in message_groups:
                    message_groups[message_id] = {
                        "chunks": [],
                        "max_similarity": msg["similarity"],
                        "total_similarity": 0,
                        "matching_chunk_count": 0
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
                    # Calculate average similarity
                    avg_similarity = group["total_similarity"] / group["matching_chunk_count"]
                    
                    # Get file chunks with metadata
                    file_chunks = self.client.table("inna_file_chunks")\
                        .select("*")\
                        .eq("message_id", message_id)\
                        .order("chunk_index")\
                        .execute()
                    
                    if file_chunks.data:
                        # Group chunks by section
                        sections = {}
                        for chunk in file_chunks.data:
                            section = chunk["section_title"] or "Main Content"
                            if section not in sections:
                                sections[section] = []
                            sections[section].append(chunk["chunk_content"])
                        
                        # Combine chunks by section
                        message.data["sections"] = {
                            title: "".join(contents)
                            for title, contents in sections.items()
                        }
                        
                        # Boost similarity score for messages with file content
                        if group["matching_chunk_count"] > 1:
                            avg_similarity *= 1.2
                    
                    # Add similarity scores and chunk information
                    message.data["max_similarity"] = group["max_similarity"]
                    message.data["avg_similarity"] = avg_similarity
                    message.data["matching_chunk_count"] = group["matching_chunk_count"]
                    message.data["matching_chunks"] = sorted([
                        {
                            "chunk_index": chunk["chunk_index"],
                            "similarity": chunk["similarity"],
                            "text": chunk["text"],
                            "section_title": chunk.get("section_title")
                        }
                        for chunk in group["chunks"]
                    ], key=lambda x: x["similarity"], reverse=True)
                    
                    # If we have a text search, check content
                    if text_search:
                        text_lower = text_search.lower()
                        message_text = (message.data.get("text") or "").lower()
                        sections_text = "\n".join(message.data.get("sections", {}).values()).lower()
                        
                        if text_lower in message_text or text_lower in sections_text:
                            avg_similarity *= 1.3
                    
                    message.data["final_similarity"] = avg_similarity
                    result_messages.append(message.data)
            
            # Sort by final similarity score
            result_messages.sort(key=lambda x: x["final_similarity"], reverse=True)
            return result_messages[:limit]
            
        except Exception as e:
            logger.error(f"Error in search process: {str(e)}", exc_info=True)
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
        try:
            memory_data = {
                "agent_role": role,
                "chat_id": chat_id,
                "context": context,
                "embedding": embedding,
                "relevance_score": relevance_score,
                "metadata": metadata or {}
            }
            
            self._log_query("INSERT", "inna_agent_memory", memory_data)
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

# Create singleton instance
db_service = DatabaseService() 