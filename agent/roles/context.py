from typing import List, Dict, Any
from .base import BaseAgent, AgentMemory
from services.azure_openai import openai_service
from services.database import db_service
import logging

logger = logging.getLogger(__name__)

class ContextAgent(BaseAgent):
    def __init__(self):
        super().__init__(role="context")
    
    async def process(self, memory: AgentMemory) -> str:
        """Process the current state and retrieve relevant context."""
        try:
            query_text = memory.current_message.get("text", "").lower()
            logger.info(f"Processing query: {query_text}")
            
            # Generate embedding for the query
            embedding = await openai_service.get_embedding(query_text)
            
            # Search parameters for comprehensive context
            search_params = {
                "chat_id": memory.chat_id,
                "query_embedding": embedding,
                "text_search": query_text,  # Always include text search
                "threshold": 0.3,  # Lower threshold to catch more relevant content
                "limit": 20,  # Higher limit to get more context
                "include_summaries": True,  # Include inna_summaries
                "include_agent_memory": True  # Include agent memory
            }
            
            # Perform the search
            messages = await db_service.search_messages_with_content(**search_params)
            logger.info(f"Found {len(messages)} relevant messages")
            
            # Process and organize the results
            doc_content = []
            chat_messages = []
            summaries = []
            agent_memories = []
            
            for msg in messages:
                relevance = msg.get("final_similarity", 0)
                msg_type = msg.get("type", "chat")  # Default to chat type
                
                if msg.get("file_content"):
                    chunks = msg.get("matching_chunks", [])
                    if chunks:
                        chunks.sort(key=lambda x: x["similarity"], reverse=True)
                        relevant_sections = []
                        
                        for chunk in chunks[:3]:  # Top 3 most relevant chunks
                            chunk_text = chunk["text"]
                            if "chunk_" in chunk_text:
                                chunk_text = chunk_text.split(":", 1)[1].strip()
                            
                            relevant_sections.append({
                                "content": chunk_text,
                                "similarity": chunk["similarity"]
                            })
                        
                        doc_content.append({
                            "sections": relevant_sections,
                            "relevance": relevance,
                            "created_at": msg.get("created_at"),
                            "file_name": msg.get("file_name", "Unknown Document")
                        })
                elif msg_type == "summary":
                    summaries.append({
                        "text": msg.get("text", ""),
                        "relevance": relevance,
                        "created_at": msg.get("created_at")
                    })
                elif msg_type == "agent_memory":
                    agent_memories.append({
                        "text": msg.get("text", ""),
                        "relevance": relevance,
                        "created_at": msg.get("created_at"),
                        "role": msg.get("role", "unknown")
                    })
                else:
                    chat_messages.append({
                        "text": msg.get("text", ""),
                        "relevance": relevance,
                        "created_at": msg.get("created_at")
                    })
            
            # Sort all lists by relevance
            doc_content.sort(key=lambda x: x["relevance"], reverse=True)
            chat_messages.sort(key=lambda x: x["relevance"], reverse=True)
            summaries.sort(key=lambda x: x["relevance"], reverse=True)
            agent_memories.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Format the context sections
            context_sections = []
            
            if doc_content:
                doc_text = "\n\n".join([
                    f"Document: {doc['file_name']} (Relevance: {doc['relevance']:.2f}):\n" +
                    "\n".join([
                        f"Section (Similarity: {section['similarity']:.2f}):\n{section['content']}"
                        for section in doc["sections"]
                    ])
                    for doc in doc_content[:5]  # Top 5 most relevant documents
                ])
                context_sections.append("### Document Content ###\n" + doc_text)
            
            if summaries:
                summary_text = "\n\n".join([
                    f"Summary (Relevance: {summary['relevance']:.2f}):\n{summary['text']}"
                    for summary in summaries[:3]  # Top 3 most relevant summaries
                ])
                context_sections.append("### Recent Summaries ###\n" + summary_text)
            
            if agent_memories:
                memory_text = "\n\n".join([
                    f"Agent Memory ({memory['role']}, Relevance: {memory['relevance']:.2f}):\n{memory['text']}"
                    for memory in agent_memories[:3]  # Top 3 most relevant memories
                ])
                context_sections.append("### Agent Memory ###\n" + memory_text)
            
            if chat_messages:
                msg_text = "\n\n".join([
                    f"Message (Relevance: {msg['relevance']:.2f}):\n{msg['text']}"
                    for msg in chat_messages[:5]  # Top 5 most relevant messages
                ])
                context_sections.append("### Chat Messages ###\n" + msg_text)
            
            formatted_context = "\n\n".join(context_sections)
            
            # Save the context to memory
            await self.save_memory(
                chat_id=memory.chat_id,
                context=f"Retrieved Context:\n{formatted_context}"
            )
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}", exc_info=True)
            return ""

context_agent = ContextAgent() 