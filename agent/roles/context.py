from typing import List, Dict, Any, Optional
from .base import BaseAgent, AgentMemory
from services.azure_openai import openai_service
from services.database import db_service
from ..memory import MemoryStore, MemoryType
import logging

logger = logging.getLogger(__name__)

class ContextAgent(BaseAgent):
    def __init__(self):
        super().__init__(role="context")
        self.memory_store = MemoryStore()
    
    async def process(self, memory: AgentMemory) -> str:
        """Process the current state and retrieve relevant context."""
        try:
            query_text = memory.current_message.get("text", "").lower()
            logger.info(f"Processing query: {query_text}")
            
            # Generate embedding for the query
            embedding = await openai_service.get_embedding(query_text)
            logger.info("Generated embedding for query")
            
            # Search parameters for comprehensive context
            search_params = {
                "chat_id": memory.chat_id,
                "query_embedding": embedding,
                "text_search": query_text,  # Always include text search
                "threshold": 0.3,  # Lower threshold to catch more relevant content
                "limit": 20  # Higher limit to get more context
            }
            
            logger.info("Starting context retrieval process")
            
            # Get relevant memories from memory store
            logger.info("Retrieving relevant memories from memory store")
            relevant_memories = await self.memory_store.get_relevant_memories(
                chat_id=memory.chat_id,
                query=query_text,
                threshold=0.3,
                limit=20
            )
            
            # Perform regular search for messages and documents
            logger.info("Searching for relevant messages and documents")
            messages = await db_service.search_messages_with_content(**search_params)
            logger.info(f"Found {len(messages)} relevant messages and {len(relevant_memories)} memories")
            
            # Process and organize the results
            doc_content = []
            chat_messages = []
            summaries = []
            agent_insights = []
            
            # Process messages from database
            logger.info("Processing and organizing search results")
            for msg in messages:
                relevance = msg.get("final_similarity", 0)
                
                if msg.get("file_content") or msg.get("sections"):
                    chunks = msg.get("matching_chunks", [])
                    sections = msg.get("sections", {})
                    logger.info(f"Processing document with {len(sections)} sections and {len(chunks)} matching chunks")
                    
                    # Generate document summary if this is a new document upload
                    if msg.get("id") == memory.current_message.get("message_id") and msg.get("file_content"):
                        logger.info("Generating summary for newly uploaded document")
                        summary = await self._generate_document_summary(msg["file_content"])
                        if summary:
                            doc_content.append({
                                "type": "document_summary",
                                "content": summary,
                                "relevance": 1.0,  # High relevance for new document
                                "file_name": msg.get("file_url", "Uploaded document"),
                                "created_at": msg.get("created_at")
                            })
                    if sections:
                        relevant_sections = []
                        # Sort sections by similarity
                        sorted_sections = sorted(
                            sections.items(),
                            key=lambda x: x[1]["similarity"],
                            reverse=True
                        )
                        logger.info(f"Sorted sections by similarity: {[(title, data['similarity']) for title, data in sorted_sections]}")
                        
                        for section_title, section_data in sorted_sections[:3]:  # Top 3 most relevant sections
                            logger.info(f"Adding section '{section_title}' with similarity {section_data['similarity']:.4f}")
                            relevant_sections.append({
                                "content": section_data["content"],
                                "similarity": section_data["similarity"],
                                "title": section_title
                            })
                        
                        doc_content.append({
                            "sections": relevant_sections,
                            "relevance": relevance,
                            "created_at": msg.get("created_at"),
                            "file_name": msg.get("file_name", "Unknown Document")
                        })
                        logger.info(f"Added document '{msg.get('file_name', 'Unknown Document')}' with {len(relevant_sections)} relevant sections")
                else:
                    chat_messages.append({
                        "text": msg.get("text", ""),
                        "relevance": relevance,
                        "created_at": msg.get("created_at")
                    })
            
            # Process memories from memory store
            logger.info("Processing memories from memory store")
            for mem in relevant_memories:
                if mem.memory_type == MemoryType.SUMMARY:
                    summaries.append({
                        "text": mem.content,
                        "relevance": mem.relevance_score,
                        "created_at": mem.created_at
                    })
                elif mem.memory_type in [MemoryType.CHAT, MemoryType.TASK]:
                    agent_insights.append({
                        "text": mem.content,
                        "relevance": mem.relevance_score,
                        "created_at": mem.created_at,
                        "type": mem.memory_type.value
                    })
            
            # Sort all lists by relevance
            doc_content.sort(key=lambda x: x["relevance"], reverse=True)
            chat_messages.sort(key=lambda x: x["relevance"], reverse=True)
            summaries.sort(key=lambda x: x["relevance"], reverse=True)
            agent_insights.sort(key=lambda x: x["relevance"], reverse=True)
            
            logger.info(f"Organized content: {len(doc_content)} documents, {len(chat_messages)} messages, "
                       f"{len(summaries)} summaries, {len(agent_insights)} insights")
            
            # Create context list
            context_list = []
            
            # Add documents to context
            for doc in doc_content[:5]:  # Top 5 most relevant documents
                context_list.append({
                    "type": "document",
                    "content": doc["sections"],
                    "relevance": doc["relevance"],
                    "file_name": doc["file_name"],
                    "created_at": doc["created_at"]
                })
            
            # Add summaries to context
            for summary in summaries[:3]:  # Top 3 most relevant summaries
                context_list.append({
                    "type": "summary",
                    "content": summary["text"],
                    "relevance": summary["relevance"],
                    "created_at": summary["created_at"]
                })
            
            # Add agent insights to context
            for insight in agent_insights[:3]:  # Top 3 most relevant insights
                context_list.append({
                    "type": "agent_insight",
                    "content": insight["text"],
                    "relevance": insight["relevance"],
                    "insight_type": insight["type"],
                    "created_at": insight["created_at"]
                })
            
            # Add chat messages to context
            for msg in chat_messages[:5]:  # Top 5 most relevant messages
                context_list.append({
                    "type": "chat",
                    "content": msg["text"],
                    "relevance": msg["relevance"],
                    "created_at": msg["created_at"]
                })
            
            logger.info(f"Created context list with {len(context_list)} items")
            
            # Format the context for display
            formatted_sections = []
            
            if doc_content:
                doc_text = "\n\n".join([
                    f"Document: {doc['file_name']} (Relevance: {doc['relevance']:.2f}):\n" +
                    "\n".join([
                        f"Section '{section['title']}' (Similarity: {section['similarity']:.2f}):\n{section['content']}"
                        for section in doc["sections"]
                    ])
                    for doc in doc_content[:5]
                ])
                formatted_sections.append("### Document Content ###\n" + doc_text)
                logger.info(f"Added {len(doc_content[:5])} documents to formatted context")
            
            if summaries:
                summary_text = "\n\n".join([
                    f"Summary (Relevance: {summary['relevance']:.2f}):\n{summary['text']}"
                    for summary in summaries[:3]
                ])
                formatted_sections.append("### Recent Summaries ###\n" + summary_text)
                logger.info(f"Added {len(summaries[:3])} summaries to formatted context")
            
            if agent_insights:
                insight_text = "\n\n".join([
                    f"Agent {insight['type'].title()} (Relevance: {insight['relevance']:.2f}):\n{insight['text']}"
                    for insight in agent_insights[:3]
                ])
                formatted_sections.append("### Agent Insights ###\n" + insight_text)
                logger.info(f"Added {len(agent_insights[:3])} insights to formatted context")
            
            if chat_messages:
                msg_text = "\n\n".join([
                    f"Message (Relevance: {msg['relevance']:.2f}):\n{msg['text']}"
                    for msg in chat_messages[:5]
                ])
                formatted_sections.append("### Chat Messages ###\n" + msg_text)
                logger.info(f"Added {len(chat_messages[:5])} messages to formatted context")
            
            formatted_context = "\n\n".join(formatted_sections)
            logger.info("Completed context formatting")
            
            # Save the context to memory
            await self.save_memory(
                chat_id=memory.chat_id,
                context=f"Retrieved Context:\n{formatted_context}"
            )
            
            # Return both the formatted string and the structured context list
            memory.context = context_list
            logger.info("Context processing completed successfully")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}", exc_info=True)
            memory.context = []
            return ""

    async def _generate_document_summary(self, content: str) -> Optional[str]:
        """Generate a summary for a document."""
        try:
            messages = [
                openai_service.create_system_message(
                    "You are a document analyst. Create a concise but comprehensive summary of the document that includes:\n"
                    "1. Main topics and key points\n"
                    "2. Important findings or conclusions\n"
                    "3. Any action items or recommendations\n"
                    "\nFormat the summary in a clear, structured way using bullet points."
                ),
                openai_service.create_user_message(
                    f"Please summarize this document:\n\n{content}"
                )
            ]
            
            summary = await openai_service.get_completion(messages)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating document summary: {str(e)}")
            return None

context_agent = ContextAgent() 