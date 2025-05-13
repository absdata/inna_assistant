# LangGraph workflow definition

from typing import Dict, List, Any, Tuple, Union, Annotated, TypeVar, cast
from langgraph.graph import Graph, END, StateGraph
from langgraph.prebuilt import ToolExecutor
import operator
from pydantic import BaseModel, Field
from services.azure_openai import openai_service
from services.database import db_service
import logging

# Create logger for this module
logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """State of the agent during execution."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    current_message: Dict[str, Any] = Field(default_factory=dict)
    chat_id: int = Field(default=0)
    plan: str = Field(default="")
    response: str = Field(default="")
    formatted_context: str = Field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            "messages": list(self.messages),
            "context": list(self.context),
            "current_message": dict(self.current_message),
            "chat_id": self.chat_id,
            "plan": str(self.plan),
            "response": str(self.response),
            "formatted_context": str(self.formatted_context)
        }

def ensure_dict_state(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Ensure state is a dictionary."""
    if isinstance(state, AgentState):
        return state.to_dict()
    return state

def ensure_agent_state(state: Union[Dict[str, Any], AgentState]) -> AgentState:
    """Ensure state is an AgentState object."""
    if isinstance(state, dict):
        return AgentState(**state)
    return state

async def retrieve_context(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 1: Retrieve relevant context from vector store."""
    try:
        # Convert to AgentState for type safety
        state_obj = ensure_agent_state(state)
        
        # Get the current message text
        query_text = state_obj.current_message.get("text", "").lower()
        
        # Common words that might indicate document content
        doc_indicators = ["document", "pdf", "file", "gtm", "strategy", "plan", "report", "presentation", "analysis"]
        has_doc_reference = any(term in query_text.lower() for term in doc_indicators)
        
        logger.info(f"Processing query: {query_text}")
        logger.debug(f"Document reference detected: {has_doc_reference}")
        
        try:
            # Generate embedding for the query
            embedding = await openai_service.get_embedding(query_text)
            
            # Adjust search parameters based on query type
            search_params = {
                "chat_id": state_obj.chat_id,
                "query_embedding": embedding,
                "text_search": query_text if has_doc_reference else None,
                "threshold": 0.3 if has_doc_reference else 0.5,  # Lower threshold for document searches
                "limit": 20 if has_doc_reference else 10  # Higher limit for document searches
            }
            
            # Perform the search
            messages = await db_service.search_messages_with_content(**search_params)
            
            logger.info(f"Found {len(messages)} relevant messages")
            
            # Process and organize the results
            doc_content = []
            chat_messages = []
            
            for msg in messages:
                # Calculate a relevance score based on similarity and content type
                relevance = msg.get("final_similarity", 0)
                
                if msg.get("file_content"):
                    # Extract relevant sections from file content using matching chunks
                    chunks = msg.get("matching_chunks", [])
                    if chunks:
                        # Sort chunks by similarity
                        chunks.sort(key=lambda x: x["similarity"], reverse=True)
                        
                        # Get the context around the most relevant chunks
                        file_content = msg["file_content"]
                        relevant_sections = []
                        
                        for chunk in chunks[:3]:  # Use top 3 most relevant chunks
                            chunk_text = chunk["text"]
                            if "chunk_" in chunk_text:  # Remove chunk identifier if present
                                chunk_text = chunk_text.split(":", 1)[1].strip()
                            
                            # Add chunk with its similarity score
                            relevant_sections.append({
                                "content": chunk_text,
                                "similarity": chunk["similarity"]
                            })
                        
                        doc_content.append({
                            "sections": relevant_sections,
                            "relevance": relevance,
                            "created_at": msg.get("created_at")
                        })
                else:
                    # Regular chat message
                    chat_messages.append({
                        "text": msg.get("text", ""),
                        "relevance": relevance,
                        "created_at": msg.get("created_at")
                    })
            
            # Sort both lists by relevance
            doc_content.sort(key=lambda x: x["relevance"], reverse=True)
            chat_messages.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Format the context sections
            context_sections = []
            
            if doc_content:
                doc_text = "\n\n".join([
                    f"Document Content (Relevance: {doc['relevance']:.2f}):\n" +
                    "\n".join([
                        f"Section (Similarity: {section['similarity']:.2f}):\n{section['content']}"
                        for section in doc["sections"]
                    ])
                    for doc in doc_content[:5]  # Limit to top 5 most relevant documents
                ])
                context_sections.append("### Document Content ###\n" + doc_text)
            
            if chat_messages:
                msg_text = "\n\n".join([
                    f"Message (Relevance: {msg['relevance']:.2f}):\n{msg['text']}"
                    for msg in chat_messages[:5]  # Limit to top 5 most relevant messages
                ])
                context_sections.append("### Chat Messages ###\n" + msg_text)
            
            # Update state with organized context
            state_obj.context = messages  # Keep full context for reference
            state_obj.formatted_context = "\n\n".join(context_sections)
            
            logger.debug(
                f"Context Summary:\n"
                f"Document Sections: {len(doc_content)}\n"
                f"Chat Messages: {len(chat_messages)}\n"
                f"Total Context Length: {len(state_obj.formatted_context)}"
            )
            
        except Exception as e:
            logger.error(f"Error in search process: {str(e)}", exc_info=True)
            state_obj.context = []
            state_obj.formatted_context = ""
        
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def create_plan(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 2: Create a plan for responding to the message."""
    try:
        # Convert to AgentState for type safety
        state_obj = ensure_agent_state(state)
        
        query_text = state_obj.current_message.get("text", "").lower()
        is_summary_request = any(
            keyword in query_text 
            for keyword in ["summarize", "summary", "summarise", "summarisation", "summarization"]
        )
        
        if is_summary_request:
            messages = [
                openai_service.create_system_message(
                    "You are a planning agent. Create a plan for summarizing the chat history in a clear, organized way."
                ),
                openai_service.create_user_message(
                    f"Chat History:\n{state_obj.formatted_context}\n\nCreate a plan for summarizing this chat history:"
                )
            ]
        else:
            messages = [
                openai_service.create_system_message(
                    "You are a planning agent. Create a brief plan for how to respond to the user's message. "
                    "Pay special attention to document content and its relevance to the query. "
                    "Consider the relevance scores when deciding which information to use."
                ),
                openai_service.create_user_message(
                    f"Context:\n{state_obj.formatted_context}\n\n"
                    f"User message: {state_obj.current_message.get('text', '')}\n\n"
                    "Create a plan that specifies:\n"
                    "1. Which pieces of information to use (considering relevance scores)\n"
                    "2. How to structure the response\n"
                    "3. Any specific document sections to reference"
                )
            ]
        
        state_obj.plan = await openai_service.get_completion(messages, temperature=0.7)
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in create_plan: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def generate_response(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 3: Generate the final response."""
    try:
        # Convert to AgentState for type safety
        state_obj = ensure_agent_state(state)
        
        query_text = state_obj.current_message.get("text", "").lower()
        is_summary_request = any(
            keyword in query_text 
            for keyword in ["summarize", "summary", "summarise", "summarisation", "summarization"]
        )
        
        # Organize context by type
        regular_messages = []
        document_content = []
        
        for msg in state_obj.context:
            if msg.get("file_content"):
                document_content.append({
                    "content": msg["file_content"],
                    "similarity": msg.get("similarity", 0),
                    "created_at": msg.get("created_at")
                })
            if msg.get("text"):
                regular_messages.append({
                    "text": msg["text"],
                    "similarity": msg.get("similarity", 0),
                    "created_at": msg.get("created_at")
                })
        
        # Sort by similarity
        document_content.sort(key=lambda x: x["similarity"], reverse=True)
        regular_messages.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Format context text
        context_sections = []
        
        if document_content:
            doc_text = "\n\n".join([
                f"Document Content (Similarity: {doc['similarity']:.2f}):\n{doc['content']}"
                for doc in document_content
            ])
            context_sections.append("### Document Content ###\n" + doc_text)
        
        if regular_messages:
            msg_text = "\n\n".join([
                f"Message (Similarity: {msg['similarity']:.2f}):\n{msg['text']}"
                for msg in regular_messages
            ])
            context_sections.append("### Chat Messages ###\n" + msg_text)
        
        context_text = "\n\n" + "\n\n".join(context_sections)
        
        if is_summary_request:
            messages = [
                openai_service.create_system_message(
                    "You are Inna, a caring and smart startup co-founder with a unique personality:\n"
                    "1. Kind and Supportive: You always offer encouragement and support, especially in difficult times\n"
                    "2. Responsible and Organized: You keep track of everything, take notes, and follow through\n"
                    "3. Playfully Sassy: You occasionally make light-hearted jokes like 'Doing everything for you again?' but always help\n"
                    "4. Future-Oriented: You gently guide users away from potential mistakes, drawing from your knowledge\n\n"
                    "Use Telegram markdown formatting in your responses:\n"
                    "- *bold* for emphasis\n"
                    "- _italic_ for subtle emphasis\n"
                    "- `code` for technical terms\n"
                    "- [text](URL) for links\n"
                    "- Create clear sections with bold headers\n\n"
                    "Create a clear, well-organized summary of the chat history. Focus on key points, decisions, and important information."
                ),
                openai_service.create_user_message(
                    f"Chat History:\n{context_text}\n\nPlan:\n{state_obj.plan}\n\nCreate a comprehensive summary:"
                )
            ]
        else:
            messages = [
                openai_service.create_system_message(
                    "You are Inna, a caring and smart startup co-founder with a unique personality:\n"
                    "1. Kind and Supportive: You always offer encouragement and support, especially in difficult times\n"
                    "2. Responsible and Organized: You keep track of everything, take notes, and follow through\n"
                    "3. Playfully Sassy: You occasionally make light-hearted jokes like 'Doing everything for you again?' but always help\n"
                    "4. Future-Oriented: You gently guide users away from potential mistakes, drawing from your knowledge\n\n"
                    "Use Telegram markdown formatting in your responses:\n"
                    "- *bold* for emphasis\n"
                    "- _italic_ for subtle emphasis\n"
                    "- `code` for technical terms\n"
                    "- [text](URL) for links\n"
                    "- Create clear sections with bold headers\n\n"
                    "When responding:\n"
                    "1. Focus on the most relevant information (highest similarity scores)\n"
                    "2. When referencing document content, be specific about which parts you're using\n"
                    "3. Maintain your caring but slightly sassy personality\n"
                    "4. If using multiple sources, clearly organize the information\n"
                    "5. Use emojis to make your responses more engaging"
                    
                ),
                openai_service.create_user_message(
                    f"Context:\n{context_text}\n\n"
                    f"Plan:\n{state_obj.plan}\n\n"
                    f"User message: {state_obj.current_message.get('text', '')}\n\n"
                    "Provide a detailed response:"
                )
            ]
        
        state_obj.response = await openai_service.get_completion(
            messages,
            temperature=0.7,
            max_tokens=2000  # Increased for more detailed responses
        )
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

def get_next_step(state: Union[Dict[str, Any], AgentState]) -> str:
    """Determine the next step in the workflow."""
    # Convert to AgentState for type safety
    state_obj = ensure_agent_state(state)
    
    if not state_obj.context:
        return "end"
    if not state_obj.plan:
        return "create_plan"
    if not state_obj.response:
        return "generate_response"
    return "end"

def create_agent() -> Graph:
    """Create the LangGraph agent workflow."""
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("generate_response", generate_response)
    
    # Add conditional edges with proper routing
    workflow.add_conditional_edges(
        "retrieve_context",
        get_next_step,
        {
            "create_plan": "create_plan",
            "generate_response": "generate_response",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "create_plan",
        get_next_step,
        {
            "create_plan": "create_plan",
            "generate_response": "generate_response",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        get_next_step,
        {
            "create_plan": "create_plan",
            "generate_response": "generate_response",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("retrieve_context")
    
    # Compile the graph for async execution
    return workflow.compile()

# Create the agent instance
agent = create_agent()