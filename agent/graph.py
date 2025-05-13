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

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            "messages": list(self.messages),
            "context": list(self.context),
            "current_message": dict(self.current_message),
            "chat_id": self.chat_id,
            "plan": str(self.plan),
            "response": str(self.response)
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
        
        # Check if this is a summarization request
        is_summary_request = any(
            keyword in query_text 
            for keyword in ["summarize", "summary", "summarise", "summarisation", "summarization"]
        )
        
        if is_summary_request:
            # For summarization, get full chat history
            logger.info("Summarization request detected - retrieving full chat history")
            chat_history = await db_service.get_chat_history(
                chat_id=state_obj.chat_id,
                limit=100  # Get last 100 messages
            )
            state_obj.context = chat_history
        else:
            # For regular queries, get similar messages with lower threshold and higher limit
            # to ensure we catch file content matches
            embedding = await openai_service.get_embedding(query_text)
            similar_messages = await db_service.get_similar_messages(
                embedding=embedding,
                chat_id=state_obj.chat_id,
                threshold=0.5,  # Lower threshold for better recall
                limit=10  # Increased limit to get more potential matches
            )
            
            # Get the full message details for each similar message
            full_messages = []
            for msg in similar_messages:
                # Get the full message from the messages table
                result = await db_service.client.table("inna_messages")\
                    .select("*")\
                    .eq("id", msg["id"])\
                    .execute()
                
                if result.data:
                    full_msg = result.data[0]
                    # If there's file content, add it to the text field for context
                    if full_msg.get("file_content"):
                        full_msg["text"] = f"{full_msg.get('text', '')}\n\nFile Content:\n{full_msg['file_content']}"
                    full_messages.append(full_msg)
            
            state_obj.context = full_messages
            
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
        
        # Format context with clear separation between messages and file content
        context_items = []
        for msg in state_obj.context:
            if msg.get("text"):
                # Split text to separate message text from file content
                text_parts = msg["text"].split("\n\nFile Content:\n")
                if len(text_parts) > 1:
                    context_items.append(f"Message: {text_parts[0]}")
                    context_items.append(f"Attached Document Content: {text_parts[1]}")
                else:
                    context_items.append(f"Message: {text_parts[0]}")
        
        context_text = "\n\n".join(context_items)
        
        if is_summary_request:
            messages = [
                openai_service.create_system_message(
                    "You are a planning agent. Create a plan for summarizing the chat history in a clear, organized way."
                ),
                openai_service.create_user_message(
                    f"Chat History:\n{context_text}\n\nCreate a plan for summarizing this chat history:"
                )
            ]
        else:
            messages = [
                openai_service.create_system_message(
                    "You are a planning agent. Create a brief plan for how to respond to the user's message using the available context. Pay special attention to any document content in the context."
                ),
                openai_service.create_user_message(
                    f"Context:\n{context_text}\n\nUser message: {state_obj.current_message.get('text', '')}\n\nCreate a plan:"
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
        
        # Format context with clear separation between messages and file content
        context_items = []
        for msg in state_obj.context:
            if msg.get("text"):
                # Split text to separate message text from file content
                text_parts = msg["text"].split("\n\nFile Content:\n")
                if len(text_parts) > 1:
                    context_items.append(f"Message: {text_parts[0]}")
                    context_items.append(f"Attached Document Content: {text_parts[1]}")
                else:
                    context_items.append(f"Message: {text_parts[0]}")
        
        context_text = "\n\n".join(context_items)
        
        if is_summary_request:
            messages = [
                openai_service.create_system_message(
                    "You are Inna, a helpful and smart startup co-founder. Create a clear, well-organized summary of the chat history. Focus on key points, decisions, and important information. Use sections and bullet points for better readability."
                ),
                openai_service.create_user_message(
                    f"Chat History:\n{context_text}\n\nPlan:\n{state_obj.plan}\n\nCreate a comprehensive summary:"
                )
            ]
        else:
            messages = [
                openai_service.create_system_message(
                    "You are Inna, a helpful and smart startup co-founder. Respond in a clear, professional manner. When referencing document content, be specific about which parts you're using to answer the question."
                ),
                openai_service.create_user_message(
                    f"Context:\n{context_text}\n\nPlan:\n{state_obj.plan}\n\nUser message: {state_obj.current_message.get('text', '')}\n\nRespond:"
                )
            ]
        
        state_obj.response = await openai_service.get_completion(messages, temperature=0.7)
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