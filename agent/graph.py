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
        
        # Get embedding for the current message
        query_text = state_obj.current_message.get("text", "")
        embedding = await openai_service.get_embedding(query_text)
        
        # Get similar messages
        similar_messages = await db_service.get_similar_messages(
            embedding=embedding,
            chat_id=state_obj.chat_id
        )
        
        # Update state
        state_obj.context = similar_messages
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def create_plan(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 2: Create a plan for responding to the message."""
    try:
        # Convert to AgentState for type safety
        state_obj = ensure_agent_state(state)
        
        context_text = "\n".join([
            f"- {msg['text']}" for msg in state_obj.context
        ])
        
        messages = [
            openai_service.create_system_message(
                "You are a planning agent. Create a brief plan for how to respond to the user's message using the available context."
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
        
        context_text = "\n".join([
            f"- {msg['text']}" for msg in state_obj.context
        ])
        
        messages = [
            openai_service.create_system_message(
                "You are Inna, a helpful and smart startup co-founder. Respond in a clear, professional manner."
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