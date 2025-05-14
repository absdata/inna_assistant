# LangGraph workflow definition

from typing import Dict, List, Any, Tuple, Union, Annotated, TypeVar, cast
from langgraph.graph import Graph, END, StateGraph
import operator
from pydantic import BaseModel, Field
from services.azure_openai import openai_service
from services.database import db_service
from agent.roles.critic import critic_agent
from agent.roles.planner import planner_agent
from agent.roles.context import context_agent
from agent.roles.responder import responder_agent
from agent.roles.base import AgentMemory
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
    criticism: str = Field(default="")  # Store critic's feedback
    task_updates: str = Field(default="")  # Store planner's task updates

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            "messages": list(self.messages),
            "context": list(self.context),
            "current_message": dict(self.current_message),
            "chat_id": self.chat_id,
            "plan": str(self.plan),
            "response": str(self.response),
            "formatted_context": str(self.formatted_context),
            "criticism": str(self.criticism),
            "task_updates": str(self.task_updates)
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
        state_obj = ensure_agent_state(state)
        
        # Create memory object for context agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="context"
        )
        
        # Get context from context agent
        formatted_context = await context_agent.process(memory)
        
        # Update state with context
        state_obj.formatted_context = formatted_context
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def create_plan(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 2: Create a plan for responding to the message."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Create memory object for planner agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.formatted_context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="planner"
        )
        
        # Get plan from planner agent
        plan = await planner_agent.process(memory)
        state_obj.plan = plan
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in create_plan: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def analyze_with_critic(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 3: Analyze the plan with the critic agent."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Skip criticism for summary requests
        if is_summary_request(state_obj.current_message.get("text", "")):
            return state_obj.to_dict()
        
        # Create memory object for critic agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.formatted_context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            plan=state_obj.plan,
            role="critic"
        )
        
        # Get criticism from critic agent
        criticism = await critic_agent.process(memory)
        state_obj.criticism = criticism
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in analyze_with_critic: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def update_tasks_with_planner(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 4: Update tasks based on the conversation."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Create memory object for planner agent (task update mode)
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.formatted_context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            role="planner"
        )
        
        # Get task updates from planner agent
        task_updates = await planner_agent.process(memory)
        state_obj.task_updates = task_updates
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in update_tasks_with_planner: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def generate_response(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 5: Generate the final response."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Create memory object for responder agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.formatted_context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            role="responder"
        )
        
        # Get response from responder agent
        response = await responder_agent.process(memory)
        state_obj.response = response
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

def get_next_step(state: Union[Dict[str, Any], AgentState]) -> str:
    """Determine the next step in the workflow."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Check if this is a summary request
        if is_summary_request(state_obj.current_message.get("text", "")):
            if state_obj.response:
                return END
            elif state_obj.plan:
                return "generate_response"
            elif state_obj.formatted_context:
                return "create_plan"
            else:
                return "retrieve_context"
        
        # Normal flow
        if state_obj.response:
            return END
        elif state_obj.task_updates:
            return "generate_response"
        elif state_obj.criticism:
            return "update_tasks_with_planner"
        elif state_obj.plan:
            return "analyze_with_critic"
        elif state_obj.formatted_context:
            return "create_plan"
        else:
            return "retrieve_context"
    except Exception as e:
        logger.error(f"Error in get_next_step: {str(e)}", exc_info=True)
        return END

def is_summary_request(text: str) -> bool:
    """Check if the message is requesting a summary."""
    summary_keywords = ["summarize", "summary", "summarise", "summarisation", "summarization"]
    return any(keyword in text.lower() for keyword in summary_keywords)

def create_agent() -> Graph:
    """Create the agent workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("analyze_with_critic", analyze_with_critic)
    workflow.add_node("update_tasks_with_planner", update_tasks_with_planner)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("retrieve_context", get_next_step)
    workflow.add_edge("create_plan", get_next_step)
    workflow.add_edge("analyze_with_critic", get_next_step)
    workflow.add_edge("update_tasks_with_planner", get_next_step)
    workflow.add_edge("generate_response", get_next_step)
    
    # Set entry point
    workflow.set_entry_point("retrieve_context")
    
    return workflow.compile()

# Create the agent instance
agent = create_agent()