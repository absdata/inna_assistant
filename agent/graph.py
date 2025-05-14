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
from config.config import config
import logging

# Create logger for this module
logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """State of the agent during execution."""
    messages: Annotated[List[Dict[str, str]], "multiple"] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    current_message: Dict[str, Any] = Field(default_factory=dict)
    chat_id: int = Field(default=0)
    plan: str = Field(default="")
    response: str = Field(default="")
    formatted_context: str = Field(default="")
    criticism: str = Field(default="")  # Store critic's feedback
    task_updates: str = Field(default="")  # Store planner's task updates
    should_process: bool = Field(default=False)  # Flag to indicate if message should be processed

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
            "task_updates": str(self.task_updates),
            "should_process": bool(self.should_process)
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

def check_agent_trigger(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Check if the message should trigger the agent."""
    try:
        state_obj = ensure_agent_state(state)
        message_text = state_obj.current_message.get("text", "").lower()
        file_content = state_obj.current_message.get("file_content")
        
        # Always process if there's a document
        if file_content:
            state_obj.should_process = True
            return state_obj.to_dict()
        
        # Check if message starts with any of the agent name triggers
        state_obj.should_process = any(
            message_text.startswith(trigger.lower())
            for trigger in config.agent_name_triggers
        )
        
        if state_obj.should_process:
            # Remove the trigger from the message text
            for trigger in config.agent_name_triggers:
                if message_text.startswith(trigger.lower()):
                    cleaned_text = message_text[len(trigger):].strip(" ,")
                    state_obj.current_message["text"] = cleaned_text
                    break
        
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in check_agent_trigger: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def retrieve_context(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 1: Retrieve relevant context from vector store."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Skip if we shouldn't process this message
        if not state_obj.should_process:
            return state_obj.to_dict()
        
        # Create memory object for context agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="context",
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            task_updates=state_obj.task_updates,
            formatted_context=state_obj.formatted_context
        )
        
        # Get context from context agent
        # This will update memory.context with the structured list
        formatted_context = await context_agent.process(memory)
        
        # Update state with both formats
        state_obj.context = memory.context  # Structured list
        state_obj.formatted_context = formatted_context  # Formatted string
        return state_obj.to_dict()
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}", exc_info=True)
        return ensure_dict_state(state)

async def create_plan(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """Node 2: Create a plan for responding to the message."""
    try:
        state_obj = ensure_agent_state(state)
        
        # Skip if we shouldn't process this message
        if not state_obj.should_process:
            return state_obj.to_dict()
        
        # Create memory object for planner agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="planner",
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            task_updates=state_obj.task_updates,
            formatted_context=state_obj.formatted_context
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
        
        # Skip if we shouldn't process this message
        if not state_obj.should_process:
            return state_obj.to_dict()
        
        # Skip criticism for summary requests
        if is_summary_request(state_obj.current_message.get("text", "")):
            return state_obj.to_dict()
        
        # Create memory object for critic agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="critic",
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            task_updates=state_obj.task_updates,
            formatted_context=state_obj.formatted_context
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
        
        # Skip if we shouldn't process this message
        if not state_obj.should_process:
            return state_obj.to_dict()
        
        # Create memory object for planner agent (task update mode)
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="planner",
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            task_updates=state_obj.task_updates,
            formatted_context=state_obj.formatted_context
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
        
        # Skip if we shouldn't process this message
        if not state_obj.should_process:
            return state_obj.to_dict()
        
        # Create memory object for responder agent
        memory = AgentMemory(
            messages=state_obj.messages,
            context=state_obj.context,
            current_message=state_obj.current_message,
            chat_id=state_obj.chat_id,
            role="responder",
            plan=state_obj.plan,
            criticism=state_obj.criticism,
            task_updates=state_obj.task_updates,
            formatted_context=state_obj.formatted_context
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
        
        # If we shouldn't process this message, end immediately
        if not state_obj.should_process:
            return END
        
        # If we have a response, we're done
        if state_obj.response:
            return END
        
        # For document uploads and summary requests, we can skip some steps
        has_document = bool(state_obj.current_message.get("file_content"))
        is_summary = is_summary_request(state_obj.current_message.get("text", ""))
        
        if has_document or is_summary:
            if state_obj.context:
                return "generate_response"
            return END
        
        # Normal flow - follow the graph edges
        return END
            
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
    workflow.add_node("check_trigger", check_agent_trigger)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("analyze_with_critic", analyze_with_critic)
    workflow.add_node("update_tasks_with_planner", update_tasks_with_planner)
    workflow.add_node("generate_response", generate_response)
    
    # Set entry point
    workflow.set_entry_point("check_trigger")
    
    # Add edges for the workflow
    workflow.add_edge("check_trigger", "retrieve_context")
    workflow.add_edge("retrieve_context", "create_plan")
    workflow.add_edge("create_plan", "analyze_with_critic")
    workflow.add_edge("analyze_with_critic", "update_tasks_with_planner")
    workflow.add_edge("update_tasks_with_planner", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Add conditional edges for early exits
    workflow.add_conditional_edges(
        "check_trigger",
        get_next_step,
        {
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "retrieve_context",
        get_next_step,
        {
            "generate_response": "generate_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "create_plan",
        get_next_step,
        {
            "generate_response": "generate_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_with_critic",
        get_next_step,
        {
            "generate_response": "generate_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "update_tasks_with_planner",
        get_next_step,
        {
            "generate_response": "generate_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_response",
        get_next_step,
        {
            END: END
        }
    )
    
    return workflow.compile()

# Create the agent instance
agent = create_agent()