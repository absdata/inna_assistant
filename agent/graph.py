# LangGraph workflow definition

from typing import Dict, List, Any, Tuple, Union, Annotated
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolExecutor
import operator
from pydantic import BaseModel, Field
from services.azure_openai import openai_service
from services.database import db_service

class AgentState(BaseModel):
    """State of the agent during execution."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    context: List[Dict[str, Any]] = Field(default_factory=list)
    current_message: Dict[str, Any] = Field(default_factory=dict)
    chat_id: int = Field(default=0)
    plan: str = Field(default="")
    response: str = Field(default="")

async def retrieve_context(state: AgentState) -> dict:
    """Node 1: Retrieve relevant context from vector store."""
    try:
        # Get embedding for the current message
        query_text = state.current_message.get("text", "")
        embedding = await openai_service.get_embedding(query_text)
        
        # Get similar messages
        similar_messages = await db_service.get_similar_messages(
            embedding=embedding,
            chat_id=state.chat_id
        )
        
        state.context = similar_messages
        return {"state": state, "next": "create_plan"}
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}", exc_info=True)
        return {"state": state, "next": END}

async def create_plan(state: AgentState) -> dict:
    """Node 2: Create a plan for responding to the message."""
    try:
        context_text = "\n".join([
            f"- {msg['text']}" for msg in state.context
        ])
        
        messages = [
            openai_service.create_system_message(
                "You are a planning agent. Create a brief plan for how to respond to the user's message using the available context."
            ),
            openai_service.create_user_message(
                f"Context:\n{context_text}\n\nUser message: {state.current_message.get('text', '')}\n\nCreate a plan:"
            )
        ]
        
        state.plan = await openai_service.get_completion(messages, temperature=0.7)
        return {"state": state, "next": "generate_response"}
    except Exception as e:
        logger.error(f"Error in create_plan: {str(e)}", exc_info=True)
        return {"state": state, "next": END}

async def generate_response(state: AgentState) -> dict:
    """Node 3: Generate the final response."""
    try:
        context_text = "\n".join([
            f"- {msg['text']}" for msg in state.context
        ])
        
        messages = [
            openai_service.create_system_message(
                "You are Inna, a helpful and smart startup co-founder. Respond in a clear, professional manner."
            ),
            openai_service.create_user_message(
                f"Context:\n{context_text}\n\nPlan:\n{state.plan}\n\nUser message: {state.current_message.get('text', '')}\n\nRespond:"
            )
        ]
        
        state.response = await openai_service.get_completion(messages, temperature=0.7)
        return {"state": state, "next": END}
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
        return {"state": state, "next": END}

def create_agent() -> Graph:
    """Create the LangGraph agent workflow."""
    # Create the workflow graph
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges based on conditional returns
    workflow.add_edge("retrieve_context", "create_plan")
    workflow.add_edge("create_plan", "generate_response")
    workflow.add_edge("retrieve_context", END)
    workflow.add_edge("create_plan", END)
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve_context")
    
    # Compile the graph for async execution
    return workflow.compile()

# Create the agent instance
agent = create_agent()