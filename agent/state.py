"""State management for the agent graph."""

from typing import Dict, List, Any, Union, Annotated
from pydantic import BaseModel, Field

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