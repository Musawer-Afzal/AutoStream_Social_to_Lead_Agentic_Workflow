from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    """State management for the LangGraph agent
    
    This state persists across conversation turns and tracks:
    - Conversation history
    - Current intent classification
    - Lead information collection progress
    - Tool execution status
    """
    messages: List[str]
    intent: str
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool
    waiting_for: Optional[str]
    conversation_history: List[Dict[str, str]]


def create_initial_state() -> AgentState:
    """Create a fresh initial state for a new conversation"""
    return {
        "messages": [],
        "intent": "",
        "name": None,
        "email": None,
        "platform": None,
        "lead_captured": False,
        "waiting_for": None,
        "conversation_history": []
    }