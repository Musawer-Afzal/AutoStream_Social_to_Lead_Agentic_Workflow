from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    """State management for the LangGraph agent"""
    messages: List[str]
    intent: str
    selected_plan: Optional[str]  # 'Basic' or 'Pro'
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool
    waiting_for: Optional[str]  # 'plan', 'name', 'email', 'platform'
    conversation_history: List[Dict[str, str]]


def create_initial_state() -> AgentState:
    """Create a fresh initial state for a new conversation"""
    return {
        "messages": [],
        "intent": "",
        "selected_plan": None,
        "name": None,
        "email": None,
        "platform": None,
        "lead_captured": False,
        "waiting_for": None,
        "conversation_history": []
    }