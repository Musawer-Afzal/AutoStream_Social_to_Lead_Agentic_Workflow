import os
from typing import Dict, Any, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from state import AgentState, create_initial_state
from rag import retrieve_info
from intent import detect_intent
from tools import mock_lead_capture, validate_email, extract_info_from_message

# Initialize LLM with Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Fast and efficient model
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    timeout=30,
    max_retries=2,
)


class AutoStreamAgent:
    """Conversational AI Agent for AutoStream using LangGraph
    
    This agent handles:
    - Multi-turn conversations with state persistence
    - Intent detection and routing
    - RAG-powered knowledge retrieval
    - Lead qualification and capture
    """
    
    def __init__(self):
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each conversation state
        workflow.add_node("process_intent", self.process_intent)
        workflow.add_node("handle_greeting", self.handle_greeting)
        workflow.add_node("handle_inquiry", self.handle_inquiry)
        workflow.add_node("handle_high_intent", self.handle_high_intent)
        
        # Set entry point
        workflow.set_entry_point("process_intent")
        
        # Add conditional routing based on intent
        workflow.add_conditional_edges(
            "process_intent",
            self.route_intent,
            {
                "greeting": "handle_greeting",
                "inquiry": "handle_inquiry",
                "high_intent": "handle_high_intent"
            }
        )
        
        # All handlers lead to end
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("handle_inquiry", END)
        workflow.add_edge("handle_high_intent", END)
        
        return workflow.compile()
    
    def process_intent(self, state: AgentState) -> Dict[str, Any]:
        """Process and classify user intent"""
        if not state.get("messages"):
            return {"intent": "greeting"}
        
        last_message = state["messages"][-1]
        intent = detect_intent(last_message, llm)
        
        return {"intent": intent}
    
    def route_intent(self, state: AgentState) -> str:
        """Route to appropriate handler based on detected intent"""
        return state.get("intent", "inquiry")
    
    def handle_greeting(self, state: AgentState) -> Dict[str, Any]:
        """Handle greeting intent with friendly welcome"""
        response = """👋 **Welcome to AutoStream Support!**

I'm your AI assistant for AutoStream, the automated video editing platform.

I can help you with:
• 📊 **Pricing & Plans** - Basic ($29) and Pro ($79) options
• ✨ **Features** - Video editing, AI captions, 4K resolution
• 💰 **Policies** - Refunds, support availability
• 🚀 **Signing Up** - Get started with Pro plan

What would you like to know about AutoStream?"""
        
        return {"messages": state["messages"] + [response]}
    
    def handle_inquiry(self, state: AgentState) -> Dict[str, Any]:
        """Handle inquiry intent with RAG retrieval"""
        last_question = state["messages"][-1]
        
        # Retrieve relevant information from knowledge base
        info = retrieve_info(last_question)
        
        response = f"""📚 **Information about AutoStream**

{info}

Is there anything specific you'd like to know more about? I can help with pricing details, feature comparisons, or signing up!"""
        
        return {"messages": state["messages"] + [response]}
    
    def handle_high_intent(self, state: AgentState) -> Dict[str, Any]:
        """Handle high-intent leads and collect information
        
        This method implements the lead qualification workflow:
        1. Check if lead already captured
        2. Extract information from messages
        3. Ask for missing information in order
        4. Trigger tool when complete
        """
        
        # Check if lead already captured
        if state.get("lead_captured", False):
            response = """✅ **Lead Already Captured**

Your information has been submitted to our sales team. They will contact you within 24 hours.

Is there anything else I can help you with?"""
            return {"messages": state["messages"] + [response]}
        
        last_message = state["messages"][-1]
        
        # Try to extract information from the message
        # Check for email first (most distinct pattern)
        if not state.get("email"):
            extracted_email = extract_info_from_message(last_message, "email")
            if extracted_email and validate_email(extracted_email):
                state["email"] = extracted_email
        
        # Check for platform
        if not state.get("platform"):
            extracted_platform = extract_info_from_message(last_message, "platform")
            if extracted_platform:
                state["platform"] = extracted_platform
        
        # Check for name (simplistic - user typically just says their name)
        if not state.get("name") and len(last_message.split()) <= 3:
            # Short messages likely contain just name
            potential_name = last_message.strip()
            if len(potential_name) >= 2 and not any(c in potential_name for c in ['@', '.', '?']):
                state["name"] = potential_name.capitalize()
        
        # Determine what to ask next based on what's missing
        if not state.get("name"):
            response = """🎯 **Great choice!**

I see you're interested in AutoStream Pro. Let me get you started.

**What's your name?** (This will be used for your account)"""
        
        elif not state.get("email"):
            response = f"""Thanks **{state['name']}**! 

**What's your email address?** 
We'll send the Pro plan details and account setup instructions there."""
        
        elif not state.get("platform"):
            response = f"""Perfect **{state['name']}**!

**Which platform do you create content for?** 
(YouTube, Instagram, TikTok, etc.)"""
        
        else:
            # All information collected - trigger lead capture
            result = mock_lead_capture(
                state["name"],
                state["email"],
                state["platform"]
            )
            
            if result["success"]:
                response = f"""🎉 **Welcome to AutoStream Pro, {state['name']}!**

✅ Lead ID: `{result['lead_id']}`
📧 Confirmation sent to: {state['email']}
🎬 Platform: {state['platform']}

**What happens next?**
1. Our sales team will contact you within 24 hours
2. You'll receive setup instructions via email
3. Get 7-day free trial on Pro plan

Anything else I can help with?"""
                state["lead_captured"] = True
            else:
                response = f"""❌ **Error**: {result['message']}

Please provide valid information to continue with the Pro plan signup."""
        
        # Return updated state
        return {
            "name": state.get("name"),
            "email": state.get("email"),
            "platform": state.get("platform"),
            "lead_captured": state.get("lead_captured", False),
            "messages": state["messages"] + [response]
        }
    
    def get_response(self, user_input: str, state: AgentState) -> Tuple[str, AgentState]:
        """Get agent response and update state
        
        Args:
            user_input: User's message
            state: Current conversation state
        
        Returns:
            Tuple of (response_text, updated_state)
        """
        # Update state with new message
        new_state = state.copy()
        messages = new_state.get("messages", [])
        messages.append(user_input)
        new_state["messages"] = messages
        
        # Run the graph to get response
        result = self.graph.invoke(new_state)
        
        # Get the last response
        last_response = result["messages"][-1] if result["messages"] else "I'm sorry, I didn't understand that."
        
        return last_response, result


# Create a global agent instance
agent = AutoStreamAgent()


def agent_step(state: AgentState) -> str:
    """Wrapper function for compatibility with main.py"""
    if not state.get("messages"):
        return "Hello! How can I help you today?"
    
    last_message = state["messages"][-1]
    response, _ = agent.get_response(last_message, state)
    return response