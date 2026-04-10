import os
import time
import re
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


class AutoStreamAgent:
    """Conversational AI Agent for AutoStream using LangGraph"""
    
    def __init__(self):
        # Initialize LLM with Gemini 2.5 Flash (Free Tier)
        self.llm = self._initialize_llm()
        self.graph = self._build_graph()
        self.last_api_call_time = 0
        self.min_api_interval = 12
    
    def _initialize_llm(self):
        """Initialize LLM with Gemini 2.5 Flash (Free Tier Optimized)"""
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_output_tokens=500,
            top_p=0.95,
            top_k=40,
        )
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("process_intent", self.process_intent)
        workflow.add_node("handle_greeting", self.handle_greeting)
        workflow.add_node("handle_inquiry", self.handle_inquiry)
        workflow.add_node("handle_high_intent", self.handle_high_intent)
        
        workflow.set_entry_point("process_intent")
        
        workflow.add_conditional_edges(
            "process_intent",
            self.route_intent,
            {
                "greeting": "handle_greeting",
                "inquiry": "handle_inquiry",
                "high_intent": "handle_high_intent"
            }
        )
        
        workflow.add_edge("handle_high_intent", END)
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("handle_inquiry", END)
        
        return workflow.compile()
    
    def process_intent(self, state: AgentState) -> Dict[str, Any]:
        """Process and classify user intent"""
        # If we're in the middle of lead collection, stay in high_intent
        if state.get("waiting_for") or state.get("name") or state.get("email") or state.get("platform"):
            if not state.get("lead_captured", False):
                return {"intent": "high_intent"}
        
        if not state.get("messages"):
            return {"intent": "greeting"}
        
        last_message = state["messages"][-1]
        intent = detect_intent(last_message)
        
        print(f"[DEBUG] Intent detected: {intent} for message: {last_message}")
        
        return {"intent": intent}
    
    def route_intent(self, state: AgentState) -> str:
        """Route to appropriate handler based on detected intent"""
        if state.get("waiting_for") or (state.get("name") or state.get("email") or state.get("platform")):
            if not state.get("lead_captured", False):
                return "high_intent"
        
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
        
        new_messages = state.get("messages", []) + [response]
        return {"messages": new_messages}
    
    def handle_inquiry(self, state: AgentState) -> Dict[str, Any]:
        """Handle inquiry intent with RAG retrieval"""
        messages = state.get("messages", [])
        last_question = messages[-1] if messages else ""
        
        info = retrieve_info(last_question)
        
        response = f"""📚 **Information about AutoStream**

{info}

Is there anything specific you'd like to know more about? I can help with pricing details, feature comparisons, or signing up!"""
        
        new_messages = messages + [response]
        return {"messages": new_messages}
    
    def handle_high_intent(self, state: AgentState) -> Dict[str, Any]:
        """Handle high-intent leads and collect information"""
        
        messages = state.get("messages", [])
        
        # Check if lead already captured
        if state.get("lead_captured", False):
            response = """✅ **Lead Already Captured**

Your information has been submitted to our sales team. They will contact you within 24 hours.

Is there anything else I can help you with?"""
            new_messages = messages + [response]
            return {"messages": new_messages}
        
        # Get the last user message
        last_message = messages[-1] if messages else ""
        
        # Create a copy of state to modify
        updated_state = state.copy()
        
        # Track what we're waiting for
        waiting_for = updated_state.get("waiting_for")
        
        # IMPORTANT: Only extract information based on what we're waiting for
        if waiting_for == "name":
            # We asked for name, so this response should be the name
            potential_name = last_message.strip()
            # Validate it's a proper name (not a sentence, not too long)
            if len(potential_name.split()) <= 3 and len(potential_name) <= 50:
                if not any(keyword in potential_name.lower() for keyword in ["want", "subscribe", "pro", "plan", "sign", "up"]):
                    updated_state["name"] = potential_name.title()
                    updated_state["waiting_for"] = "email"  # Next, ask for email
                    print(f"[DEBUG] Name captured: {updated_state['name']}")
        
        elif waiting_for == "email":
            # We asked for email, so this response should be the email
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            email_match = re.search(email_pattern, last_message)
            if email_match:
                updated_state["email"] = email_match.group()
                updated_state["waiting_for"] = "platform"  # Next, ask for platform
                print(f"[DEBUG] Email captured: {updated_state['email']}")
        
        elif waiting_for == "platform":
            # We asked for platform, so this response should be the platform
            platforms = {
                "youtube": "YouTube", "instagram": "Instagram", "tiktok": "TikTok",
                "facebook": "Facebook", "twitter": "Twitter", "linkedin": "LinkedIn",
                "twitch": "Twitch", "snapchat": "Snapchat"
            }
            platform_found = None
            for key, value in platforms.items():
                if key in last_message.lower():
                    platform_found = value
                    break
            
            if platform_found:
                updated_state["platform"] = platform_found
                print(f"[DEBUG] Platform captured: {updated_state['platform']}")
        
        # If this is the first high-intent message (waiting_for is None)
        if not waiting_for and not updated_state.get("name") and not updated_state.get("email") and not updated_state.get("platform"):
            # This is the initial high-intent message - don't capture as name
            updated_state["waiting_for"] = "name"
            print(f"[DEBUG] Starting lead collection - will ask for name")
        
        # Determine what to ask next
        if updated_state.get("waiting_for") == "name" and not updated_state.get("name"):
            response = """🎯 **Great choice!**

I see you're interested in AutoStream Pro. Let me get you started.

**What's your name?** (This will be used for your account)"""
        
        elif updated_state.get("waiting_for") == "email" and not updated_state.get("email"):
            # Name should already be captured at this point
            name = updated_state.get("name", "there")
            response = f"""Thanks **{name}**! 

**What's your email address?** 
We'll send the Pro plan details and account setup instructions there."""
        
        elif updated_state.get("waiting_for") == "platform" and not updated_state.get("platform"):
            name = updated_state.get("name", "there")
            response = f"""Perfect **{name}**!

**Which platform do you create content for?** 
(YouTube, Instagram, TikTok, etc.)"""
        
        elif updated_state.get("name") and updated_state.get("email") and updated_state.get("platform"):
            # All information collected - trigger lead capture
            result = mock_lead_capture(
                updated_state["name"],
                updated_state["email"],
                updated_state["platform"]
            )
            
            if result["success"]:
                response = f"""🎉 **Welcome to AutoStream Pro, {updated_state['name']}!**

✅ Lead ID: `{result['lead_id']}`
📧 Confirmation sent to: {updated_state['email']}
🎬 Platform: {updated_state['platform']}

**What happens next?**
1. Our sales team will contact you within 24 hours
2. You'll receive setup instructions via email
3. Get 7-day free trial on Pro plan

Anything else I can help with?"""
                updated_state["lead_captured"] = True
                updated_state["waiting_for"] = None
            else:
                response = f"""❌ **Error**: {result['message']}

Please provide valid information to continue with the Pro plan signup."""
        else:
            # Fallback - shouldn't reach here
            response = "I'm sorry, let me start over. What's your name?"
            updated_state["waiting_for"] = "name"
        
        new_messages = messages + [response]
        
        # Return updated state
        return {
            "messages": new_messages,
            "intent": "high_intent",
            "name": updated_state.get("name"),
            "email": updated_state.get("email"),
            "platform": updated_state.get("platform"),
            "lead_captured": updated_state.get("lead_captured", False),
            "waiting_for": updated_state.get("waiting_for"),
            "conversation_history": updated_state.get("conversation_history", [])
        }
    
    def get_response(self, user_input: str, state: AgentState) -> Tuple[str, AgentState]:
        """Get agent response and update state"""
        new_state = state.copy()
        messages = new_state.get("messages", [])
        messages.append(user_input)
        new_state["messages"] = messages
        
        result = self.graph.invoke(new_state)
        
        last_response = result["messages"][-1] if result.get("messages") else "I'm sorry, I didn't understand that."
        
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