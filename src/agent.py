import os
import re
from typing import Dict, Any, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from state import AgentState
from rag import retrieve_info
from intent import detect_intent
from tools import mock_lead_capture


class AutoStreamAgent:
    """Conversational AI Agent for AutoStream using LangGraph"""
    
    def __init__(self):
        # Initialize LLM with Gemini 2.5 Flash (Relacing Deprecated 1.5 Flash)
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
    
    def _is_question(self, text: str) -> bool:
        """Check if the message is a question"""
        question_indicators = ['?', 'what', 'how', 'is there', 'do you', 'can i', 'could i', 
                               'would you', 'should i', 'which', 'where', 'when', 'why']
        text_lower = text.lower()
        return '?' in text or any(text_lower.startswith(word) for word in question_indicators)
    
    def _extract_plan_from_message(self, text: str) -> Optional[str]:
        """
        Extract plan information from user message
        Only extracts if it's a statement of intent, not a question
        """
        text_lower = text.lower()
        
        # Don't extract from questions
        if self._is_question(text):
            return None
        
        # Don't extract if it's asking for information
        info_patterns = ['give me', 'show me', 'list', 'what are', 'tell me about', 'what is']
        if any(pattern in text_lower for pattern in info_patterns):
            # Check if it's a purchase intent disguised as command
            purchase_words = ['want', 'need', 'buy', 'subscribe', 'sign']
            if not any(word in text_lower for word in purchase_words):
                return None
        
        # Strong purchase intent indicators
        strong_purchase = [
            'i want', 'i need', 'i would like', 'i\'d like',
            'sign me up', 'subscribe me', 'i\'ll take', 'i will take',
            'get me', 'please add'
        ]
        
        # Check for strong purchase intent
        has_strong_intent = any(phrase in text_lower for phrase in strong_purchase)
        
        # Check for plan mention
        has_pro = 'pro' in text_lower
        has_basic = 'basic' in text_lower
        
        # Extract plan only if there's strong purchase intent
        if has_strong_intent:
            if has_pro:
                return 'Pro'
            elif has_basic:
                return 'Basic'
        
        # Check for standalone plan requests without strong intent
        if not has_strong_intent:
            # If it's just "pro plan" without "want/need", it's likely an inquiry
            return None
        
        return None
    
    def _extract_platform_from_message(self, text: str) -> Optional[str]:
        """
        Extract platform information from user message
        Only extracts if it's clearly stated as their platform, not a question
        """
        text_lower = text.lower()
        
        # Don't extract from questions
        if self._is_question(text):
            return None
        
        # Platform mapping
        platforms = {
            'youtube': 'YouTube',
            'instagram': 'Instagram',
            'tiktok': 'TikTok',
            'facebook': 'Facebook',
            'twitter': 'Twitter',
            'linkedin': 'LinkedIn',
            'twitch': 'Twitch',
            'snapchat': 'Snapchat'
        }
        
        # Look for "my [platform]" pattern (strong indicator)
        for key, value in platforms.items():
            if f'my {key}' in text_lower:
                return value
        
        # Look for "for my [platform]" pattern
        for key, value in platforms.items():
            if f'for my {key}' in text_lower:
                return value
        
        # If purchase intent is clear, check for platform mentions
        purchase_indicators = ['want', 'need', 'get', 'take', 'buy', 'subscribe', 
                               'sign up', 'purchase', 'interested in', 'would like']
        is_purchase_intent = any(indicator in text_lower for indicator in purchase_indicators)
        
        if is_purchase_intent:
            for key, value in platforms.items():
                if key in text_lower:
                    return value
        
        return None
    
    def process_intent(self, state: AgentState) -> Dict[str, Any]:
        """Process and classify user intent"""
        # If we're in the middle of lead collection, stay in high_intent
        if state.get("waiting_for") or state.get("name") or state.get("email") or state.get("platform") or state.get("selected_plan"):
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
        if state.get("waiting_for") or state.get("name") or state.get("email") or state.get("platform") or state.get("selected_plan"):
            if not state.get("lead_captured", False):
                return "high_intent"
        
        return state.get("intent", "inquiry")
    
    def handle_greeting(self, state: AgentState) -> Dict[str, Any]:
        """Handle greeting intent with friendly welcome"""
        response = """👋 **Welcome to AutoStream Support!**

I'm your AI assistant for AutoStream, the automated video editing platform.

I can help you with:
• **Pricing & Plans** - Basic ($29) and Pro ($79) options
• **Features** - Video editing, AI captions, 4K resolution
• **Policies** - Refunds, support availability
• **Signing Up** - Get started with a plan

What would you like to know about AutoStream?"""
        
        new_messages = state.get("messages", []) + [response]
        return {"messages": new_messages}
    
    def handle_inquiry(self, state: AgentState) -> Dict[str, Any]:
        """Handle inquiry intent with RAG retrieval"""
        messages = state.get("messages", [])
        last_question = messages[-1] if messages else ""
        
        info = retrieve_info(last_question)
        
        response = f"""**Information about AutoStream**

{info}

Is there anything specific you'd like to know more about? I can help with pricing details, feature comparisons, or signing up!"""
        
        new_messages = messages + [response]
        return {"messages": new_messages}
    
    def handle_high_intent(self, state: AgentState) -> Dict[str, Any]:
        """Handle high-intent leads and collect information"""
        
        messages = state.get("messages", [])
        
        # Check if lead already captured
        if state.get("lead_captured", False):
            response = """**Lead Already Captured**

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
        
        # SMART EXTRACTION: Try to extract plan and platform from the initial message
        if not waiting_for and not updated_state.get("selected_plan"):
            # This is the first high-intent message
            extracted_plan = self._extract_plan_from_message(last_message)
            extracted_platform = self._extract_platform_from_message(last_message)
            
            if extracted_plan:
                updated_state["selected_plan"] = extracted_plan
                print(f"[DEBUG] Smart extraction - Plan: {extracted_plan}")
            
            if extracted_platform:
                updated_state["platform"] = extracted_platform
                print(f"[DEBUG] Smart extraction - Platform: {extracted_platform}")
            
            # Determine next step based on what we extracted
            if updated_state.get("selected_plan") and updated_state.get("platform"):
                # Both plan and platform extracted, ask for name and email
                updated_state["waiting_for"] = "name"
                print(f"[DEBUG] Both plan and platform extracted - moving to name")
            elif updated_state.get("selected_plan"):
                # Only plan extracted, ask for name (will ask platform later)
                updated_state["waiting_for"] = "name"
                print(f"[DEBUG] Plan extracted - moving to name")
            else:
                # Nothing extracted, ask for plan selection
                updated_state["waiting_for"] = "plan"
                print(f"[DEBUG] Nothing extracted - asking for plan selection")
        
        # Step 1: Ask for plan selection if needed
        elif waiting_for == "plan":
            # Check if user selected a plan
            last_message_lower = last_message.lower()
            if "pro" in last_message_lower or "79" in last_message_lower:
                updated_state["selected_plan"] = "Pro"
                updated_state["waiting_for"] = "name"
                print(f"[DEBUG] Plan selected: {updated_state['selected_plan']}")
            elif "basic" in last_message_lower or "29" in last_message_lower:
                updated_state["selected_plan"] = "Basic"
                updated_state["waiting_for"] = "name"
                print(f"[DEBUG] Plan selected: {updated_state['selected_plan']}")
            else:
                # Invalid selection, ask again
                response = """Please select either:
• **Pro** plan ($79/month) - Unlimited videos, 4K, AI captions
• **Basic** plan ($29/month) - 10 videos/month, 720p

Which plan would you like to sign up for?"""
                new_messages = messages + [response]
                return {
                    "messages": new_messages,
                    "intent": "high_intent",
                    "selected_plan": updated_state.get("selected_plan"),
                    "name": updated_state.get("name"),
                    "email": updated_state.get("email"),
                    "platform": updated_state.get("platform"),
                    "lead_captured": updated_state.get("lead_captured", False),
                    "waiting_for": updated_state.get("waiting_for"),
                    "conversation_history": updated_state.get("conversation_history", [])
                }
        
        # Step 2: Capture name
        elif waiting_for == "name":
            potential_name = last_message.strip()
            if len(potential_name.split()) <= 3 and len(potential_name) <= 50:
                if not any(keyword in potential_name.lower() for keyword in ["want", "subscribe", "pro", "basic", "plan", "sign", "up", "yes", "no"]):
                    updated_state["name"] = potential_name.title()
                    # After name, decide what to ask next
                    if updated_state.get("platform"):
                        # Platform already extracted, ask for email
                        updated_state["waiting_for"] = "email"
                    else:
                        # Ask for platform next
                        updated_state["waiting_for"] = "platform"
                    print(f"[DEBUG] Name captured: {updated_state['name']}")
        
        # Step 3: Capture platform (if not already extracted)
        elif waiting_for == "platform":
            # First check if platform was mentioned in this message
            extracted_platform = self._extract_platform_from_message(last_message)
            
            if extracted_platform:
                updated_state["platform"] = extracted_platform
                updated_state["waiting_for"] = "email"
                print(f"[DEBUG] Platform captured: {updated_state['platform']}")
            else:
                # Try to extract from the message using platform keywords
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
                    updated_state["waiting_for"] = "email"
                    print(f"[DEBUG] Platform captured: {updated_state['platform']}")
                else:
                    # No platform found, ask again
                    response = """**Which platform do you create content for?** 
(YouTube, Instagram, TikTok, Facebook, etc.)

This helps us tailor your plan experience."""
                    new_messages = messages + [response]
                    return {
                        "messages": new_messages,
                        "intent": "high_intent",
                        "selected_plan": updated_state.get("selected_plan"),
                        "name": updated_state.get("name"),
                        "email": updated_state.get("email"),
                        "platform": updated_state.get("platform"),
                        "lead_captured": updated_state.get("lead_captured", False),
                        "waiting_for": updated_state.get("waiting_for"),
                        "conversation_history": updated_state.get("conversation_history", [])
                    }
        
        # Step 4: Capture email
        elif waiting_for == "email":
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            email_match = re.search(email_pattern, last_message)
            if email_match:
                updated_state["email"] = email_match.group()
                print(f"[DEBUG] Email captured: {updated_state['email']}")
        
        # Generate appropriate response based on what we're waiting for
        if updated_state.get("waiting_for") == "plan" and not updated_state.get("selected_plan"):
            response = """**Great! Let's get you started with AutoStream.**

Which plan would you like to sign up for?

**Pro Plan** - $79/month
• Unlimited videos
• 4K resolution  
• AI captions
• 24/7 support

**Basic Plan** - $29/month
• 10 videos/month
• 720p resolution
• Email support

Please type **Pro** or **Basic** to continue."""
        
        elif updated_state.get("waiting_for") == "name" and not updated_state.get("name"):
            plan = updated_state.get("selected_plan", "Pro")
            response = f"""Excellent choice! The {plan} plan is perfect for your needs.

**What's your name?** (This will be used for your account)"""
        
        elif updated_state.get("waiting_for") == "platform" and not updated_state.get("platform"):
            name = updated_state.get("name", "there")
            plan = updated_state.get("selected_plan", "Pro")
            response = f"""Thanks **{name}**!

**Which platform do you create content for?** 
(YouTube, Instagram, TikTok, Facebook, etc.)

This helps us tailor your {plan} plan experience."""
        
        elif updated_state.get("waiting_for") == "email" and not updated_state.get("email"):
            name = updated_state.get("name", "there")
            plan = updated_state.get("selected_plan", "Pro")
            response = f"""Perfect **{name}**! 

**What's your email address?** 
We'll send the {plan} plan details and account setup instructions there."""
        
        elif updated_state.get("selected_plan") and updated_state.get("name") and updated_state.get("email") and updated_state.get("platform"):
            # All information collected - trigger lead capture
            result = mock_lead_capture(
                updated_state["name"],
                updated_state["email"],
                updated_state["platform"],
                updated_state["selected_plan"]
            )
            
            if result["success"]:
                response = f"""**Welcome to AutoStream {updated_state['selected_plan']} Plan, {updated_state['name']}!**

Lead ID: `{result['lead_id']}`
Confirmation sent to: {updated_state['email']}
Platform: {updated_state['platform']}
Plan: {updated_state['selected_plan']}

**What happens next?**
1. Our sales team will contact you within 24 hours
2. You'll receive setup instructions via email
3. Get 7-day free trial on the {updated_state['selected_plan']} plan

Anything else I can help with?"""
                updated_state["lead_captured"] = True
                updated_state["waiting_for"] = None
            else:
                response = f"""**Error**: {result['message']}

Please provide valid information to continue with the signup."""
        else:
            # Fallback - restart the process
            response = "Let me start over. Which plan would you like to sign up for? Pro or Basic?"
            updated_state["waiting_for"] = "plan"
            updated_state["selected_plan"] = None
            updated_state["name"] = None
            updated_state["email"] = None
            updated_state["platform"] = None
        
        new_messages = messages + [response]
        
        # Return updated state
        return {
            "messages": new_messages,
            "intent": "high_intent",
            "selected_plan": updated_state.get("selected_plan"),
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