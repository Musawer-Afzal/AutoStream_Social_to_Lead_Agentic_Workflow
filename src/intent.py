import re
from typing import Optional

class IntentClassifier:
    """Intent classification for user messages"""
    
    # More comprehensive keyword lists
    GREETING_KEYWORDS = [
        "hi", "hello", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "sup", "howdy",
        "what's up", "yo", "hiya", "how are you"
    ]
    
    # Expanded high-intent keywords
    HIGH_INTENT_KEYWORDS = [
        "buy", "subscribe", "sign up", "signup", "sign-up", "start", "try", 
        "want", "interested", "purchase", "get pro", "i'll take", 
        "give me", "how to join", "enroll", "register",
        "upgrade", "switch to pro", "need pro", "want pro",
        "sign me up", "i want", "i need", "take pro",
        "signing up", "signing-up", "joining", "become a member",
        "create account", "make account", "start pro", "get started"
    ]
    
    INQUIRY_KEYWORDS = [
        "price", "cost", "plan", "pricing", "feature", "how much",
        "what is", "tell me about", "explain", "difference",
        "compare", "refund", "support", "help", "policy",
        "policies", "details", "information", "know about"
    ]
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def classify(self, user_input: str, llm=None) -> str:
        """Classify user intent using rule-based (saves API tokens)"""
        return self._rule_based_classify(user_input)
    
    def _rule_based_classify(self, text: str) -> str:
        """Enhanced rule-based intent detection"""
        text_lower = text.lower().strip()
        
        # Check for high intent first (most important for lead gen)
        for keyword in self.HIGH_INTENT_KEYWORDS:
            if keyword in text_lower:
                return "high_intent"
        
        # Check for greeting
        for keyword in self.GREETING_KEYWORDS:
            if keyword in text_lower:
                return "greeting"
        
        # Default to inquiry (covers pricing questions, feature questions, etc.)
        return "inquiry"
    
    def is_lead_qualification_complete(self, state) -> bool:
        """Check if all lead information has been collected"""
        return all([
            state.get("name"),
            state.get("email"),
            state.get("platform")
        ])


# Create global instance
intent_classifier = IntentClassifier()

def detect_intent(user_input: str, llm=None) -> str:
    """Convenience function for intent detection"""
    return intent_classifier.classify(user_input, llm)