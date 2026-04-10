import re
from typing import Optional

class IntentClassifier:
    """Intent classification for user messages
    
    Classifies messages into:
    - greeting: Casual conversation starters
    - inquiry: Questions about product, pricing, features
    - high_intent: Strong purchase or signup signals
    """
    
    # Keywords for different intent types
    GREETING_KEYWORDS = [
        "hi", "hello", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "sup", "howdy"
    ]
    
    HIGH_INTENT_KEYWORDS = [
        "buy", "subscribe", "sign up", "start", "try", "want", 
        "interested", "purchase", "get pro", "i'll take", 
        "give me", "how to join", "enroll", "register",
        "upgrade", "switch to pro", "need pro", "want pro"
    ]
    
    INQUIRY_KEYWORDS = [
        "price", "cost", "plan", "pricing", "feature", "how much",
        "what is", "tell me about", "explain", "difference",
        "compare", "refund", "support", "help"
    ]
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def classify(self, user_input: str, llm=None) -> str:
        """
        Classify user intent using LLM first, fallback to rules
        
        Returns:
            str: 'greeting', 'inquiry', or 'high_intent'
        """
        # Try LLM classification if available
        if llm:
            try:
                intent = self._llm_classify(user_input, llm)
                if intent in ["greeting", "inquiry", "high_intent"]:
                    return intent
            except Exception as e:
                print(f"LLM classification failed: {e}")
        
        # Fallback to rule-based
        return self._rule_based_classify(user_input)
    
    def _llm_classify(self, user_input: str, llm) -> str:
        """Use LLM for intent classification"""
        prompt = f"""Classify the user's intent into exactly one of these categories:
- greeting: Casual hello, hi, hey, etc.
- inquiry: Asking about pricing, features, policies, or general questions
- high_intent: Expressing desire to buy, subscribe, sign up, or try the product

User message: "{user_input}"

Respond with ONLY the category name (greeting/inquiry/high_intent):"""
        
        response = llm.invoke(prompt)
        return response.content.strip().lower()
    
    def _rule_based_classify(self, text: str) -> str:
        """Rule-based intent detection as fallback"""
        text_lower = text.lower()
        
        # Check for high intent first (most important)
        for keyword in self.HIGH_INTENT_KEYWORDS:
            if keyword in text_lower:
                return "high_intent"
        
        # Check for greeting
        for keyword in self.GREETING_KEYWORDS:
            if keyword in text_lower:
                return "greeting"
        
        # Default to inquiry
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