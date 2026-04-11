import re
from typing import Optional

class IntentClassifier:
    """Intent classification for user messages"""
    
    # Greeting keywords
    GREETING_KEYWORDS = [
        "hi", "hello", "hey", "greetings", "good morning", 
        "good afternoon", "good evening", "sup", "howdy",
        "what's up", "yo", "hiya", "how are you"
    ]
    
    # HIGH INTENT - Purchase/signup intent patterns
    # These are phrases that clearly indicate the user wants to take action
    HIGH_INTENT_PATTERNS = [
        # Direct purchase intent
        r"i want (a|the)? pro",
        r"i want (a|the)? basic",
        r"i need (a|the)? pro",
        r"i need (a|the)? basic",
        r"i would like (a|the)? pro",
        r"i would like (a|the)? basic",
        r"i'll take (the )?pro",
        r"i'll take (the )?basic",
        r"give me (a|the)? pro",
        r"give me (a|the)? basic",
        
        # Subscribe/signup intent
        r"subscribe to pro",
        r"subscribe to basic",
        r"sign up for pro",
        r"sign up for basic",
        r"signup for pro",
        r"signup for basic",
        
        # Purchase verbs
        r"buy (the )?pro",
        r"buy (the )?basic",
        r"purchase (the )?pro",
        r"purchase (the )?basic",
        
        # Get started with specific plan
        r"get started with pro",
        r"get started with basic",
        r"start (the )?pro",
        r"start (the )?basic",
    ]
    
    # INQUIRY - Asking for information (NOT purchase intent)
    INQUIRY_PATTERNS = [
        # Pricing questions
        r"what .* price",
        r"how much",
        r"pricing",
        r"cost",
        r"plan(s)?",
        
        # Information requests
        r"tell me about",
        r"explain",
        r"difference",
        r"compare",
        r"features?",
        r"refund",
        r"support",
        r"policy|policies",
        r"details",
        r"information",
        
        # Question about availability
        r"do you have",
        r"is there",
        r"are there",
        r"available",
    ]
    
    def __init__(self, llm=None):
        self.llm = llm
        # Compile regex patterns for efficiency
        self.high_intent_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.HIGH_INTENT_PATTERNS]
        self.inquiry_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.INQUIRY_PATTERNS]
    
    def classify(self, user_input: str, llm=None) -> str:
        """Classify user intent using rule-based with regex patterns"""
        return self._rule_based_classify(user_input)
    
    def _rule_based_classify(self, text: str) -> str:
        """Enhanced rule-based intent detection using regex patterns"""
        text_lower = text.lower().strip()
        
        # Check if it's a question
        is_question = self._is_question(text_lower)
        
        # FIRST: Check for high intent patterns (even if it's a question, but usually not)
        for pattern in self.high_intent_regex:
            if pattern.search(text_lower):
                print(f"[DEBUG] High intent pattern matched: {pattern.pattern}")
                return "high_intent"
        
        # SECOND: Check for inquiry patterns
        for pattern in self.inquiry_regex:
            if pattern.search(text_lower):
                return "inquiry"
        
        # THIRD: Check for greeting
        for keyword in self.GREETING_KEYWORDS:
            if keyword in text_lower:
                return "greeting"
        
        # Check for standalone plan mentions without purchase intent
        if "pro plan" in text_lower or "basic plan" in text_lower:
            if not any(word in text_lower for word in ["want", "need", "get", "buy", "subscribe", "sign"]):
                return "inquiry"
        
        # DEFAULT: Inquiry
        return "inquiry"
    
    def _is_question(self, text: str) -> bool:
        """Check if the message is asking a question"""
        # Direct question mark
        if '?' in text:
            return True
        
        # Question starters
        question_starters = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'can you', 'could you', 'would you', 'do you', 'is there',
            'are there'
        ]
        
        for starter in question_starters:
            if text.startswith(starter):
                return True
        
        return False


# Create global instance
intent_classifier = IntentClassifier()

def detect_intent(user_input: str, llm=None) -> str:
    """Convenience function for intent detection"""
    return intent_classifier.classify(user_input, llm)