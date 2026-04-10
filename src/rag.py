import json
import os
from typing import Dict, List, Optional

class KnowledgeBase:
    """RAG-powered knowledge retrieval from local JSON
    
    This class handles:
    - Loading knowledge base from JSON file
    - Retrieving relevant information based on user queries
    - Formatting responses for the agent
    """
    
    def __init__(self, file_path: str = None):
        if file_path is None:
            # Get the absolute path to data folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            file_path = os.path.join(project_root, "data", "knowledge_base.json")
        
        self.file_path = file_path
        self.data = self._load_knowledge()
    
    def _load_knowledge(self) -> Dict:
        """Load knowledge base from JSON file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Knowledge base not found at {self.file_path}")
            return self._get_default_knowledge()
    
    def _get_default_knowledge(self) -> Dict:
        """Return default knowledge base if file not found"""
        return {
            "pricing": {
                "basic": {"price": "$29/month", "features": ["10 videos/month", "720p resolution"]},
                "pro": {"price": "$79/month", "features": ["Unlimited videos", "4K resolution", "AI captions"]}
            },
            "policies": {
                "refund": "No refunds after 7 days",
                "support": "24/7 support only for Pro plan"
            }
        }
    
    def retrieve(self, query: str) -> str:
        """Retrieve relevant information based on user query"""
        query_lower = query.lower()
        
        # Pricing related queries
        if any(word in query_lower for word in ["price", "cost", "plan", "pricing", "how much", "fee", "monthly"]):
            return self._format_pricing()
        
        # Refund related queries
        elif any(word in query_lower for word in ["refund", "return", "cancel", "cancellation", "money back"]):
            return f"💰 Refund Policy: {self.data['policies']['refund']}"
        
        # Support related queries
        elif any(word in query_lower for word in ["support", "help", "24/7", "customer service", "contact"]):
            return f"🎧 Support: {self.data['policies']['support']}"
        
        # Features related queries
        elif any(word in query_lower for word in ["feature", "capability", "can it", "resolution", "caption"]):
            return self._format_features()
        
        # About AutoStream
        elif any(word in query_lower for word in ["what is", "about", "autostream"]):
            return self.data.get("faq", {}).get("what_is_autostream", "AutoStream is an AI-powered automated video editing platform.")
        
        # Default response
        return self._format_pricing()
    
    def _format_pricing(self) -> str:
        """Format pricing information in a user-friendly way"""
        pricing = self.data["pricing"]
        return f"""
📊 **AutoStream Pricing Plans**

**Basic Plan**: {pricing['basic']['price']}
  ✓ {pricing['basic']['features'][0]}
  ✓ {pricing['basic']['features'][1]}

**Pro Plan**: {pricing['pro']['price']}
  ✓ {pricing['pro']['features'][0]}
  ✓ {pricing['pro']['features'][1]}
  ✓ {pricing['pro']['features'][2]}

**Policies**:
  • {self.data['policies']['refund']}
  • {self.data['policies']['support']}
"""
    
    def _format_features(self) -> str:
        """Format features information"""
        return f"""
✨ **AutoStream Key Features**

**Basic Plan**:
- 10 videos per month
- 720p HD resolution
- Basic editing tools

**Pro Plan**:
- Unlimited videos
- 4K Ultra HD resolution
- AI-powered captions
- Advanced editing suite
- Priority rendering
"""
    
    def get_all_info(self) -> str:
        """Get complete knowledge base information"""
        return self._format_pricing()


# Create a global instance
knowledge_base = KnowledgeBase()

def retrieve_info(query: str) -> str:
    """Convenience function for RAG retrieval"""
    return knowledge_base.retrieve(query)