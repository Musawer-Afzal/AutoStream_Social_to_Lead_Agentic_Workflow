import json
import os
from typing import Dict

class KnowledgeBase:
    """RAG-powered knowledge retrieval from local JSON"""
    
    def __init__(self, file_path: str = None):
        if file_path is None:
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
        
        # Feature-specific queries
        if any(word in query_lower for word in ["feature", "capability", "can it", "what can", "functionality"]):
            # Check if asking specifically about Pro
            if "pro" in query_lower:
                return self._format_pro_features()
            # Check if asking specifically about Basic
            elif "basic" in query_lower:
                return self._format_basic_features()
            # General features question
            else:
                return self._format_all_features()
        
        # Pricing specific queries
        elif any(word in query_lower for word in ["price", "cost", "plan", "pricing", "how much", "fee", "monthly"]):
            if "pro" in query_lower and "basic" not in query_lower:
                return self._format_pro_pricing()
            elif "basic" in query_lower and "pro" not in query_lower:
                return self._format_basic_pricing()
            else:
                return self._format_pricing()
        
        # Resolution questions
        elif any(word in query_lower for word in ["resolution", "720p", "4k", "quality", "hd"]):
            if "4k" in query_lower or "pro" in query_lower:
                return "**Video Quality**: Pro plan supports **4K Ultra HD** resolution, while Basic plan supports **720p HD**."
            else:
                return "**Video Quality**: Basic plan: 720p HD | Pro plan: 4K Ultra HD"
        
        # AI Captions questions
        elif any(word in query_lower for word in ["caption", "subtitles", "ai caption", "auto caption"]):
            return "**AI Captions**: Pro plan includes AI-powered automatic caption generation for all videos. Basic plan does not include this feature."
        
        # Video limits
        elif any(word in query_lower for word in ["video limit", "unlimited", "how many video", "videos per month"]):
            return "**Video Limits**: Basic plan: 10 videos/month | Pro plan: Unlimited videos"
        
        # Refund related queries
        elif any(word in query_lower for word in ["refund", "return", "cancel", "cancellation", "money back"]):
            return f"**Refund Policy**: {self.data['policies']['refund']}"
        
        # Support related queries
        elif any(word in query_lower for word in ["support", "help", "24/7", "customer service", "contact"]):
            return f"**Support**: {self.data['policies']['support']}"
        
        # Compare plans
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return self._format_comparison()
        
        # About AutoStream
        elif any(word in query_lower for word in ["what is", "about", "autostream"]):
            return "🤖 **About AutoStream**: AutoStream is an AI-powered automated video editing platform that helps content creators edit videos faster with smart features like AI captions, auto-formatting, and intelligent scene detection."
        
        # Default to full pricing info
        return self._format_pricing()
    
    def _format_basic_pricing(self) -> str:
        """Format Basic plan pricing only"""
        pricing = self.data["pricing"]
        return f"""
**Basic Plan**: {pricing['basic']['price']}
  ✓ {pricing['basic']['features'][0]}
  ✓ {pricing['basic']['features'][1]}
"""
    
    def _format_pro_pricing(self) -> str:
        """Format Pro plan pricing only"""
        pricing = self.data["pricing"]
        return f"""
**Pro Plan**: {pricing['pro']['price']}
  ✓ {pricing['pro']['features'][0]}
  ✓ {pricing['pro']['features'][1]}
  ✓ {pricing['pro']['features'][2]}
"""
    
    def _format_pricing(self) -> str:
        """Format complete pricing information"""
        pricing = self.data["pricing"]
        return f"""
**AutoStream Pricing Plans**

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
    
    def _format_pro_features(self) -> str:
        """Format Pro plan features in detail"""
        return f"""
**AutoStream Pro Plan Features**

**Price**: $79/month

**Core Features**:
• **Unlimited Videos** - No monthly limits
• **4K Resolution** - Ultra HD quality output
• **AI Captions** - Automatic subtitle generation
• **Advanced Editing** - Multi-track editing
• **Priority Rendering** - Faster export times
• **Mobile App Access** - Edit on the go

**Support**: 24/7 priority support included

Perfect for professional content creators who need unlimited videos and premium features!
"""
    
    def _format_basic_features(self) -> str:
        """Format Basic plan features"""
        return f"""
**AutoStream Basic Plan Features**

**Price**: $29/month

**Core Features**:
• **10 Videos/month** - Perfect for beginners
• **720p Resolution** - HD quality
• **Basic Editing** - Trim, cut, merge
• **Web Access Only**

**Support**: Email support within 48 hours

Great for getting started with video editing!
"""
    
    def _format_all_features(self) -> str:
        """Format all features comparison"""
        return f"""
**AutoStream Features Comparison**

**Basic Plan** ($29/month):
• 10 videos per month
• 720p HD resolution
• Basic editing tools
• Email support

**Pro Plan** ($79/month):
• Unlimited videos
• 4K Ultra HD resolution
• AI-powered captions
• Advanced editing suite
• Priority rendering
• 24/7 priority support

Upgrade to Pro for unlimited videos and AI captions!
"""
    
    def _format_comparison(self) -> str:
        """Format plan comparison"""
        return f"""
**AutoStream Plan Comparison**

| Feature | Basic | Pro |
|---------|-------|-----|
| Price | $29/month | $79/month |
| Videos/month | 10 | Unlimited |
| Resolution | 720p | 4K |
| AI Captions | ❌ | ✅ |
| Support | Email (48hr) | 24/7 Priority |
| Advanced Editing | ❌ | ✅ |

**Recommendation**: Choose Pro for professional content creation with unlimited videos and AI captions!
"""
    
    def get_all_info(self) -> str:
        """Get complete knowledge base information"""
        return self._format_pricing()


# Create a global instance
knowledge_base = KnowledgeBase()

def retrieve_info(query: str) -> str:
    """Convenience function for RAG retrieval"""
    return knowledge_base.retrieve(query)