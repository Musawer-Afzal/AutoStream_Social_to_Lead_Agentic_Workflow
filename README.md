# 🎬 AutoStream AI Agent - Conversational Lead Generation

An intelligent conversational AI agent built with LangGraph and Google Gemini 2.5 Flash that converts social media conversations into qualified business leads for AutoStream, a SaaS video editing platform.

## 📋 Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Architecture Explanation](#architecture-explanation)
- [WhatsApp Deployment](#whatsapp-deployment)
- [Demo Flow](#demo-flow)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Intent Detection**: Classifies user messages into greetings, inquiries, or high-intent leads
- **RAG-Powered Knowledge Base**: Retrieves pricing, features, and policy information from local JSON
- **Smart Lead Capture**: Extracts plan preferences and platform information from natural language
- **State Management**: Maintains conversation context across multiple turns using LangGraph
- **Tool Execution**: Triggers mock lead capture API only after collecting all required information
- **Rate Limiting**: Built-in protection for free tier API limits

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| LLM | Google Gemini 2.5 Flash (Free Tier) |
| Framework | LangChain + LangGraph |
| State Management | LangGraph State (TypedDict) |
| RAG | Local JSON Knowledge Base |
| API Rate Limiting | Token Bucket + Sliding Window |

## 📁 Project Structure
```
autostream-agent/
│
├── main.py # Entry point - CLI interface
├── requirements.txt # Python dependencies
├── .env # Environment variables (API keys)
├── README.md # Documentation
│
├── data/
│ └── knowledge_base.json # RAG knowledge source
│
└── src/
├── init.py
├── agent.py # Main agent logic with LangGraph
├── state.py # State management (TypedDict)
├── rag.py # RAG retrieval from knowledge base
├── intent.py # Intent classification (rule-based)
├── tools.py # Lead capture mock API & validation
└── rate_limiter.py # API rate limiting protection
```

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- Google Gemini API key (free tier available)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Musawer-Afzal/AutoStream_Social_to_Lead_Agentic_Workflow
cd autostream-agent
```

### Step 2: Create Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Set Up API Key
1. Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a .env file in the project root:
    GOOGLE_API_KEY=your_api_key_here

### Step 5: Run the Agent
python main.py

### Step 6: Test the Conversation
Try this sample conversation:
You: Hi
Agent: [Welcome message]

You: What are your pricing plans?
Agent: [Shows pricing from RAG]

You: I want to get the Basic plan for my YouTube channel
Agent: [Smart extracts plan & platform] What's your name?

You: John Doe
Agent: What's your email?

You: john@example.com
Agent: [Lead captured successfully]

**Available Commands**
**Command	Action**
quit / exit | End conversation
reset | Start a new conversation
debug | Toggle debug mode (shows state)
stats | Show API usage statistics

### Architecture Explanation
## Why LangGraph?
I chose **LangGraph** over AutoGen for several key reasons:

1. **Explicit State Management**: LangGraph's TypedDict-based state provides type safety and clear visibility into what data persists across conversation turns. This is crucial for lead capture where we need to track partial information (name collected, email pending, etc.).

2. **Cyclic Workflow Support**: Unlike linear chains, LangGraph naturally supports the lead qualification loop - after detecting high intent, the agent must cycle through asking for name, email, and platform, potentially returning to previous states if validation fails.

3. **Conditional Routing**: LangGraph's conditional edges allow dynamic routing based on intent and collected data. The agent can seamlessly transition between greeting, inquiry, and lead collection modes.

4. **Simplicity for This Use Case**: While AutoGen is powerful for multi-agent systems, this single-agent conversational workflow with clear state transitions is more elegantly expressed in LangGraph's graph-based paradigm.

## How State Is Managed
State management follows a **centralized, typed approach**:
```python
class AgentState(TypedDict):
    messages: List[str]           # Conversation history
    intent: str                   # Current intent (greeting/inquiry/high_intent)
    selected_plan: Optional[str]  # 'Basic' or 'Pro'
    name: Optional[str]           # User's name
    email: Optional[str]          # User's email
    platform: Optional[str]       # Content platform
    lead_captured: bool           # Tool execution flag
    waiting_for: Optional[str]    # Current field being collected
```

**Key Design Decisions**:

**Persistent Across Turns**: State is passed through the graph on every invocation, preserving context across 5-6 conversation turns as required.

**Waiting Field Pattern**: The `waiting_for` field tracks exactly what information we're expecting next (`'plan'`, `'name'`, `'email'`, `'platform'`), preventing confusion when users provide information out of order.

**Smart Extraction**: The agent attempts to extract plan and platform from natural language (e.g., "Basic plan for my YouTube channel") before asking, creating a frictionless experience.

**Tool Execution Guard**: `lead_captured` ensures the mock API is called exactly once, after all three fields are validated.

### WhatsApp Deployment
## How to Integrate This Agent with WhatsApp Using Webhooks
The agent can be deployed as a WhatsApp chatbot using Meta's WhatsApp Business API. Here's the complete architecture and implementation approach:

## Architecture Overview
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐     ┌──────────────┐
│  WhatsApp   │────▶│  Webhook Server  │────▶│   Session   │────▶│  AutoStream  │
│    User     │◀────│   (FastAPI)      │◀────│   Manager   │◀────│    Agent     │
└─────────────┘     └──────────────────┘     └─────────────┘     └──────────────┘
                           │                         │
                           ▼                         ▼
                    ┌─────────────┐           ┌─────────────┐
                    │  Meta APIs  │           │   Redis/    │
                    │  (Callback) │           │  DynamoDB   │
                    └─────────────┘           └─────────────┘

# Implementation Steps
**1. Set Up WhatsApp Business API**
# Prerequisites
- Facebook Business Account
- Verified Business Display Name
- WhatsApp Business App from Meta Developers

**2. Webhook Server (FastAPI Example)**
Create `whatsapp_webhook.py`:
```python
from fastapi import FastAPI, Request, Response
from twilio.twiml.messaging_response import MessagingResponse
import redis
import json
from agent import AutoStreamAgent
from state import create_initial_state

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
agent = AutoStreamAgent()

def get_session(phone_number: str):
    """Retrieve or create session state for user"""
    key = f"session:{phone_number}"
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return create_initial_state()

def save_session(phone_number: str, state):
    """Save session state with 30-minute TTL"""
    key = f"session:{phone_number}"
    redis_client.setex(key, 1800, json.dumps(state))

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    user_message = form_data.get('Body', '')
    user_phone = form_data.get('From', '')
    
    # Get or create session
    session_state = get_session(user_phone)
    
    # Process with agent
    response, new_state = agent.get_response(user_message, session_state)
    
    # Save updated session
    save_session(user_phone, new_state)
    
    # Send response back to WhatsApp
    twiml = MessagingResponse()
    twiml.message(response)
    return Response(content=str(twiml), media_type="application/xml")

@app.get("/webhook")  # For verification
async def verify_webhook(request: Request):
    mode = request.query_params.get('hub.mode')
    token = request.query_params.get('hub.verify_token')
    challenge = request.query_params.get('hub.challenge')
    
    if mode == 'subscribe' and token == 'your_verify_token':
        return Response(content=challenge)
    return Response(status_code=403)
```

**3. Session Management Strategy**
```python
# Session Manager with Redis
class SessionManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=6379,
            decode_responses=True
        )
        self.ttl = 1800  # 30 minutes
    
    def get_or_create(self, user_id: str) -> AgentState:
        key = f"wa_session:{user_id}"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)
        
        # Create new session with initial state
        state = create_initial_state()
        self.save(user_id, state)
        return state
    
    def save(self, user_id: str, state: AgentState):
        key = f"wa_session:{user_id}"
        self.redis.setex(key, self.ttl, json.dumps(state))
    
    def delete(self, user_id: str):
        key = f"wa_session:{user_id}"
        self.redis.delete(key)
```

**4. Deployment Configuration**
```python
# docker-compose.yml
version: '3.8'
services:
  webhook:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

**5. Ngrok for Local Testing**
`
# Expose local webhook to internet
ngrok http 8000

# Configure in Meta Developer Console
Webhook URL: https://your-ngrok-url.ngrok.io/webhook
Verify Token: your_verify_token
`

**6. Meta Developer Configuration**
1. Go to [Meta for Developers](https://developers.facebook.com/)
2. Select your WhatsApp App
3. Navigate to **WhatsApp > Configuration**
4. Set Webhook URL and Verify Token
5. Subscribe to `messages` and `message_deliveries` events

### Key Considerations for Production
Concern	| Solution
Concurrent Users | Redis for distributed session storage
Rate Limiting | Per-user rate limiter + API quotas
Media Messages | Handle images/videos via media_url endpoint
Error Recovery | Dead Letter Queue for failed messages
Monitoring | Logging with structured JSON + Sentry
Cost Optimization | Cache frequent queries (pricing info)

## Environment Variables for WhatsApp Deployment
```python
# .env.whatsapp
GOOGLE_API_KEY=your_gemini_api_key
WHATSAPP_ACCESS_TOKEN=your_whatsapp_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WEBHOOK_VERIFY_TOKEN=your_custom_verify_token
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Demo Flow
Here's a complete demo conversation showing all features:
👤 User: Hi
🤖 Agent: 👋 Welcome to AutoStream Support! I can help with pricing, features, or signing up.

👤 User: What are your pricing plans?
🤖 Agent: 📊 Basic Plan: $29/month (10 videos, 720p)
              Pro Plan: $79/month (Unlimited, 4K, AI captions)

👤 User: I want the Pro plan for my YouTube channel
🤖 Agent: [Smart extraction - Plan: Pro, Platform: YouTube]
          Excellent choice! What's your name?

👤 User: Sarah Johnson
🤖 Agent: Thanks Sarah! What's your email?

👤 User: sarah@youtube.com
🤖 Agent: Perfect! [Triggers lead capture]
          🎉 Welcome to AutoStream Pro Plan, Sarah!
          Lead ID: LD-ABC123

## Troubleshooting
Issue | Solution
`ModuleNotFoundError` | Run `pip install -r requirements.txt`
API Key Error | Check `.env` file has `GOOGLE_API_KEY=your_key`
Rate Limit Exceeded | Wait 60 seconds or type `stats` to check usage
404 Model Error | Update model name in `agent.py` to `gemini-2.5-flash`

### License
MIT License

### Acknowledgments
* Google Gemini for free tier LLM access
* LangChain/LangGraph community
* Meta WhatsApp Business API

**Built for ServiceHive Inflx Assignment | AutoStream AI Agent v1.0**

This README includes:
1. **How to run locally** - Step-by-step instructions with commands
2. **Architecture explanation** - Why LangGraph was chosen (200+ words) + state management details
3. **WhatsApp deployment** - Complete webhook integration with code examples, architecture diagram, and production considerations