#!/usr/bin/env python3
"""
AutoStream Conversational AI Agent
Main entry point for the application

This agent converts social media conversations into qualified leads
using LangGraph for state management and Gemini Flash for intelligence.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.state import create_initial_state
from src.agent import agent


def print_banner():
    """Print welcome banner with instructions"""
    print("\n" + "="*70)
    print("🎬 " * 10)
    print("="*70)
    print("         AUTOSTREAM AI AGENT - Conversational Lead Generation")
    print("="*70)
    print("🎬 " * 10)
    print("\n📋 **Commands:**")
    print("   • Type your message and press Enter")
    print("   • Type 'quit' or 'exit' to end the conversation")
    print("   • Type 'reset' to start a new conversation")
    print("   • Type 'debug' to toggle debug mode")
    print("\n" + "-"*70)
    print("💡 **Try asking:**")
    print("   • 'What are your pricing plans?'")
    print("   • 'Tell me about Pro plan features'")
    print("   • 'I want to subscribe to Pro'")
    print("-"*70 + "\n")


def print_debug_state(state, debug_mode):
    """Print debug information about current state"""
    if debug_mode:
        print("\n" + "🔍"*35)
        print("DEBUG MODE - Current State:")
        print(f"   Intent: {state.get('intent', 'N/A')}")
        print(f"   Name: {state.get('name', 'Not set')}")
        print(f"   Email: {state.get('email', 'Not set')}")
        print(f"   Platform: {state.get('platform', 'Not set')}")
        print(f"   Lead Captured: {state.get('lead_captured', False)}")
        print(f"   Messages Count: {len(state.get('messages', []))}")
        print("🔍"*35 + "\n")


def main():
    """Main conversation loop"""
    state = create_initial_state()
    debug_mode = '--debug' in sys.argv
    
    print_banner()
    
    conversation_active = True
    while conversation_active:
        try:
            # Get user input
            user_input = input("👤 **You:** ").strip()
            
            # Check for empty input
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n🤖 **Agent:** Thank you for chatting with AutoStream! Have a great day! 👋\n")
                conversation_active = False
                break
            
            if user_input.lower() == 'reset':
                state = create_initial_state()
                print("\n🤖 **Agent:** Conversation reset! How can I help you today?\n")
                continue
            
            if user_input.lower() == 'debug':
                debug_mode = not debug_mode
                status = "ON" if debug_mode else "OFF"
                print(f"\n🔍 Debug mode turned {status}\n")
                continue
            
            # ADD THIS NEW COMMAND FOR RATE LIMIT INFO
            if user_input.lower() == 'stats' or user_input.lower() == 'usage':
                from src.agent import get_api_stats
                stats = get_api_stats()
                print(f"**API Usage Statistics:**")
                print(f"   • Total requests today: {stats['total_requests']}")
                print(f"   • Blocked requests: {stats['blocked_requests']}")
                print(f"   • Remaining this minute: {stats['remaining_this_minute']}")
                print(f"   • Remaining today: {stats['remaining_today']}")
                print(f"   • Minute reset in: {stats['minute_reset_in']:.1f}s")
                print(f"   • Day reset in: {stats['day_reset_in']/3600:.1f}h\n")
                continue
            
            # Get agent response
            response, state = agent.get_response(user_input, state)
            print(f"\n🤖 **Agent:** {response}\n")
            
            # Print debug info if enabled
            if debug_mode:
                print_debug_state(state, debug_mode)
                
        except KeyboardInterrupt:
            print("\n\n🤖 **Agent:** Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ **Error:** {str(e)}")
            print("Please try again or type 'reset' to restart the conversation.\n")


if __name__ == "__main__":
    main()