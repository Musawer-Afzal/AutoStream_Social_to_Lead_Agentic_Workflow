import re
from typing import Tuple, Optional, Dict, Any

def validate_email(email: str) -> bool:
    """Validate email format using regex pattern"""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None

def validate_name(name: str) -> bool:
    """Validate name is not empty and has reasonable length"""
    if not name or not isinstance(name, str):
        return False
    
    name_clean = name.strip()
    return len(name_clean) >= 2 and len(name_clean) <= 100

def validate_platform(platform: str) -> bool:
    """Validate platform is provided and recognized"""
    if not platform or not isinstance(platform, str):
        return False
    
    valid_platforms = [
        "youtube", "instagram", "tiktok", "facebook", 
        "twitter", "linkedin", "snapchat", "twitch",
        "discord", "podcast", "blog", "website"
    ]
    
    platform_clean = platform.lower().strip()
    return platform_clean in valid_platforms or len(platform_clean) >= 2

def mock_lead_capture(name: str, email: str, platform: str) -> Dict[str, Any]:
    """
    Mock API function for lead capture
    
    In production, this would make an HTTP request to your CRM,
    database, or lead management system.
    
    Args:
        name: User's full name
        email: User's email address
        platform: Content creation platform
    
    Returns:
        dict: Response with status, message, and lead ID
    """
    # Validate all inputs before "saving"
    if not validate_name(name):
        return {
            "success": False,
            "message": "Invalid name provided. Please use 2-100 characters.",
            "lead_id": None
        }
    
    if not validate_email(email):
        return {
            "success": False,
            "message": "Invalid email format. Please provide a valid email address.",
            "lead_id": None
        }
    
    if not validate_platform(platform):
        return {
            "success": False,
            "message": "Please specify a valid content platform (YouTube, Instagram, TikTok, etc.)",
            "lead_id": None
        }
    
    # Simulate API call
    print(f"\n{'='*60}")
    print(f"📞 LEAD CAPTURE API - MOCK CALL")
    print(f"{'='*60}")
    print(f"✅ Name: {name.strip()}")
    print(f"✅ Email: {email.strip().lower()}")
    print(f"✅ Platform: {platform.strip().capitalize()}")
    print(f"⏰ Timestamp: {__import__('datetime').datetime.now()}")
    print(f"{'='*60}\n")
    
    # Generate a mock lead ID
    import hashlib
    lead_hash = hashlib.md5(f"{name}{email}{platform}".encode()).hexdigest()[:8]
    
    return {
        "success": True,
        "message": f"Lead captured successfully: {name}, {email}, {platform}",
        "lead_id": f"LD-{lead_hash.upper()}"
    }

def extract_info_from_message(message: str, info_type: str) -> Optional[str]:
    """
    Extract specific information from user message
    
    Args:
        message: User's message text
        info_type: Type of info to extract ('email', 'platform', or 'name')
    
    Returns:
        Extracted value or None if not found
    """
    message_lower = message.lower()
    
    if info_type == "email":
        # Extract email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, message)
        if match:
            return match.group()
    
    elif info_type == "platform":
        # Common content platforms
        platforms = {
            "youtube": "YouTube",
            "instagram": "Instagram", 
            "tiktok": "TikTok",
            "facebook": "Facebook",
            "twitter": "Twitter",
            "linkedin": "LinkedIn",
            "twitch": "Twitch",
            "snapchat": "Snapchat"
        }
        
        for key, value in platforms.items():
            if key in message_lower:
                return value
    
    elif info_type == "name":
        # Simple name extraction - look for "my name is X" or "I'm X"
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"this is (\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(1)
                return name.capitalize()
    
    return None