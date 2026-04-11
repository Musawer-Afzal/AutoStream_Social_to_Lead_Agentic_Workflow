"""
Rate Limiter for Gemini API to manage free tier usage
Free tier limits: 60 requests per minute (typically)
"""

import time
import threading
# from datetime import datetime, timedelta
from collections import deque
from functools import wraps
from typing import Dict, Optional

class TokenBucketRateLimiter:
    """
    Token Bucket algorithm for rate limiting
    
    This allows bursts of requests up to the bucket capacity,
    then refills at a steady rate.
    """
    
    def __init__(self, capacity: int = 10, refill_rate: float = 1.0):
        """
        Args:
            capacity: Maximum number of tokens (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens
        
        Returns:
            bool: True if tokens acquired, False if rate limited
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
    
    def wait_and_acquire(self, tokens: int = 1, timeout: float = 5.0) -> bool:
        """
        Wait until tokens are available or timeout
        
        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time in seconds
        
        Returns:
            bool: True if acquired within timeout, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.acquire(tokens):
                return True
            time.sleep(0.1)
        
        return False


class SlidingWindowRateLimiter:
    """
    Sliding Window algorithm for rate limiting
    More accurate for per-minute limits
    """
    
    def __init__(self, max_requests: int = 50, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()
    
    def can_make_request(self) -> bool:
        """Check if a request can be made"""
        with self.lock:
            now = time.time()
            
            # Remove requests outside the window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
    
    def get_remaining_requests(self) -> int:
        """Get number of requests remaining in current window"""
        with self.lock:
            now = time.time()
            
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            return max(0, self.max_requests - len(self.requests))
    
    def get_reset_time(self) -> float:
        """Get seconds until oldest request expires"""
        with self.lock:
            if not self.requests:
                return 0
            oldest = self.requests[0]
            return max(0, (oldest + self.window_seconds) - time.time())


class APIRateLimiter:
    """
    Main rate limiter for Gemini API calls
    Combines multiple strategies for free tier protection
    """
    
    def __init__(self):
        # Gemini free tier limits (adjust based on actual limits)
        self.per_minute_limit = 60  # 60 requests per minute
        self.per_day_limit = 1500   # 1500 requests per day (typical free tier)
        
        # Initialize limiters
        self.minute_limiter = SlidingWindowRateLimiter(
            max_requests=self.per_minute_limit,
            window_seconds=60
        )
        
        self.day_limiter = SlidingWindowRateLimiter(
            max_requests=self.per_day_limit,
            window_seconds=86400  # 24 hours
        )
        
        # Token bucket for burst control (smaller bursts)
        self.burst_limiter = TokenBucketRateLimiter(
            capacity=5,  # Max 5 requests in quick succession
            refill_rate=0.5  # 1 request every 2 seconds
        )
        
        # Statistics tracking
        self.total_requests = 0
        self.blocked_requests = 0
        self.stats_lock = threading.Lock()
    
    def can_call_api(self) -> tuple[bool, Optional[str]]:
        """
        Check if an API call is allowed
        
        Returns:
            tuple: (allowed, message)
        """
        # Check daily limit
        if not self.day_limiter.can_make_request():
            remaining_time = self.day_limiter.get_reset_time()
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            return False, f"Daily limit reached. Resets in {hours}h {minutes}m"
        
        # Check per-minute limit
        if not self.minute_limiter.can_make_request():
            remaining = self.minute_limiter.get_reset_time()
            return False, f"Rate limit exceeded. Try again in {remaining:.1f} seconds"
        
        # Check burst limit
        if not self.burst_limiter.acquire():
            return False, "Too many requests in quick succession. Please slow down."
        
        # Update stats
        with self.stats_lock:
            self.total_requests += 1
        
        return True, None
    
    def record_blocked(self):
        """Record a blocked request for statistics"""
        with self.stats_lock:
            self.blocked_requests += 1
    
    def get_stats(self) -> Dict:
        """Get current rate limiter statistics"""
        with self.stats_lock:
            return {
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "remaining_this_minute": self.minute_limiter.get_remaining_requests(),
                "remaining_today": self.day_limiter.get_remaining_requests(),
                "minute_reset_in": self.minute_limiter.get_reset_time(),
                "day_reset_in": self.day_limiter.get_reset_time(),
            }
    
    def get_remaining_for_prompt(self) -> str:
        """Get user-friendly remaining quota info"""
        stats = self.get_stats()
        return f"""**API Usage Stats:**
• Remaining this minute: {stats['remaining_this_minute']} requests
• Remaining today: {stats['remaining_today']} requests
• Total used today: {stats['total_requests']} requests
• Blocked requests: {stats['blocked_requests']}"""


# Global instance
rate_limiter = APIRateLimiter()


def rate_limit(func):
    """
    Decorator to apply rate limiting to API calls
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if API call is allowed
        allowed, message = rate_limiter.can_call_api()
        
        if not allowed:
            # Return a fallback response instead of making the API call
            return f"⚠️ {message}\n\nPlease wait a moment before continuing. I'll still try to help using rule-based responses."
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # If API call fails due to rate limiting, record it
            if "429" in str(e) or "rate limit" in str(e).lower():
                rate_limiter.record_blocked()
                raise Exception(f"API rate limit exceeded. {message}")
            raise
    
    return wrapper