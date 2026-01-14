"""
Session-Based Caching Service for NSE RAG Application

Provides per-user session caching with:
- TTL (Time-To-Live) expiration
- LRU (Least Recently Used) eviction
- Meeting-based cache invalidation
- Global cache fallback

Each user session gets its own cache namespace for faster repeated queries.
"""

import hashlib
import json
import logging
import time
import threading
from typing import Any, Optional, Dict
from cachetools import TTLCache
import os
from datetime import datetime

# ---------------------------------------------
# Session Cache Logger - writes to logs/session_cache.log
# ---------------------------------------------
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("session_cache")
logger.setLevel(logging.INFO)

# File handler for session cache log
session_log_handler = logging.FileHandler("logs/session_cache.log", encoding="utf-8")
session_log_handler.setLevel(logging.INFO)
session_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.addHandler(session_log_handler)

# Also output to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

# ---------------------------------------------
# Query Logger - writes ONLY queries to logs/queries.log
# ---------------------------------------------
query_logger = logging.getLogger("queries")
query_logger.setLevel(logging.INFO)
query_logger.propagate = False  # Don't send to root logger

query_log_handler = logging.FileHandler("logs/queries.log", encoding="utf-8")
query_log_handler.setLevel(logging.INFO)
query_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
query_logger.addHandler(query_log_handler)

# Configuration
SESSION_CACHE_SIZE = int(os.getenv("SESSION_CACHE_SIZE", "100"))  # Items per session
SESSION_TTL = int(os.getenv("SESSION_TTL", "1800"))  # 30 minutes default
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))  # Max concurrent sessions
SESSION_CLEANUP_INTERVAL = int(os.getenv("SESSION_CLEANUP_INTERVAL", "300"))  # 5 min

# Thread-safe session storage
_session_lock = threading.RLock()
_session_caches: Dict[str, TTLCache] = {}
_session_metadata: Dict[str, Dict[str, Any]] = {}  # Tracks meeting_id, last_access, etc.

# Statistics per session
_session_stats: Dict[str, Dict[str, int]] = {}


def _generate_cache_key(query: str, filters: Optional[Dict]) -> str:
    """Generate a deterministic cache key from query and filters."""
    cache_data = {"query": query, "filters": filters or {}}
    serialized = json.dumps(cache_data, sort_keys=True)
    hash_obj = hashlib.md5(serialized.encode('utf-8'))
    return hash_obj.hexdigest()


def _get_or_create_session(session_id: str, meeting_id: Optional[int] = None) -> TTLCache:
    """
    Get or create a session cache.
    If meeting_id changes, the session cache is cleared.
    """
    with _session_lock:
        current_meeting = None
        if session_id in _session_metadata:
            current_meeting = _session_metadata[session_id].get("meeting_id")
        
        # Check if meeting changed - if so, clear the session cache
        if meeting_id is not None and current_meeting is not None:
            if meeting_id != current_meeting:
                logger.info(f"Meeting changed for session {session_id[:8]}... "
                           f"({current_meeting} ? {meeting_id}), clearing cache")
                if session_id in _session_caches:
                    _session_caches[session_id].clear()
                if session_id in _session_stats:
                    _session_stats[session_id] = {"hits": 0, "misses": 0}
        
        # Create session if doesn't exist
        if session_id not in _session_caches:
            # Evict oldest sessions if at capacity
            if len(_session_caches) >= MAX_SESSIONS:
                _evict_oldest_session()
            
            _session_caches[session_id] = TTLCache(maxsize=SESSION_CACHE_SIZE, ttl=SESSION_TTL)
            _session_stats[session_id] = {"hits": 0, "misses": 0}
            logger.info(f"Created new session cache: {session_id[:8]}...")
        
        # Update metadata
        _session_metadata[session_id] = {
            "meeting_id": meeting_id,
            "last_access": time.time(),
            "created_at": _session_metadata.get(session_id, {}).get("created_at", time.time())
        }
        
        return _session_caches[session_id]


def _evict_oldest_session():
    """Evict the least recently used session."""
    with _session_lock:
        if not _session_metadata:
            return
        
        oldest_session = min(
            _session_metadata.items(),
            key=lambda x: x[1].get("last_access", 0)
        )[0]
        
        _session_caches.pop(oldest_session, None)
        _session_metadata.pop(oldest_session, None)
        _session_stats.pop(oldest_session, None)
        logger.info(f" Evicted oldest session: {oldest_session[:8]}...")


def cache_for_session(
    session_id: str,
    query: str,
    filters: Optional[Dict],
    result: Dict,
    meeting_id: Optional[int] = None
) -> None:
    """
    Cache a query result for a specific session.
    
    Args:
        session_id: User session identifier
        query: Query string
        filters: Optional filters dict
        result: Query result to cache
        meeting_id: Current meeting ID (for cache invalidation)
    """
    try:
        cache = _get_or_create_session(session_id, meeting_id)
        key = _generate_cache_key(query, filters)
        
        cache[key] = {
            "result": result,
            "cached_at": time.time()
        }
        
        # Log query to queries.log (only query, no answer)
        query_logger.info(f"SESSION={session_id[:8]}... | MEETING={meeting_id or 'N/A'} | QUERY={query}")
        
        logger.debug(f" Cached result for session {session_id[:8]}...: {query[:30]}...")
    except Exception as e:
        logger.warning(f"Failed to cache for session: {e}")


def get_from_session(
    session_id: str,
    query: str,
    filters: Optional[Dict],
    meeting_id: Optional[int] = None
) -> Optional[Dict]:
    """
    Get cached result from session cache.
    
    Args:
        session_id: User session identifier
        query: Query string
        filters: Optional filters dict
        meeting_id: Current meeting ID (for cache invalidation)
    
    Returns:
        Cached result or None if not found
    """
    try:
        cache = _get_or_create_session(session_id, meeting_id)
        key = _generate_cache_key(query, filters)
        
        with _session_lock:
            cached = cache.get(key)
            
            if cached:
                _session_stats[session_id]["hits"] += 1
                age = time.time() - cached["cached_at"]
                logger.info(f" Session cache HIT (age: {age:.1f}s) for: {query[:30]}...")
                return cached["result"]
            else:
                _session_stats[session_id]["misses"] += 1
                logger.debug(f" Session cache MISS for: {query[:30]}...")
                return None
                
    except Exception as e:
        logger.warning(f"Failed to get from session cache: {e}")
        return None


def clear_session_cache(session_id: str) -> None:
    """Clear all cached items for a session."""
    with _session_lock:
        if session_id in _session_caches:
            _session_caches[session_id].clear()
            _session_stats[session_id] = {"hits": 0, "misses": 0}
            logger.info(f" Cleared session cache: {session_id[:8]}...")


def get_session_stats(session_id: str) -> Dict[str, Any]:
    """
    Get statistics for a specific session.
    
    Returns:
        Dict with session cache stats
    """
    with _session_lock:
        if session_id not in _session_caches:
            return {"error": "Session not found"}
        
        stats = _session_stats.get(session_id, {"hits": 0, "misses": 0})
        metadata = _session_metadata.get(session_id, {})
        cache = _session_caches[session_id]
        
        total = stats["hits"] + stats["misses"]
        hit_rate = stats["hits"] / total if total > 0 else 0.0
        
        return {
            "session_id": session_id[:8] + "...",
            "cache_size": len(cache),
            "max_size": SESSION_CACHE_SIZE,
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": round(hit_rate, 3),
            "meeting_id": metadata.get("meeting_id"),
            "created_at": metadata.get("created_at"),
            "last_access": metadata.get("last_access"),
            "ttl_seconds": SESSION_TTL
        }


def get_all_sessions_stats() -> Dict[str, Any]:
    """Get statistics for all active sessions."""
    with _session_lock:
        total_items = sum(len(c) for c in _session_caches.values())
        total_hits = sum(s["hits"] for s in _session_stats.values())
        total_misses = sum(s["misses"] for s in _session_stats.values())
        total = total_hits + total_misses
        
        return {
            "active_sessions": len(_session_caches),
            "max_sessions": MAX_SESSIONS,
            "total_cached_items": total_items,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": round(total_hits / total, 3) if total > 0 else 0.0,
            "session_cache_size": SESSION_CACHE_SIZE,
            "session_ttl": SESSION_TTL
        }


def cleanup_expired_sessions() -> int:
    """
    Remove expired session caches.
    Called periodically by background task.
    
    Returns:
        Number of sessions cleaned up
    """
    cleaned = 0
    current_time = time.time()
    
    with _session_lock:
        expired_sessions = []
        
        for session_id, metadata in _session_metadata.items():
            last_access = metadata.get("last_access", 0)
            if current_time - last_access > SESSION_TTL:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            _session_caches.pop(session_id, None)
            _session_metadata.pop(session_id, None)
            _session_stats.pop(session_id, None)
            cleaned += 1
        
        if cleaned > 0:
            logger.info(f" Cleaned up {cleaned} expired sessions")
    
    return cleaned