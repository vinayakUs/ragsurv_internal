"""
In-Memory Caching Service for NSE RAG Application

Provides thread-safe LRU caching for:
- Query embeddings
- LLM responses
- Search results
- BM25 scores

Reduces latency for repeated queries and improves multi-user performance.
"""

import hashlib
import json
import logging
import time
from typing import Any, Optional, Dict, List
from functools import lru_cache
from cachetools import TTLCache
import threading

logger = logging.getLogger(__name__)

# Configuration - read from environment with optimized defaults for 70 concurrent users
import os
QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "2000"))  # Max cached queries
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "3000"))  # Max cached embeddings
LLM_CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "1500"))  # Max cached LLM responses
DEFAULT_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour in seconds for better reuse

# Thread-safe caches with TTL
_cache_lock = threading.RLock()
_query_cache = TTLCache(maxsize=QUERY_CACHE_SIZE, ttl=DEFAULT_TTL)
_embedding_cache = TTLCache(maxsize=EMBEDDING_CACHE_SIZE, ttl=DEFAULT_TTL)
_llm_cache = TTLCache(maxsize=LLM_CACHE_SIZE, ttl=DEFAULT_TTL)

# Statistics
_cache_stats = {
    "query_hits": 0,
    "query_misses": 0,
    "embedding_hits": 0,
    "embedding_misses": 0,
    "llm_hits": 0,
    "llm_misses": 0,
}


def _generate_cache_key(prefix: str, data: Any) -> str:
    """
    Generate a deterministic cache key from data.
    
    Args:
        prefix: Key prefix (e.g., 'query', 'embedding', 'llm')
        data: Data to hash (dict, string, etc.)
    
    Returns:
        Cache key string
    """
    if isinstance(data, dict):
        # Sort dict keys for deterministic hashing
        serialized = json.dumps(data, sort_keys=True)
    elif isinstance(data, (list, tuple)):
        serialized = json.dumps(data)
    else:
        serialized = str(data)
    
    hash_obj = hashlib.md5(serialized.encode('utf-8'))
    return f"{prefix}:{hash_obj.hexdigest()}"


def cache_query_result(query: str, filters: Optional[Dict], result: Dict) -> None:
    """
    Cache a query result.
    
    Args:
        query: Query string
        filters: Optional filters dict
        result: Query result to cache
    """
    try:
        cache_data = {"query": query, "filters": filters or {}}
        key = _generate_cache_key("query", cache_data)
        
        with _cache_lock:
            _query_cache[key] = {
                "result": result,
                "cached_at": time.time()
            }
        
        logger.debug(f"?? Cached query result: {query[:50]}...")
    except Exception as e:
        logger.warning(f"Failed to cache query result: {e}")


def get_cached_query_result(query: str, filters: Optional[Dict]) -> Optional[Dict]:
    """
    Get cached query result.
    
    Args:
        query: Query string
        filters: Optional filters dict
    
    Returns:
        Cached result or None if not found
    """
    try:
        cache_data = {"query": query, "filters": filters or {}}
        key = _generate_cache_key("query", cache_data)
        
        with _cache_lock:
            cached = _query_cache.get(key)
            if cached:
                _cache_stats["query_hits"] += 1
                age = time.time() - cached["cached_at"]
                logger.info(f"? Cache HIT for query (age: {age:.1f}s): {query[:50]}...")
                return cached["result"]
            else:
                _cache_stats["query_misses"] += 1
                logger.debug(f"? Cache MISS for query: {query[:50]}...")
                return None
    except Exception as e:
        logger.warning(f"Failed to get cached query: {e}")
        return None


def cache_embedding(text: str, embedding: List[float]) -> None:
    """
    Cache a text embedding.
    
    Args:
        text: Input text
        embedding: Generated embedding vector
    """
    try:
        key = _generate_cache_key("embedding", text)
        
        with _cache_lock:
            _embedding_cache[key] = {
                "embedding": embedding,
                "cached_at": time.time()
            }
        
        logger.debug(f"?? Cached embedding for: {text[:30]}...")
    except Exception as e:
        logger.warning(f"Failed to cache embedding: {e}")


def get_cached_embedding(text: str) -> Optional[List[float]]:
    """
    Get cached embedding for text.
    
    Args:
        text: Input text
    
    Returns:
        Cached embedding or None if not found
    """
    try:
        key = _generate_cache_key("embedding", text)
        
        with _cache_lock:
            cached = _embedding_cache.get(key)
            if cached:
                _cache_stats["embedding_hits"] += 1
                logger.debug(f"? Cache HIT for embedding: {text[:30]}...")
                return cached["embedding"]
            else:
                _cache_stats["embedding_misses"] += 1
                return None
    except Exception as e:
        logger.warning(f"Failed to get cached embedding: {e}")
        return None


def cache_llm_response(prompt: str, response: str) -> None:
    """
    Cache an LLM response.
    
    Args:
        prompt: LLM prompt
        response: LLM response
    """
    try:
        key = _generate_cache_key("llm", prompt)
        
        with _cache_lock:
            _llm_cache[key] = {
                "response": response,
                "cached_at": time.time()
            }
        
        logger.debug(f"?? Cached LLM response for: {prompt[:50]}...")
    except Exception as e:
        logger.warning(f"Failed to cache LLM response: {e}")


def get_cached_llm_response(prompt: str) -> Optional[str]:
    """
    Get cached LLM response.
    
    Args:
        prompt: LLM prompt
    
    Returns:
        Cached response or None if not found
    """
    try:
        key = _generate_cache_key("llm", prompt)
        
        with _cache_lock:
            cached = _llm_cache.get(key)
            if cached:
                _cache_stats["llm_hits"] += 1
                age = time.time() - cached["cached_at"]
                logger.info(f"? Cache HIT for LLM (age: {age:.1f}s): {prompt[:50]}...")
                return cached["response"]
            else:
                _cache_stats["llm_misses"] += 1
                return None
    except Exception as e:
        logger.warning(f"Failed to get cached LLM response: {e}")
        return None


def clear_all_caches() -> None:
    """Clear all caches."""
    with _cache_lock:
        _query_cache.clear()
        _embedding_cache.clear()
        _llm_cache.clear()
        logger.info("??? All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dict with cache stats including hit rates
    """
    with _cache_lock:
        query_total = _cache_stats["query_hits"] + _cache_stats["query_misses"]
        embedding_total = _cache_stats["embedding_hits"] + _cache_stats["embedding_misses"]
        llm_total = _cache_stats["llm_hits"] + _cache_stats["llm_misses"]
        
        return {
            "query_cache": {
                "size": len(_query_cache),
                "max_size": QUERY_CACHE_SIZE,
                "hits": _cache_stats["query_hits"],
                "misses": _cache_stats["query_misses"],
                "hit_rate": _cache_stats["query_hits"] / query_total if query_total > 0 else 0.0
            },
            "embedding_cache": {
                "size": len(_embedding_cache),
                "max_size": EMBEDDING_CACHE_SIZE,
                "hits": _cache_stats["embedding_hits"],
                "misses": _cache_stats["embedding_misses"],
                "hit_rate": _cache_stats["embedding_hits"] / embedding_total if embedding_total > 0 else 0.0
            },
            "llm_cache": {
                "size": len(_llm_cache),
                "max_size": LLM_CACHE_SIZE,
                "hits": _cache_stats["llm_hits"],
                "misses": _cache_stats["llm_misses"],
                "hit_rate": _cache_stats["llm_hits"] / llm_total if llm_total > 0 else 0.0
            }
        }


def get_cache_info() -> str:
    """Get human-readable cache info."""
    stats = get_cache_stats()
    
    info = "?? Cache Statistics:\n"
    for cache_name, cache_stat in stats.items():
        hit_rate_pct = cache_stat["hit_rate"] * 100
        info += f"  {cache_name}: {cache_stat['size']}/{cache_stat['max_size']} items, "
        info += f"{cache_stat['hits']} hits, {cache_stat['misses']} misses "
        info += f"({hit_rate_pct:.1f}% hit rate)\n"
    
    return info
