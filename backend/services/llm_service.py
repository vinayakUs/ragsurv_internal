# -*- coding: utf-8 -*-
"""
LLM service - 100% Local Ollama (Mistral)

Replaces:
 - OpenAI
 - OpenRouter

Uses:
 - Local Ollama server
 - Mistral / Mixtral models

Public API (unchanged):
 - ensure_model_loaded()
 - generate_response()
 - answer_from_chunks()
 - health_check()
"""

import os
import logging
import requests
import asyncio
from typing import Optional, List
from datetime import datetime


# Import caching service
try:
    from backend.services.cache_service import cache_llm_response, get_cached_llm_response
    CACHING_ENABLED = True
except ImportError:
    CACHING_ENABLED = False
    logging.warning("Cache service not available for LLM, running without caching")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =========================
# Ollama Configuration
# =========================
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.17.136.240:11434").rstrip("/")
OLLAMA_HOST_FALLBACK = os.environ.get("OLLAMA_HOST_FALLBACK", "http://172.17.51.248:11434").rstrip("/")
OLLAMA_FALLBACK_TIMEOUT = int(os.environ.get("OLLAMA_FALLBACK_TIMEOUT", "10"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
DEFAULT_MODEL = OLLAMA_MODEL

# Track which host is currently active
_current_ollama_host = OLLAMA_HOST
_fallback_active = False
_last_health_check = 0
_health_check_interval = 30  # seconds

# Concurrency control - limit simultaneous LLM requests
MAX_CONCURRENT_LLM_REQUESTS = int(os.environ.get("MAX_CONCURRENT_LLM", "25"))
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

# Connection pool configuration (optimized for speed)
OLLAMA_POOL_CONNECTIONS = int(os.environ.get("OLLAMA_POOL_CONNECTIONS", "60"))
OLLAMA_POOL_MAXSIZE = int(os.environ.get("OLLAMA_POOL_MAXSIZE", "60"))

# =========================
# HTTP Connection Pooling
# =========================
_ollama_session = None

def _get_ollama_session() -> requests.Session:
    """Get or create a connection-pooled HTTP session for Ollama requests."""
    global _ollama_session
    if _ollama_session is None:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        _ollama_session = requests.Session()
        
        # Retry strategy for transient failures
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        # Connection pooling adapter - configurable for load balancing
        adapter = HTTPAdapter(
            pool_connections=OLLAMA_POOL_CONNECTIONS,  # Number of pools
            pool_maxsize=OLLAMA_POOL_MAXSIZE,          # Connections per pool
            max_retries=retry_strategy
        )
        
        _ollama_session.mount("http://", adapter)
        _ollama_session.mount("https://", adapter)
        
        logger.info(f"?? Created Ollama connection pool (size=20, max_concurrent={MAX_CONCURRENT_LLM_REQUESTS})")
    
    return _ollama_session


def _check_ollama_health(host: str) -> bool:
    """Quick health check for Ollama server."""
    try:
        session = _get_ollama_session()
        response = session.get(f"{host}/api/tags", timeout=OLLAMA_FALLBACK_TIMEOUT)
        return response.status_code == 200
    except Exception:
        return False


def _get_active_ollama_host() -> str:
    """Get the currently active Ollama host with automatic fallback."""
    global _current_ollama_host, _fallback_active, _last_health_check
    
    import time
    current_time = time.time()
    
    # Only check health periodically to avoid overhead
    if current_time - _last_health_check < _health_check_interval:
        return _current_ollama_host
    
    _last_health_check = current_time
    
    # Try primary host first
    if _check_ollama_health(OLLAMA_HOST):
        if _fallback_active:
            logger.info(f"[OK] Primary Ollama restored: {OLLAMA_HOST}")
            _fallback_active = False
        _current_ollama_host = OLLAMA_HOST
        return _current_ollama_host
    
    # Fallback to secondary host
    logger.warning(f"[WARN] Primary Ollama unreachable, trying fallback: {OLLAMA_HOST_FALLBACK}")
    if _check_ollama_health(OLLAMA_HOST_FALLBACK):
        if not _fallback_active:
            logger.info(f"[FALLBACK] Switched to fallback Ollama: {OLLAMA_HOST_FALLBACK}")
            _fallback_active = True
        _current_ollama_host = OLLAMA_HOST_FALLBACK
        return _current_ollama_host
    
    # Both failed, return primary and let it fail naturally
    logger.error(f"[ERROR] Both Ollama hosts unreachable!")
    return OLLAMA_HOST


def _ollama_url(path: str) -> str:
    """Build Ollama URL using active host (with fallback support)."""
    active_host = _get_active_ollama_host()
    return f"{active_host}{path if path.startswith('/') else '/' + path}"


# =========================
# Model Availability
# =========================
def _list_ollama_models() -> Optional[List[str]]:
    try:
        session = _get_ollama_session()
        r = session.get(_ollama_url("/api/tags"), timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m.get("name") for m in data.get("models", [])]
    except Exception as e:
        logger.error("Failed to list Ollama models: %s", e)
        return None


def ensure_model_loaded(model_name: Optional[str] = None) -> bool:
    model_name = model_name or DEFAULT_MODEL
    models = _list_ollama_models()

    if not models:
        logger.error("Ollama not running or no models installed.")
        return False

    for m in models:
        if model_name in m:
            logger.info("Ollama model '%s' is available.", model_name)
            return True

    logger.warning("Model '%s' not found. Available models: %s", model_name, models)
    return False


# =========================
# Generation
# =========================
def _generate_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Generate with Ollama, with automatic fallback on failure."""
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "num_predict": max_tokens,
    }

    session = _get_ollama_session()
    
    # Try with current host (includes automatic fallback logic)
    try:
        r = session.post(
            _ollama_url("/api/generate"),
            json=payload,
            timeout=120,  # Reduced from 300s for faster failure detection
        )
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
    
    except Exception as e:
        # If primary fails, force try fallback immediately
        global _current_ollama_host, _fallback_active
        if not _fallback_active:
            logger.warning(f"[WARN] Primary Ollama failed, forcing fallback: {e}")
            _current_ollama_host = OLLAMA_HOST_FALLBACK
            _fallback_active = True
            
            # Retry with fallback
            try:
                r = session.post(
                    f"{OLLAMA_HOST_FALLBACK}/api/generate",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                logger.info(f"[OK] Fallback Ollama succeeded")
                return (data.get("response") or "").strip()
            except Exception as fallback_error:
                logger.error(f"[ERROR] Both Ollama hosts failed: {fallback_error}")
                raise
        else:
            raise


def generate_response(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Generate LLM response with caching support."""
    # Check cache first
    if CACHING_ENABLED:
        cache_key = f"{prompt}|{model or DEFAULT_MODEL}|{temperature}"
        cached = get_cached_llm_response(cache_key)
        if cached is not None:
            return cached
    #logger.info("Generating response using local Ollama (%s)", model or DEFAULT_MODEL)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{current_time} | Generating response using local Ollama ({model or DEFAULT_MODEL})")
    result = _generate_with_ollama(prompt, model, temperature, max_tokens)
    logger.info(f"{current_time} | Generated response using local Ollama ({model or DEFAULT_MODEL})")
    
    # Cache the result
    if CACHING_ENABLED and result:
        cache_key = f"{prompt}|{model or DEFAULT_MODEL}|{temperature}"
        cache_llm_response(cache_key, result)
    
    return result


# =========================
# RAG / Chunk Answering
# =========================
# Performance configurations
LLM_MAX_CHUNKS = int(os.environ.get("LLM_MAX_CHUNKS", "8"))
LLM_ANSWER_MAX_TOKENS = int(os.environ.get("LLM_ANSWER_MAX_TOKENS", "1024"))

def answer_from_chunks(
    chunks: List[str],
    question: str,
    model: Optional[str] = None,
    is_first_chunk: bool = False,
) -> str:
    if not chunks:
        return "No answer found in the knowledge base."

    chunks = [str(c).strip() for c in chunks if c and str(c).strip()]
    if not chunks:
        return "No answer found in the knowledge base."

    # ULTRA-FAST MODE for first chunk
    if is_first_chunk:
        context = "\n\n".join(chunks[:2])  # Only 2 chunks for speed
        
        # Ultra-short prompt
        prompt = (
            f"Context:\n{context}\n\n"
            f"Q: {question}\n\n"
            "Brief JSON:\n"
            '{"title":"...","answer":"..."}'
        )
        
    else:
        # QUALITY MODE for other chunks
        context = "\n\n".join(chunks[:LLM_MAX_CHUNKS])
        
        prompt = (
            "You are an expert summarizer. Your task is to extract and summarize facts from the provided text.\n"
            "1. **IGNORE missing information**: Do not write sentences saying what is NOT in the text.\n"
            "2. **Extract Facts**: Summarize all information in the context that is even remotely related to the User Topic.\n"
            "3. **STRICT MARKDOWN**: \n"
            "   - ALL data tables must be rendered as Markdown tables.\n"
            "   - ALL lists must be rendered as Markdown bullet points.\n"
            "   - Key metrics should be **bolded**.\n"
            "4. **Pivot**: If the exact answer is not found, summarize the closest available data.\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"User Topic: {question}\n\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  "title": "<short title>",\n'
            '  "answer": "<Detailed summary using heavy Markdown formatting (Tables/Lists).>"\n'
            "}\n\n"
            "Return '{\"title\":\"No Data\"...}' ONLY if the context is completely empty or 100% irrelevant."
        )
    
    # DYNAMIC num_predict based on context size (prevents over-generation)
    context_tokens = len(context) // 4  # Rough estimation: 1 token ≈ 4 chars
    if is_first_chunk:
        # First chunk: answer ~20% longer than context
        ideal_tokens = int(context_tokens * 1.2)
        max_tokens = min(512, max(128, ideal_tokens))
        temp = 0.0  # Deterministic for speed
    else:
        # Quality chunks: answer can be 50% longer
        ideal_tokens = int(context_tokens * 1.5)
        max_tokens = min(LLM_ANSWER_MAX_TOKENS, max(256, ideal_tokens))
        temp = 0.1

    try:
        response = generate_response(
            prompt, 
            model=model, 
            temperature=temp, 
            max_tokens=max_tokens
        )
        if not response or "no answer found" in response.lower():
            return "No answer found in the knowledge base."
        return response.strip()
    except Exception as e:
        logger.exception("Chunk answer failed: %s", e)
        return "No answer found in the knowledge base."

# =========================
# Health Check
# =========================
def health_check() -> dict:
    return {
        "llm_ok": ensure_model_loaded(),
        "using_local_ollama": True,
        "ollama_host": OLLAMA_HOST,
        "model": DEFAULT_MODEL,
    }


# =========================
# Manual Test
# =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print("Health:", health_check())
    print(generate_response("Say hello from local Mistral.", max_tokens=50))