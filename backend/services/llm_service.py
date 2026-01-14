"""
LLM service — 100% Local Ollama (Mistral)

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
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://172.17.51.248:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b-instruct")
DEFAULT_MODEL = OLLAMA_MODEL

# Concurrency control - limit simultaneous LLM requests (increased for 70 users)
MAX_CONCURRENT_LLM_REQUESTS = int(os.environ.get("MAX_CONCURRENT_LLM", "10"))
_llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM_REQUESTS)

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
        
        # Connection pooling adapter
        adapter = HTTPAdapter(
            pool_connections=20,  # Number of pools
            pool_maxsize=20,      # Connections per pool
            max_retries=retry_strategy
        )
        
        _ollama_session.mount("http://", adapter)
        _ollama_session.mount("https://", adapter)
        
        logger.info(f"?? Created Ollama connection pool (size=20, max_concurrent={MAX_CONCURRENT_LLM_REQUESTS})")
    
    return _ollama_session


def _ollama_url(path: str) -> str:
    return f"{OLLAMA_HOST}{path if path.startswith('/') else '/' + path}"


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
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "num_predict": max_tokens,
    }

    session = _get_ollama_session()
    r = session.post(
        _ollama_url("/api/generate"),
        json=payload,
        timeout=300,
    )
    r.raise_for_status()

    data = r.json()
    return (data.get("response") or "").strip()


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
) -> str:
    if not chunks:
        return "No answer found in the knowledge base."

    chunks = [str(c).strip() for c in chunks if c and str(c).strip()]
    if not chunks:
        return "No answer found in the knowledge base."

    # Performance: Use only top N chunks (configurable, default 5)
    context = "\n\n".join(chunks[:LLM_MAX_CHUNKS])

    # Optimized prompt - shorter and more focused for faster LLM response
    # prompt = (
    #     f"Context:\n{context}\n\n"
    #     f"Question: {question}\n\n"
    #     "Answer based on the context above. Return JSON only:\n"
    #     '{"title":"<short title>","answer":"<answer from context>","reasoning":"<one sentence>"}\n'
    #     "If no answer found, return:\n"
    #     '{"title":"No Answer","answer":"No answer found.","reasoning":"No relevant context."}'
    # )
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

    try:
        response = generate_response(
            prompt, 
            model=model, 
            temperature=0.1, 
            max_tokens=LLM_ANSWER_MAX_TOKENS
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