import logging
from typing import List, Dict, Any, Optional
import os
import math

# Tokenization
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # fallback when not installed

# Sentence segmentation
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _NLTK_READY = True
    nltk = None
    sent_tokenize = None
    _NLTK_READY = False
    pass

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except Exception:
    nltk = None
    sent_tokenize = None
    _NLTK_READY = False

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = 'gpt-3.5-turbo') -> int:
    """Approximate token count using tiktoken if available, else fallback to words."""
    if not text:
        return 0
    if tiktoken is not None:
        try:
            # Heuristic: use cl100k_base for many chat models
            enc = tiktoken.get_encoding('cl100k_base')
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback: 1 token ≈ 0.75 words (rough heuristic)
    words = text.split()
    return max(1, math.ceil(len(words) / 0.75))


def sentence_split(text: str) -> List[str]:
    """Split text into sentences using NLTK if available; fallback to simple rule."""
    if not text:
        return []
    if sent_tokenize is not None:
        try:
            return [s.strip() for s in sent_tokenize(text) if s and s.strip()]
        except Exception:
            pass
    # Simple fallback: split on punctuation
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p and p.strip()]


def adaptive_sentence_chunks(text: str, max_tokens: int = 400, overlap_sentences: int = 1) -> List[str]:
    """
    Build chunks by accumulating sentences until reaching token budget.
    Slight sentence overlap to preserve context.
    """
    sents = sentence_split(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for s in sents:
        st = count_tokens(s)
        if cur and cur_tokens + st > max_tokens:
            chunks.append(' '.join(cur).strip())
            # start new chunk with overlap
            if overlap_sentences > 0:
                cur = cur[-overlap_sentences:]
                cur_tokens = sum(count_tokens(x) for x in cur)
            else:
                cur = []
                cur_tokens = 0
        cur.append(s)
        cur_tokens += st
    if cur:
        chunks.append(' '.join(cur).strip())
    # filter tiny chunks
    return [c for c in chunks if count_tokens(c) >= 20 or len(c.split()) >= 50]


def get_dynamic_chunks(docs: List[Dict[str, Any]], max_context_tokens: int = 3000) -> List[Dict[str, Any]]:
    """
    Select top docs that fit within the token budget. Assumes docs are pre-sorted by relevance.
    Each doc has a 'text' field.
    """
    out: List[Dict[str, Any]] = []
    used = 0
    for d in docs:
        t = d.get('text', '')
        if not t:
            continue
        ct = count_tokens(t)
        if used + ct > max_context_tokens:
            break
        out.append(d)
        used += ct
    return out


def dedupe_by_id(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for d in docs:
        did = d.get('id')
        if did in seen:
            continue
        seen.add(did)
        out.append(d)
    return out