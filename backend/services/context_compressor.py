from typing import List, Dict
import os
import logging

# Prefer OpenAI for higher-quality compression if available; fallback to local LLM
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from .llm_service import generate_response

logger = logging.getLogger(__name__)


def compress_context(docs: List[Dict], max_chars: int = 6000) -> str:
    """
    Compress a list of docs (with 'text' and optional 'source') into a concise, detail-preserving summary.
    Returns a single string suitable for LLM context.
    """
    if not docs:
        return ""

    # Build concatenated input with provenance markers
    parts = []
    for d in docs:
        src = d.get('document_name') or d.get('source') or 'unknown'
        txt = (d.get('text') or '').strip()
        if not txt:
            continue
        parts.append(f"From {src}:\n{txt}")
    context_text = "\n\n".join(parts)
    if len(context_text) > max_chars:
        context_text = context_text[:max_chars]

    api_key = os.environ.get('OPENAI_API_KEY')
    if OpenAI is not None and api_key:
        try:
            client = OpenAI()
            prompt = (
                "Summarize the key facts from the following texts while preserving critical details, numbers, and citations. "
                "Group by source where helpful. Keep it concise and factual. "
                "You must summarize only using the given text. Do not add any external information.\n\n" + context_text
            )
            resp = client.chat.completions.create(
                model=os.environ.get('OPENAI_COMPRESS_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return (resp.choices[0].message.content or '').strip()
        except Exception as e:
            logger.warning(f"OpenAI compression failed, falling back: {e}")

    # Fallback: use local LLM summarization
    prompt = (
        "Summarize the key facts from the following texts while preserving critical details, numbers, and named entities. "
        "Be concise and structured. You must summarize only using the given text. Do not add any external information.\n\n" + context_text
    )
    return generate_response(prompt)
