from typing import List
import os
import logging

# Prefer OpenAI if API key available; else fall back to local LLM
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from .llm_service import generate_response

logger = logging.getLogger(__name__)

# Performance optimization: Set to true to skip LLM query expansion
SKIP_QUERY_EXPANSION = os.environ.get("SKIP_QUERY_EXPANSION", "false").lower() == "true"


def expand_query(user_query: str, max_alternatives: int = 3) -> List[str]:
    """
    Generate semantically diverse reformulations of the query.
    Falls back to returning the original query if expansion is unavailable.
    """
    q = (user_query or '').strip()
    if not q:
        return []

    # Performance optimization: Skip expansion entirely if disabled
    if SKIP_QUERY_EXPANSION:
        logger.debug("Query expansion skipped (SKIP_QUERY_EXPANSION=true)")
        return [q]

    api_key = os.environ.get('OPENAI_API_KEY')
    if OpenAI is not None and api_key:
        try:
            client = OpenAI()
            prompt = f"Generate {max_alternatives} semantically diverse reformulations of this query in bullet points without numbering or extra text.\n\nQuery: {q}"
            resp = client.chat.completions.create(
                model=os.environ.get('OPENAI_MQR_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = resp.choices[0].message.content.strip()
            lines = [ln.strip('-* ').strip() for ln in content.splitlines() if ln.strip()]
            alts = [ln for ln in lines if ln]
            # Ensure uniqueness and include original query
            uniq = []
            for x in alts:
                if x.lower() not in {u.lower() for u in uniq}:
                    uniq.append(x)
            return [q] + uniq[:max_alternatives]
        except Exception as e:
            logger.warning(f"OpenAI expansion failed, falling back: {e}")

    # Fallback: use local LLM if available
    try:
        prompt = (
            "Rephrase the following query into up to "
            f"{max_alternatives} diverse alternatives. Return one per line, no numbering.\n\n" 
            f"Query: {q}"
        )
        text = generate_response(prompt)
        lines = [ln.strip('-* ').strip() for ln in (text or '').splitlines() if ln.strip()]
        alts = [ln for ln in lines if ln and ln.lower() != q.lower()]
        return [q] + alts[:max_alternatives]
    except Exception:
        return [q]
