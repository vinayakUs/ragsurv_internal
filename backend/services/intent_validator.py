"""
Intent Validator Service

LLM-based query intent classification for the corporate MOM RAG system.
Distinguishes between organizational queries (about documents/meetings) and off-topic requests.

Key Design:
- Uses LLM for classification (not keyword blocking)
- LLM understands CONTEXT, so "What jokes policy was discussed?" is ALLOWED
- Only trivial queries (empty, numbers-only) are pre-filtered

Usage:
    from backend.services.intent_validator import validate_query_intent
    
    is_valid, error_msg, intent = validate_query_intent(user_query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
"""

import os
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Configuration
INTENT_VALIDATION_ENABLED = os.environ.get("INTENT_VALIDATION_ENABLED", "true").lower() == "true"
INTENT_LLM_TIMEOUT = int(os.environ.get("INTENT_LLM_TIMEOUT", "30"))

# =============================================================================
# Pre-check Patterns (Only for trivial/obvious cases)
# =============================================================================
# These are NOT content-based - they catch malformed inputs only

def _is_trivial_query(query: str) -> Tuple[bool, str]:
    """
    Quick pre-check for obviously invalid queries.
    Returns (is_trivial, error_message).
    """
    query = query.strip()
    
    # Too short
    if len(query) < 3:
        return True, "Query too short. Please enter a complete question."
    
    # Only numbers and spaces
    if re.match(r'^[\d\s\.\,\+\-\*\/\=]+$', query):
        return True, "Please enter a text query, not a calculation."
    
    # Single word greetings (exact match only)
    single_word_reject = {
        'hi', 'hello', 'hey', 'bye', 'thanks', 'thank', 'ok', 'okay', 
        'yes', 'no', 'sure', 'fine', 'good', 'great', 'nice', 'cool'
    }
    if query.lower() in single_word_reject:
        return True, "Please enter a question about meetings or documents."
    
    return False, ""


# =============================================================================
# LLM Intent Classification
# =============================================================================

# The prompt is designed to understand CONTEXT
INTENT_CLASSIFICATION_PROMPT = """You are a query classifier for a corporate document system.

The system contains:
- Meeting minutes, agendas, and notes
- Committee reports and board decisions
- Action items and resolutions
- Participant and attendance records
- Compliance and regulatory documents
- Financial reports and budgets

TASK: Classify if the query is asking about ORGANIZATIONAL CONTENT or is OFF-TOPIC.

ORGANIZATIONAL (valid query): Questions about meetings, documents, reports, decisions, members, committees, dates, policies, financials, compliance, or ANY topic that might be in organizational documents - even if the specific topic seems casual or unusual.

OFF-TOPIC (reject): Pure entertainment (tell me a joke), general knowledge unrelated to organization (capital of France), personal assistant requests (set a timer), coding/math help, creative writing.

IMPORTANT: If the query COULD be asking about documented organizational content, classify as ORGANIZATIONAL.

Examples:
- "Tell me a joke" ? OFF-TOPIC (entertainment request, not about documents)
- "What jokes policy was discussed in HR meeting?" ? ORGANIZATIONAL (asking about meeting content)
- "What is 2+2?" ? OFF-TOPIC (math, not organizational)
- "What was the Q3 budget?" ? ORGANIZATIONAL (financial query)
- "Write a poem" ? OFF-TOPIC (creative writing)
- "Who attended the board meeting?" ? ORGANIZATIONAL (attendance query)
- "What's the weather?" ? OFF-TOPIC (general info, not organizational)
- "How did weather affect operations?" ? ORGANIZATIONAL (could be in reports)

Query: "{query}"

Reply with ONLY one word: ORGANIZATIONAL or OFF-TOPIC"""


def _classify_with_llm(query: str) -> Tuple[str, bool]:
    """
    Use LLM to classify query intent.
    Returns (intent, success).
    """
    try:
        from backend.services.llm_service import generate_response
        
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        
        response = generate_response(
            prompt=prompt,
            temperature=0.0,  # Deterministic
            max_tokens=10     # Only need one word
        )
        
        response = response.strip().upper()
        
        # Parse response
        if "ORGANIZATIONAL" in response:
            return "ORGANIZATIONAL", True
        elif "OFF-TOPIC" in response or "OFF_TOPIC" in response or "OFFTOPIC" in response:
            return "OFF-TOPIC", True
        else:
            # Couldn't parse - default to allowing (fail open for usability)
            logger.warning(f"Could not parse LLM intent response: '{response}', defaulting to ORGANIZATIONAL")
            return "ORGANIZATIONAL", True
            
    except Exception as e:
        logger.error(f"LLM intent classification failed: {e}")
        # On LLM failure, allow query through (fail open)
        return "ORGANIZATIONAL", False


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_query_intent(query: str) -> Tuple[bool, str, str]:
    """
    Validate query intent using LLM-based classification.
    
    Args:
        query: User's input query
        
    Returns:
        Tuple of (is_valid, error_message, intent)
        - is_valid: True if query should proceed to RAG
        - error_message: Empty if valid, otherwise rejection reason
        - intent: "ORGANIZATIONAL", "OFF-TOPIC", or "SKIPPED"
    """
    if not INTENT_VALIDATION_ENABLED:
        return True, "", "SKIPPED"
    
    if not query or not isinstance(query, str):
        return True, "", "SKIPPED"  # Empty queries handled elsewhere
    
    query = query.strip()
    
    # Quick pre-check for trivial queries
    is_trivial, trivial_error = _is_trivial_query(query)
    if is_trivial:
        logger.info(f"[INTENT] Pre-check rejected: '{query[:30]}...' - {trivial_error}")
        return False, trivial_error, "TRIVIAL"
    
    # LLM classification for all other queries
    intent, llm_success = _classify_with_llm(query)
    
    if intent == "ORGANIZATIONAL":
        logger.debug(f"[INTENT] Allowed: '{query[:30]}...'")
        return True, "", intent
    else:
        error_msg = "This query appears to be off-topic. Please ask questions related to meetings, documents, reports, or organizational content."
        logger.info(f"[INTENT] Rejected (OFF-TOPIC): '{query[:30]}...'")
        return False, error_msg, intent


def validate_query_intent_batch(queries: list) -> Tuple[bool, str, int, str]:
    """
    Validate a batch of queries.
    
    Returns:
        Tuple of (all_valid, error_message, failed_index, failed_intent)
    """
    for i, query in enumerate(queries):
        is_valid, error, intent = validate_query_intent(query)
        if not is_valid:
            return False, f"Query at index {i}: {error}", i, intent
    
    return True, "", -1, "ALL_VALID"


# =============================================================================
# CLI Testing
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_cases = [
        # Should be REJECTED (OFF-TOPIC)
        ("Tell me a joke", False),
        ("What is the capital of France?", False),
        ("Write a poem about nature", False),
        ("What is 2+2?", False),
        ("Hello", False),
        ("123", False),
        
        # Should be ALLOWED (ORGANIZATIONAL)
        ("What was discussed in the board meeting?", True),
        ("Who are the members of the audit committee?", True),
        ("Summarize the Q3 financial report", True),
        ("What action items are pending?", True),
        ("What jokes policy was discussed in HR meeting?", True),
        ("How did weather affect Q3 operations?", True),
    ]
    
    print("=" * 60)
    print("INTENT VALIDATOR TEST")
    print("=" * 60)
    
    for query, should_pass in test_cases:
        is_valid, error, intent = validate_query_intent(query)
        passed = is_valid == should_pass
        status = "?" if passed else "?"
        result = "ALLOW" if is_valid else "REJECT"
        print(f"{status} [{intent}] '{query[:40]}...' ? {result}")
    
    print("=" * 60)
