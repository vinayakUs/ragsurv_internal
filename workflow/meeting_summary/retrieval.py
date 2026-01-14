
import logging
import os
import sys
from typing import List, Dict, Any, Optional

# Ensure backend modules are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.services.query_engine import query_knowledge_base_retrieval_only
from backend.services.qdrant_service import ensure_collection_exists

logger = logging.getLogger("meeting_retrieval")

class RetrievalService:
    """
    Retrieves context for meeting summary generation.
    """
    def __init__(self):
        try:
            ensure_collection_exists()
        except Exception:
            pass

    def get_context_for_agenda(self, meeting_id: int, agenda_text: str) -> Dict[str, List[str]]:
        """
        Retrieves relevant chunks continuously for an agenda item,
        grouped by document type (e.g., MOM, ATR, Submission, Supporting).
        """
        logger.info(f"Retrieving context for agenda: '{agenda_text[:30]}...' (Meeting {meeting_id})")
        
        try:
            # BM25 is handled internally by query_knowledge_base_retrieval_only if not provided
            results = query_knowledge_base_retrieval_only(
                query=agenda_text,
                filters={'meeting_id': int(meeting_id)},
                group_by=['doc_type']
            )
            
            if results.get("status") != "success":
                logger.warning(f"Retrieval failed or no results: {results.get('message')}")
                return {}

            # Parse results:
            # Structure: results['results']['doc_type']['MOM']['items'] -> list of chunks
            
            grouped_context = {}
            doc_type_groups = results.get("results", {}).get("doc_type", {})
            
            for doc_type, group_data in doc_type_groups.items():
                items = group_data.get("items", [])
                chunks = []
                for item in items:
                    text = item.get("text")
                    if text:
                        chunks.append(text)
                
                if chunks:
                    grouped_context[doc_type] = chunks
            
            logger.info(f"Retrieved context types: {list(grouped_context.keys())}")
            return grouped_context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {}
