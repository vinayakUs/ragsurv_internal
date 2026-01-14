import logging
from typing import List, Dict, Any, Optional
import os 
from backend.services.query_engine import query_knowledge_base_retrieval_only
from backend.services.qdrant_service import ensure_collection_exists

logger = logging.getLogger("agenda_retrieval")

def get_relevant_chunks(meeting_id: str, query: str, doc_type: str) -> List[str]:
    """
    Retrieves relevant document chunks for a given query by calling the internal query engine.
    """
    logger.info(f"[{meeting_id}] Retrieving {doc_type} chunks for query: '{query}'")

    # Ensure Qdrant Connection
    try:
        ensure_collection_exists()
    except Exception as e:
        logger.error(f"Cannot connect to Qdrant: {e}. Exiting.")
        return []

    # Run internal retrieval (BM25 is handled internally if None)
    try:
        data = query_knowledge_base_retrieval_only(
            query=query,
            bm25_service=None, # Internal function will lazy-load if needed
            filters={'meeting_id': meeting_id, 'doc_type': doc_type},
            group_by=['doc_type']
        )
        
        # Extract text chunks from standard response format
        extracted_chunks = []
        
        # Check success status
        if data.get("status") != "success":
            logger.warning(f"[{meeting_id}] Retrieval status: {data.get('status')} - {data.get('message')}")
            return []

        results = data.get("results", {})
        
        # Navigate standard structure: results -> [group_by_field] -> [group_value] -> items
        # here group_by='doc_type', so results -> doc_type -> [MOM/AGENDA/etc] -> items
        
        type_groups = results.get("doc_type", {})
        target_group = type_groups.get(doc_type, {})
        items = target_group.get("items", [])
        
        for item in items:
            text = item.get("text")
            if text:
                extracted_chunks.append(text)

        # Remove duplicates while preserving order
        unique_chunks = list(dict.fromkeys(extracted_chunks))
        
        logger.info(f"[{meeting_id}] Retrieved {len(unique_chunks)} unique chunks.")
        return unique_chunks

    except Exception as e:
        logger.error(f"[{meeting_id}] Error in get_relevant_chunks: {e}")
        return []

def get_raw_chunks(
    query: str, 
    group_by: Optional[List[str]] = None, 
    filters: Optional[Dict[str, Any]] = None
):
    """
    Retrieves chunks for a given query (Legacy/Debug helper).
    """
    logger.info(f"Retrieving for query: '{query}'")
    
    if group_by is None:
        group_by = ['meeting_id', 'doc_type']

    try:
        results = query_knowledge_base_retrieval_only(
            query=query,
            bm25_service=None,
            group_by=group_by,
            filters=filters
        )
        
        if results.get('status') == 'success':
            return results
        else:
            logger.warning(f"No results or error: {results.get('message')}")
            return {}

    except Exception as e:
        logger.exception("Error during retrieval execution")
        return {}