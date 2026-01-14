"""
Parallel Query Processing Service

Enables concurrent processing of multiple queries using asyncio.
Provides async wrappers for the synchronous query_knowledge_base_v2 function.
"""

import os
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from backend.services.query_engine import query_knowledge_base_v2
from backend.services.bm25_service import BM25Service

logger = logging.getLogger(__name__)

# Configuration - optimized for 70 concurrent users
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_QUERIES", "20"))
QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT_SECONDS", "120"))
MAX_WORKERS = int(os.getenv("ASYNC_EXECUTOR_MAX_WORKERS", "20"))

# Global concurrency control semaphore
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "30"))
_query_semaphore = None

def get_query_semaphore() -> asyncio.Semaphore:
    """Get or create the query concurrency semaphore."""
    global _query_semaphore
    if _query_semaphore is None:
        _query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
        logger.info(f"?? Created query semaphore (max concurrent: {MAX_CONCURRENT_QUERIES})")
    return _query_semaphore

# Thread pool for CPU-bound operations
_executor = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info(f"? Created thread pool executor with {MAX_WORKERS} workers")
    return _executor


async def process_single_query_async(
    query: str,
    bm25_service: BM25Service,
    filters: Optional[Dict[str, Any]] = None,
    query_index: int = 0
) -> Dict[str, Any]:
    """
    Process a single query asynchronously.
    
    Args:
        query: Query string to process
        bm25_service: BM25 service instance
        filters: Optional metadata filters (meeting_id, doc_type)
        query_index: Index of query in batch (for logging)
    
    Returns:
        Query result dictionary with status, answer, sources, etc.
    """
    try:
        logger.info(f"?? Processing query {query_index}: '{query[:50]}...'")
        start_time = time.time()
        
        # Run the synchronous query function in the executor
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                get_executor(),
                query_knowledge_base_v2,
                query,
                bm25_service,
                None,  # group_by (use default)
                filters
            ),
            timeout=QUERY_TIMEOUT
        )
        
        processing_time = time.time() - start_time
        logger.info(
            f"? Query {query_index} completed in {processing_time:.2f}s "
            f"(status: {result.get('status', 'unknown')})"
        )
        
        # Add query info to result
        result["query"] = query
        result["query_index"] = query_index
        
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"?? Query {query_index} timed out after {QUERY_TIMEOUT}s")
        return {
            "status": "error",
            "message": f"Query timed out after {QUERY_TIMEOUT} seconds",
            "query": query,
            "query_index": query_index,
            "grouped_by": [],
            "results": {},
            "summary": {"total_groups": 0, "groups_breakdown": {}},
            "processing_time": QUERY_TIMEOUT
        }
        
    except Exception as e:
        logger.exception(f"? Error processing query {query_index}")
        return {
            "status": "error",
            "message": f"Error processing query: {str(e)}",
            "query": query,
            "query_index": query_index,
            "grouped_by": [],
            "results": {},
            "summary": {"total_groups": 0, "groups_breakdown": {}},
            "processing_time": 0.0
        }


async def process_batch_queries(
    queries: List[str],
    bm25_service: BM25Service,
    filters: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple queries in parallel.
    
    Args:
        queries: List of query strings
        bm25_service: BM25 service instance
        filters: Optional list of filter dicts (one per query) or None
    
    Returns:
        List of results in same order as input queries
    """
    if not queries:
        return []
    
    # Ensure filters list matches queries length
    if filters is None:
        filters = [None] * len(queries)
    elif len(filters) != len(queries):
        logger.warning(
            f"Filters length ({len(filters)}) doesn't match queries length "
            f"({len(queries)}). Using global filter for all."
        )
        global_filter = filters[0] if filters else None
        filters = [global_filter] * len(queries)
    
    logger.info(f"?? Starting batch processing of {len(queries)} queries")
    batch_start = time.time()
    
    # Create tasks for all queries
    tasks = [
        process_single_query_async(
            query=query,
            bm25_service=bm25_service,
            filters=filter_dict,
            query_index=idx
        )
        for idx, (query, filter_dict) in enumerate(zip(queries, filters))
    ]
    
    # Process all queries concurrently
    # return_exceptions=True ensures partial results on failure
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    batch_time = time.time() - batch_start
    
    # Handle any exceptions that occurred
    final_results = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"? Query {idx} raised exception: {result}")
            final_results.append({
                "status": "error",
                "message": f"Query failed with exception: {str(result)}",
                "query": queries[idx],
                "query_index": idx,
                "grouped_by": [],
                "results": {},
                "summary": {"total_groups": 0, "groups_breakdown": {}},
                "processing_time": 0.0
            })
        else:
            final_results.append(result)
    
    # Calculate statistics
    successful = sum(1 for r in final_results if r.get("status") == "success")
    failed = len(final_results) - successful
    
    logger.info(
        f"? Batch processing completed: {successful}/{len(queries)} successful "
        f"in {batch_time:.2f}s ({batch_time/len(queries):.2f}s avg per query)"
    )
    
    return final_results


def validate_batch_request(queries: List[str]) -> tuple[bool, Optional[str]]:
    """
    Validate batch query request.
    
    Args:
        queries: List of query strings
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not queries:
        return False, "Queries list cannot be empty"
    
    if len(queries) > MAX_BATCH_SIZE:
        return False, f"Maximum {MAX_BATCH_SIZE} queries allowed per batch request"
    
    for idx, query in enumerate(queries):
        if not query or not query.strip():
            return False, f"Query at index {idx} is empty"
    
    return True, None