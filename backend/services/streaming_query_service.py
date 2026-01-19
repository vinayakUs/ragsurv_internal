"""
Streaming Query Processing Service

Provides parallel chunk processing with SSE streaming support.
Each chunk is processed through the LLM in parallel and results
are yielded as they complete for real-time frontend updates.
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from backend.services.llm_service import answer_from_chunks
from backend.services.json_ans_formatter import extract_final_json_from_llm
from backend.services.structured_logger import get_structured_logger, get_request_id

logger = logging.getLogger(__name__)
slog = get_structured_logger("streaming")

# Configuration for parallel/sequential processing
# ENABLE_PARALLEL_CHUNK_PROCESSING: Set to 'false' to revert to sequential mode
ENABLE_PARALLEL_CHUNK_PROCESSING = os.getenv("ENABLE_PARALLEL_CHUNK_PROCESSING", "true").lower() == "true"

# MAX_PARALLEL_CHUNKS: Number of chunks to process concurrently per query
# With load balancing: each worker can process this many chunks in parallel
# Recommended: 2-4 for single Ollama, 4-8 for load-balanced Ollama cluster
MAX_PARALLEL_CHUNKS = int(os.getenv("MAX_PARALLEL_CHUNKS", "3"))
CHUNK_TIMEOUT_SECONDS = int(os.getenv("CHUNK_TIMEOUT_SECONDS", "120"))
MAX_WORKERS = int(os.getenv("STREAMING_EXECUTOR_WORKERS", "8"))

# Thread pool for LLM processing (CPU-bound via network)
_streaming_executor = None


def get_streaming_executor() -> ThreadPoolExecutor:
    """Get or create the streaming thread pool executor."""
    global _streaming_executor
    if _streaming_executor is None:
        _streaming_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        slog.info("EXECUTOR_CREATED", workers=MAX_WORKERS)
    return _streaming_executor


def process_chunk_group_sync(
    chunk_group: List[Dict[str, Any]],
    query: str,
    chunk_index: int
) -> Dict[str, Any]:
    """
    Synchronous processing of a single chunk group with LLM.
    This is run in a thread pool executor.
    
    Args:
        chunk_group: List of chunks to process
        query: User query string
        chunk_index: Index of this chunk group for ordering
    
    Returns:
        Dict with processed result and chunk_index
    """
    try:
        start_time = time.time()
        
        # ULTRA-FAST MODE for first chunk: prioritize small, high-quality chunks
        if chunk_index == 0:
            # Sort by composite score: high quality + small size = faster processing
            # Formula: (weighted_score * 1000) - (text_length * 0.1)
            # This ensures we get high-quality chunks that are also small
            chunk_group.sort(
                key=lambda x: x.get("weighted_score", 0) * 1000 - len(x.get("text", "")) * 0.1,
                reverse=True
            )
            top_chunks = chunk_group[:2]  # Only 2 best small chunks for speed
        else:
            # Quality mode: standard sorting
            chunk_group.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
            top_chunks = chunk_group[:5]  # 5 chunks for quality
        
        # Prepare text for LLM
        chunk_texts = [
            chunk.get("text", "") 
            for chunk in top_chunks 
            if chunk.get("text", "").strip()
        ]
        
        if not chunk_texts:
            return {
                "chunk_index": chunk_index,
                "status": "empty",
                "result": None,
                "processing_time": 0.0
            }
        
        # Generate answer using LLM (prioritize first chunk for speed)
        is_first = (chunk_index == 0)
        answer_text = answer_from_chunks(chunk_texts, query, is_first_chunk=is_first)
        
        # Build source documents
        source_documents: List[str] = []
        for chunk in top_chunks:
            doc_name = chunk.get("document_name", "")
            page = chunk.get("page_number") or chunk.get("page", "")
            text = chunk.get("text", "").strip()
            excerpt = text[:180].replace("\n", " ") + "..." if len(text) > 180 else text
            if doc_name:
                entry = f'{doc_name} (page {page}) - "{excerpt}"' if page else f'{doc_name} - "{excerpt}"'
                if entry not in source_documents:
                    source_documents.append(entry)
        
        # Extract top source
        top_chunk = top_chunks[0]
        source = top_chunk.get("document_name", "")
        score = round(top_chunk.get("weighted_score", 0.0) * 10, 1)
        
        # Parse JSON response
        try:
            answer_json = extract_final_json_from_llm(answer_text)
        except Exception:
            answer_json = None
        
        if not answer_json:
            return {
                "chunk_index": chunk_index,
                "status": "no_answer",
                "result": None,
                "processing_time": time.time() - start_time
            }
        
        result = {
            "title": str(answer_json.get('title', 'Query Result')) if answer_json else "Query Result",
            "answer": str(answer_json.get('answer', answer_text)) if answer_json else answer_text,
            "source_documents": source_documents,
            "top_source": source,
            "score": str(score),
            "chunk_count": len(top_chunks),
            "chunks": top_chunks,
            "top_source_doc_id": top_chunk.get("doc_id"),
            "top_meeting_department": top_chunk.get("department"),
            "top_meeting_group": top_chunk.get("group"),
            "meeting_date": top_chunk.get("meeting_date"),
            "meeting_name": top_chunk.get("meeting_name")
        }
        
        processing_time = time.time() - start_time
        slog.info("CHUNK_PROCESSED", chunk=chunk_index, duration_ms=processing_time*1000)
        
        return {
            "chunk_index": chunk_index,
            "status": "success",
            "result": result,
            "processing_time": processing_time
        }
        
    except Exception as e:
        slog.exception("CHUNK_ERROR", chunk=chunk_index)
        return {
            "chunk_index": chunk_index,
            "status": "error",
            "error": str(e),
            "result": None,
            "processing_time": 0.0
        }


async def process_chunk_async(
    chunk_group: List[Dict[str, Any]],
    query: str,
    chunk_index: int
) -> Dict[str, Any]:
    """
    Async wrapper to process a single chunk group.
    Uses thread pool executor for the blocking LLM call.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                get_streaming_executor(),
                process_chunk_group_sync,
                chunk_group,
                query,
                chunk_index
            ),
            timeout=CHUNK_TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        slog.error("CHUNK_TIMEOUT", chunk=chunk_index, timeout_s=CHUNK_TIMEOUT_SECONDS)
        return {
            "chunk_index": chunk_index,
            "status": "timeout",
            "error": f"Processing timed out after {CHUNK_TIMEOUT_SECONDS}s",
            "result": None,
            "processing_time": CHUNK_TIMEOUT_SECONDS
        }


async def process_chunks_parallel_streaming_v2(
    chunk_groups: List[tuple],
    query: str,
    max_parallel: int = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process chunk groups in PARALLEL with controlled concurrency.
    
    This enables true parallel processing where multiple chunks are sent to the LLM
    simultaneously, controlled by a semaphore to prevent overload.
    
    Args:
        chunk_groups: List of (group_key, chunks) tuples
        query: User query string
        max_parallel: Maximum concurrent chunks (defaults to MAX_PARALLEL_CHUNKS)
    
    Yields:
        Dict with chunk_index, status, result as each completes (unordered)
    """
    if not chunk_groups:
        return
    
    total_chunks = len(chunk_groups)
    max_parallel = max_parallel or MAX_PARALLEL_CHUNKS
    
    slog.info("PARALLEL_START", total_chunks=total_chunks, max_parallel=max_parallel)
    start_time = time.time()
    
    # Semaphore to limit concurrent chunk processing
    semaphore = asyncio.Semaphore(max_parallel)
    
    async def process_with_semaphore(idx: int, group_key: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Process a single chunk with semaphore control."""
        async with semaphore:
            slog.info("CHUNK_START", chunk=idx, total=total_chunks)
            result = await process_chunk_async(chunks, query, idx)
            return result
    
    # Create tasks for all chunks
    tasks = [
        process_with_semaphore(idx, group_key, chunks)
        for idx, (group_key, chunks) in enumerate(chunk_groups)
    ]
    
    # Process chunks in parallel and yield results as they complete
    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            completed += 1
            
            # Add progress metadata
            result["progress_info"] = {
                "completed": completed,
                "total": total_chunks,
                "remaining": total_chunks - completed,
                "mode": "parallel"
            }
            
            yield result
            
        except Exception as e:
            slog.exception("PARALLEL_CHUNK_ERROR", error=str(e)[:50])
            completed += 1
            yield {
                "chunk_index": -1,
                "status": "error",
                "error": str(e),
                "result": None,
                "progress_info": {
                    "completed": completed,
                    "total": total_chunks,
                    "remaining": total_chunks - completed,
                    "mode": "parallel"
                }
            }
    
    total_time = time.time() - start_time
    slog.info(
        "PARALLEL_COMPLETE",
        total_chunks=total_chunks,
        duration_ms=total_time*1000,
        avg_per_chunk_ms=(total_time/total_chunks)*1000 if total_chunks > 0 else 0,
        speedup_vs_sequential=f"{(total_chunks * 30 / total_time):.1f}x" if total_time > 0 else "N/A"
    )


async def process_chunks_sequential_streaming(
    chunk_groups: List[tuple],  # List of (group_key, chunk_list) tuples
    query: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process chunk groups SEQUENTIALLY (one at a time) and yield results with queue status.
    
    This replaces the previous parallel processing approach to ensure only one
    chunk is sent to the LLM at a time. Other chunks wait in a queue.
    
    Args:
        chunk_groups: List of (group_key, chunks) tuples
        query: User query string
    
    Yields:
        Dict with chunk_index, status, result, and queue_info as each completes
    """
    if not chunk_groups:
        return
    
    total_chunks = len(chunk_groups)
    slog.info("SEQUENTIAL_START", total_chunks=total_chunks, mode="one_at_a_time")
    start_time = time.time()
    
    # Process chunks ONE AT A TIME - sequential processing
    for idx, (group_key, chunks) in enumerate(chunk_groups):
        queue_remaining = total_chunks - idx - 1
        
        slog.info("CHUNK_QUEUED", 
                  chunk=idx, 
                  position=idx+1, 
                  total=total_chunks, 
                  remaining=queue_remaining)
        
        # Process this single chunk (blocks until complete)
        result = await process_chunk_async(chunks, query, idx)
        
        # Add queue metadata to result for frontend display
        result["queue_info"] = {
            "position": idx + 1,
            "total": total_chunks,
            "remaining": queue_remaining,
            "status": "completed"
        }
        
        yield result
    
    total_time = time.time() - start_time
    slog.info("SEQUENTIAL_COMPLETE", 
              total_chunks=total_chunks, 
              duration_ms=total_time*1000,
              avg_per_chunk_ms=(total_time/total_chunks)*1000 if total_chunks > 0 else 0)


# Smart routing function - chooses parallel or sequential based on config
async def process_chunks_parallel_streaming(
    chunk_groups: List[tuple],
    query: str,
    max_parallel: int = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Process chunks using parallel or sequential mode based on configuration.
    
    Checks ENABLE_PARALLEL_CHUNK_PROCESSING environment variable:
    - true: Uses parallel processing (process_chunks_parallel_streaming_v2)
    - false: Uses sequential processing (process_chunks_sequential_streaming)
    
    This provides backward compatibility and easy rollback via environment variable.
    """
    if ENABLE_PARALLEL_CHUNK_PROCESSING:
        slog.info("PROCESSING_MODE", mode="parallel", max_parallel=max_parallel or MAX_PARALLEL_CHUNKS)
        async for result in process_chunks_parallel_streaming_v2(chunk_groups, query, max_parallel):
            yield result
    else:
        slog.info("PROCESSING_MODE", mode="sequential")
        async for result in process_chunks_sequential_streaming(chunk_groups, query):
            yield result


def format_sse_event(data: Dict[str, Any], event_type: str = "chunk") -> str:
    """
    Format data as an SSE event string.
    
    Args:
        data: Data to serialize as JSON
        event_type: SSE event type (chunk, done, error)
    
    Returns:
        Formatted SSE event string
    """
    json_data = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {json_data}\n\n"


async def create_streaming_response(
    chunk_groups: List[tuple],
    query: str,
    session_id: str = "anonymous"
) -> AsyncGenerator[str, None]:
    """
    Create an async generator that yields SSE events for streaming response.
    
    Now processes chunks SEQUENTIALLY (one at a time) with queue status.
    
    Args:
        chunk_groups: List of (group_key, chunks) tuples
        query: User query string
        session_id: Session ID for logging
    
    Yields:
        SSE event strings with queue information
    """
    processing_mode = "parallel" if ENABLE_PARALLEL_CHUNK_PROCESSING else "sequential"
    slog.info("STREAM_START", total_chunks=len(chunk_groups), mode=processing_mode)
    
    total_chunks = len(chunk_groups)
    processed = 0
    successful = 0
    
    # Send initial event with total count and processing mode
    processing_mode = "parallel" if ENABLE_PARALLEL_CHUNK_PROCESSING else "sequential"
    mode_message = (
        f"Processing {total_chunks} chunk(s) in parallel (max {MAX_PARALLEL_CHUNKS} concurrent)"
        if ENABLE_PARALLEL_CHUNK_PROCESSING
        else f"Processing {total_chunks} chunk(s) sequentially"
    )
    
    yield format_sse_event({
        "type": "start",
        "total_chunks": total_chunks,
        "query": query,
        "processing_mode": processing_mode,
        "max_parallel": MAX_PARALLEL_CHUNKS if ENABLE_PARALLEL_CHUNK_PROCESSING else 1,
        "queue_info": {
            "total": total_chunks,
            "mode": processing_mode,
            "message": mode_message
        }
    }, "start")
    
    # Stream chunk results as they complete (sequentially)
    async for chunk_result in process_chunks_parallel_streaming(chunk_groups, query):
        processed += 1
        if chunk_result.get("status") == "success":
            successful += 1
        
        # Get queue info from result (added by process_chunks_sequential_streaming)
        queue_info = chunk_result.get("queue_info", {})
        
        yield format_sse_event({
            "type": "chunk",
            "chunk_index": chunk_result.get("chunk_index"),
            "status": chunk_result.get("status"),
            "result": chunk_result.get("result"),
            "processing_time": chunk_result.get("processing_time"),
            "progress": f"{processed}/{total_chunks}",
            "queue_info": {
                "position": queue_info.get("position", processed),
                "total": queue_info.get("total", total_chunks),
                "remaining": queue_info.get("remaining", total_chunks - processed),
                "status": queue_info.get("status", "completed"),
                "next_in_queue": queue_info.get("remaining", 0) > 0
            }
        }, "chunk")
    
    # Send completion event
    yield format_sse_event({
        "type": "done",
        "total_processed": processed,
        "successful": successful,
        "failed": processed - successful,
        "queue_info": {
            "status": "all_completed",
            "message": f"All {total_chunks} chunk(s) processed"
        }
    }, "done")
    
    slog.info("STREAM_COMPLETE", successful=successful, failed=total_chunks-successful)
