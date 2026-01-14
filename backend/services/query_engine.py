import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import logging
import functools
import time
import math
from backend.services.embedding_service import generate_query_embedding
from backend.services.qdrant_service import search_similar_chunks, count_documents
from backend.services.llm_service import answer_from_chunks
from .hybrid_search import hybrid_search
from .bm25_service import BM25Service
from .json_ans_formatter import extract_final_json_from_llm
from .query_expansion_service import expand_query
from .utils import dedupe_by_id

def normalize_scores_sigmoid(scores: List[float]) -> List[float]:
    """
    Normalize a list of raw scores to (0,1) using sigmoid.
    Keeps relative ordering and avoids negatives.
    """
    # Protect against extreme values numerically (optional)
    normalized = []
    for s in scores:
        try:
            val = 1.0 / (1.0 + math.exp(-float(s)))
        except OverflowError:
            val = 0.0 if s < 0 else 1.0
        normalized.append(float(val))
    return normalized


logger = logging.getLogger(__name__)


# Cache embeddings for faster repeated queries
@functools.lru_cache(maxsize=128)
def cached_generate_query_embedding(q_text: str):
    return generate_query_embedding(q_text)


_reranker: Optional[Union[object, bool]] = None


def _get_reranker() -> Optional[Union[object, bool]]:
    """Lazily load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            import os
            _reranker = CrossEncoder(os.getenv("RERANKER_MODEL_PATH","/mnt/desv/backend/ms-marco-MiniLM-L-6-v2"))
            # _reranker = CrossEncoder("/mnt/dev/backend/ms-marco-MiniLM-L-6-v2")
            logger.info("âœ… CrossEncoder initialized successfully.")
        except Exception as e:
            logger.warning(f"âš ï¸ CrossEncoder unavailable, skipping rerank: {e}")
            _reranker = False
    return _reranker



def retrieve_ranked_chunks(
    query: str, 
    bm25_service: Optional[BM25Service] = None, 
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Core retrieval logic: Expansion -> Hybrid Search -> Dedupe -> Filter -> Rerank -> Weight -> Sort
    """
    if not query or not query.strip():
        return []

    # Auto-initialize BM25 if not provided
    if bm25_service is None:
        try:
            bm25_service = BM25Service.get_client()
        except Exception as e:
            logger.warning(f"Could not auto-load BM25Service: {e}")

    # Local helper for Qdrant (similar to original)
    def qdrant_search_fn(q_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        emb = cached_generate_query_embedding(q_text)
        if emb is None:
            return []
        vec_hits = search_similar_chunks(emb, top_k=top_k, filters=filters)
        results: List[Dict[str, Any]] = []
        for r in vec_hits or []:
            rid = f"{r.get('document_id', '')}:{r.get('chunk_index', '')}"
            # Preserve all metadata fields from vector search results
            chunk_data = {
                "id": rid,
                "text": r.get("text", ""),
                "score": r.get("score", 0.0),
                "document_name": r.get("document_name", ""),
                "page_number": r.get("page_number", None),
            }
            # Add any other fields from the original result (e.g. metadata)
            for k, v in r.items():
                if k not in chunk_data:
                    chunk_data[k] = v
            results.append(chunk_data)
        return results

    # Query expansion
    subqueries = expand_query(query) or [query]
    merged: List[Dict[str, Any]] = []
    
    for sq in subqueries:

        hits = hybrid_search(sq, bm25_service, qdrant_search_fn, filters=filters)

        if hits:
            merged.extend(hits)


    merged = dedupe_by_id(merged)
    
    # Relax relevance threshold
    merged = [
        doc for doc in merged
        if doc.get("combined", doc.get("score", 0.0)) >= 0.12
    ]


    if not merged:
        return []

    # Rerank
    reranker = _get_reranker()
    if reranker and hasattr(reranker, "predict"):
        pairs = [(query, doc.get("text", "")) for doc in merged]
        try:
            raw_scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
            norm_scores = normalize_scores_sigmoid(raw_scores if raw_scores is not None else [])
            for i, doc in enumerate(merged):
                if i < len(norm_scores):
                    doc["rank_score"] = float(norm_scores[i])
                else:
                    doc["rank_score"] = float(doc.get("combined", doc.get("score", 0.0)))
        except Exception as e:
            logger.warning(f"[WARN] Reranking failed: {e}")
            for doc in merged:
                doc["rank_score"] = float(doc.get("combined", doc.get("score", 0.0)))
    else:
        for doc in merged:
            doc["rank_score"] = float(doc.get("combined", doc.get("score", 0.0)))

    SOURCE_WEIGHTS = {
        "ESM DOCS.docx": 1.3,
        "FAQs - Additional Surveillance Measure (ASM)_14.4.25_NEWTEMP.pdf": 1.0,
        "sample_employee_data.xlsx": 0.8,
    }
    
    for doc in merged:
        rank = doc.get("rank_score", 0.0)
        if rank < 0: rank = 0.0
        combined = doc.get("combined", 0.0)
        if combined < 0: combined = 0.0
        
        src = (doc.get("source") or doc.get("document_name") or "").lower()
        w = SOURCE_WEIGHTS.get(src, 1.0)
        
        doc["weighted_score"] = (0.7 * rank) + (0.3 * combined)
        doc["weighted_score"] *= w

    merged.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
    return merged


def group_chunks_hierarchically(
    chunks: List[Dict], 
    fields: List[str], 
    leaf_processor: Callable[[List[Dict], str], Any],
    query: str
) -> Dict:
    """Recursively group chunks by fields in hierarchical order"""
    if not fields:
        return chunks

    current_field = fields[0]
    remaining_fields = fields[1:]

    grouped = defaultdict(list)
    for chunk in chunks:
        key = chunk.get(current_field, 'unknown')
        grouped[key].append(chunk)

    result = {}
    for key, group_chunks in grouped.items():
        if remaining_fields:
            # Continue grouping deeper
            result[str(key)] = {
                "metadata": {current_field: key},
                remaining_fields[0]: group_chunks_hierarchically(
                    group_chunks, remaining_fields, leaf_processor, query
                )
            }
        else:
            # Leaf level - process chunks into results
            result[str(key)] = {
                "metadata": {current_field: key},
                "items": leaf_processor(group_chunks, query)
            }

    return result


def process_group_with_llm(group_chunks: List[Dict], query: str) -> List[Dict]:
    """Process a group of chunks into final results with LLM answer"""
    # Take top 8 chunks for this group
    group_chunks.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
    top_chunks = group_chunks[:8]

    # Prepare text for LLM
    chunk_texts = [chunk.get("text", "") for chunk in top_chunks if chunk.get("text", "").strip()]

    if not chunk_texts:
        return []

    # Generate answer for this group
    answer_text = answer_from_chunks(chunk_texts, query)
    logger.info(f'''response from llm**{answer_text}**''')

    # Build structured source list
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
        return []

    # Filter out "No Data" / "No Answer" responses so they don't show in UI
    title_lower = str(answer_json.get('title', '')).lower()
    if title_lower in ["no data", "no answer"]:
        return []


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

    return [result]


def process_group_raw(group_chunks: List[Dict], query: str) -> List[Dict]:
    """Process a group of chunks and return them directly without LLM"""
    # Take top chunks for this group
    group_chunks.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
    top_chunks = group_chunks[:20]  # Return top 20 chunks if raw retrieval

    return top_chunks


def retrieve_chunks_for_streaming(
    query: str,
    bm25_service: Optional[BM25Service] = None,
    group_by: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> tuple[List[tuple], Dict[str, Any]]:
    """
    Retrieve and group chunks for streaming processing.
    Returns a flat list of (group_key, chunks) tuples for parallel processing.
    
    Args:
        query: User query string
        bm25_service: BM25 service instance
        group_by: Fields to group by (default: ['meeting_id', 'doc_type'])
        filters: Optional metadata filters
    
    Returns:
        Tuple of (chunk_groups, metadata) where:
        - chunk_groups: List of (group_key, chunk_list) tuples
        - metadata: Dict with query info and grouping metadata
    """
    if group_by is None:
        group_by = ['meeting_id', 'doc_type']
    
    if not query or not query.strip():
        return [], {"status": "error", "message": "Please enter a valid question."}
    
    # Retrieve ranked chunks
    merged = retrieve_ranked_chunks(query, bm25_service, filters)
    
    if not merged:
        return [], {"status": "no_results", "message": "No relevant information found."}
    
    # Group chunks hierarchically (just grouping, no LLM processing)
    def flatten_groups(chunks: List[Dict], fields: List[str], prefix: str = "") -> List[tuple]:
        """Recursively group and flatten into (key, chunks) tuples."""
        if not fields:
            return [(prefix.strip("_"), chunks)] if chunks else []
        
        current_field = fields[0]
        remaining_fields = fields[1:]
        
        grouped = defaultdict(list)
        for chunk in chunks:
            key = chunk.get(current_field, 'unknown')
            grouped[key].append(chunk)
        
        result = []
        for key, group_chunks in grouped.items():
            group_key = f"{prefix}{key}"
            if remaining_fields:
                result.extend(flatten_groups(group_chunks, remaining_fields, f"{group_key}_"))
            else:
                if group_chunks:
                    result.append((group_key, group_chunks))
        
        return result
    
    chunk_groups = flatten_groups(merged, group_by)
    
    metadata = {
        "status": "success",
        "grouped_by": group_by,
        "total_chunks": len(merged),
        "total_groups": len(chunk_groups)
    }
    
    logger.info(f"[STREAM] Retrieved {len(merged)} chunks grouped into {len(chunk_groups)} groups for streaming")
    
    return chunk_groups, metadata


def _calculate_group_stats(data: Dict, level: int = 0) -> Dict:
    """Recursively count groups at each level"""
    counts = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "metadata" or key == "items":
                continue
            if "items" in value:
                counts[f"level_{level}"] = counts.get(f"level_{level}", 0) + 1
            else:
                counts[f"level_{level}"] = counts.get(f"level_{level}", 0) + 1
                sub_counts = _calculate_group_stats(value, level + 1)
                for k, v in sub_counts.items():
                    counts[k] = counts.get(k, 0) + v
    return counts


def query_knowledge_base_retrieval_only(
    query: str, 
    bm25_service: Optional[BM25Service] = None, 
    group_by: Optional[List[str]] = None, 
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Query the knowledge base but return only retrieved chunks without LLM processing.
    """
    try:
        start_time = time.time()
        
        if group_by is None:
            group_by = ['meeting_id', 'doc_type']

        if not query or not query.strip():
             return {
                "status": "error",
                "message": "Please enter a valid question.",
                "grouped_by": group_by,
                "results": {},
                "summary": {"total_groups": 0, "groups_breakdown": {}},
                "processing_time": 0.0
            }

        merged = retrieve_ranked_chunks(query, bm25_service, filters)

        logger.info(f"merged {merged}")
        
        if not merged:
            return {
                "status": "no_results",
                "message": "No relevant information found.",
                "grouped_by": group_by,
                "results": {},
                "summary": {"total_groups": 0, "groups_breakdown": {}},
                "processing_time": time.time() - start_time
            }

        hierarchical_results = group_chunks_hierarchically(
            merged, group_by, leaf_processor=process_group_raw, query=query
        )
        
        results = {
            group_by[0]: hierarchical_results
        }
        
        groups_breakdown = _calculate_group_stats(hierarchical_results)
        total_groups = sum(groups_breakdown.values())
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Found results across {total_groups} group(s)",
            "grouped_by": group_by,
            "results": results,
            "summary": {
                "total_groups": total_groups,
                "groups_breakdown": groups_breakdown
            },
            "processing_time": processing_time
        }
    except Exception as e:
        logger.exception("[ERROR] Error querying knowledge base (retrieval only)")
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "grouped_by": group_by if group_by else [],
            "results": {},
            "summary": {"total_groups": 0, "groups_breakdown": {}},
            "processing_time": 0.0
        }


def query_knowledge_base_v2(query: str, bm25_service: Optional[BM25Service] = None, group_by=None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Query the knowledge base using hybrid retrieval, reranking, and LLM grounding.
    """
    try:
        start_time = time.time()

        # Default grouping
        if group_by is None:
            group_by = ['meeting_id', 'doc_type']

        if not query or not query.strip():
            return {
                "status": "error",
                "message": "Please enter a valid question.",
                "grouped_by": group_by,
                "results": {},
                "summary": {"total_groups": 0, "groups_breakdown": {}},
                "processing_time": 0.0
            }

        if (bm25_service is None or not bm25_service.docs) and count_documents() == 0:
            return {
                "status": "error",
                "message": "No documents indexed. Please upload documents first.",
                "grouped_by": group_by,
                "results": {},
                "summary": {"total_groups": 0, "groups_breakdown": {}},
                "processing_time": 0.0
            }

        merged = retrieve_ranked_chunks(query, bm25_service, filters)

        if not merged:
            return {
                "status": "no_results",
                "message": "No relevant information found. Please rephrase or try another query.",
                "grouped_by": group_by,
                "results": {},
                "summary": {"total_groups": 0, "groups_breakdown": {}},
                "processing_time": time.time() - start_time
            }

        # Hierarchical grouping with LLM processing
        hierarchical_results = group_chunks_hierarchically(
            merged, group_by, leaf_processor=process_group_with_llm, query=query
        )

        results = {
            group_by[0]: hierarchical_results
        }
        
        groups_breakdown = _calculate_group_stats(hierarchical_results)
        total_groups = sum(groups_breakdown.values())

        processing_time = time.time() - start_time
        logger.info(f"[OK] Query processed in {processing_time:.2f}s | {total_groups} total groups found.")

        return {
            "status": "success",
            "message": f"Found results across {total_groups} group(s)",
            "grouped_by": group_by,
            "results": results,
            "summary": {
                "total_groups": total_groups,
                "groups_breakdown": groups_breakdown
            },
            "processing_time": processing_time
        }

    except Exception as e:
        logger.exception("[ERROR] Error querying knowledge base")
        return {
            "status": "error",
            "message": f"An error occurred while processing your query: {str(e)}",
            "grouped_by": group_by if group_by else [],
            "results": {},
            "summary": {"total_groups": 0, "groups_breakdown": {}},
            "processing_time": 0.0
        }
