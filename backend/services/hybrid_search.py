from .bm25_service import BM25Service
import logging

logger = logging.getLogger(__name__)


def exact_match_check(docs, query_text):
    q = query_text.strip().lower()
    for d in docs:
        if q in d.get('text', '').lower():
            return {"id": d['id'], "text": d['text']}
    return None



def normalize_scores_bm25(scores):
    vals = [s['bm25_score'] for s in scores if 'bm25_score' in s]
    if not vals:
        return scores

    mn, mx = min(vals), max(vals)
    for s in scores:
        if mx == mn:
            s['bm25_norm'] = 1.0  
        else:
            s['bm25_norm'] = (s['bm25_score'] - mn) / (mx - mn)
    return scores


def normalize_scores_vector(vec_results):
    vals = [r['score'] for r in vec_results if 'score' in r]
    if not vals:
        return vec_results

    mn, mx = min(vals), max(vals)
    for r in vec_results:
        if mx == mn:
            r['vec_norm'] = 1.0  
        else:
            r['vec_norm'] = (r['score'] - mn) / (mx - mn)
    return vec_results



def hybrid_search(
    query_text,
    bm25_service: BM25Service,
    qdrant_search_fn,
    top_k=7,
    min_combined_score: float = 0.05,  # safer default
    require_both: bool = False,
    filters=None
):
    """
    Enterprise-grade hybrid search (BM25 + Vector).
    """

    logger.info(
        " Hybrid search %s",
        f"with filters {filters}" if filters else "(global search)"
    )

    bm25_results = bm25_service.query(
        query_text, top_k=top_k, filters=filters
    ) or []

    vector_results = qdrant_search_fn(
        query_text, top_k=top_k
    ) or []

    logger.info(
        f"📊 BM25={len(bm25_results)} | Vector={len(vector_results)}"
    )

    if not bm25_results and not vector_results:
        logger.warning("No search results from either engine")
        return []

    # Normalize
    bm25_res = normalize_scores_bm25(bm25_results)
    vec_res = normalize_scores_vector(vector_results)

    # Metadata lookup
    doc_meta_map = {
        doc['id']: doc for doc in getattr(bm25_service, 'docs', [])
    }

    merged = {}

    # BM25 results
    for r in bm25_res:
        base = doc_meta_map.get(r['id'], {})
        merged[r['id']] = {
            **base,
            'id': r['id'],
            'text': r.get('text') or base.get('text', ''),
            'bm25_score': r['bm25_score'],
            'bm25_norm': r.get('bm25_norm', 0.0),
            'vec_score': 0.0,
            'vec_norm': 0.0,
            'document_name': r.get('document_name') or base.get('document_name', '')
        }

    # Vector results
    for r in vec_res:
        if r['id'] in merged:
            merged[r['id']]['vec_score'] = r.get('score', 0.0)
            merged[r['id']]['vec_norm'] = r.get('vec_norm', 0.0)
        else:
            base = doc_meta_map.get(r['id'], {})
            merged[r['id']] = {
                **base,
                'id': r['id'],
                'text': r.get('text') or base.get('text', ''),
                'bm25_score': 0.0,
                'bm25_norm': 0.0,
                'vec_score': r.get('score', 0.0),
                'vec_norm': r.get('vec_norm', 0.0),
                'document_name': base.get('document_name', '')
            }


    BM25_WEIGHT = 0.6
    VEC_WEIGHT = 0.4

    merged_list = []

    for m in merged.values():
        if not m.get('text', '').strip():
            continue

        m['combined'] = (
            BM25_WEIGHT * m['bm25_norm']
            + VEC_WEIGHT * m['vec_norm']
        )

        merged_list.append(m)

        # Debug log (safe)
        logger.debug(
            f"ID={m['id']} bm25={m['bm25_norm']:.3f} "
            f"vec={m['vec_norm']:.3f} combined={m['combined']:.3f}"
        )

    # Sort
    merged_list.sort(key=lambda x: x['combined'], reverse=True)

    # Threshold filter
    merged_list = [
        m for m in merged_list
        if m['combined'] >= min_combined_score
    ]

    # Optional strictness
    if require_both:
        merged_list = [
            m for m in merged_list
            if (
                m['bm25_norm'] >= 0.08
                or m['vec_norm'] >= 0.08
            )
        ]

    logger.info(f"Final hits: {len(merged_list)}")

    return merged_list[:top_k]
