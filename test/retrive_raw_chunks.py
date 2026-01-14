# import sys
# import os
# import logging
# from typing import Optional, Dict, Any, List

# # Add backend to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from backend.services.bm25_service import BM25Service
# from backend.services.query_engine import query_knowledge_base_retrieval_only
# from backend.services.qdrant_service import ensure_collection_exists

# # Setup basic logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def get_raw_chunks(
#     query: str, 
#     bm25_service: Optional[BM25Service] = None, 
#     group_by: Optional[List[str]] = None, 
#     filters: Optional[Dict[str, Any]] = None
# ):
#     """
#     Retrieves and prints raw chunks for a given query.
#     Matches signature of query_knowledge_base_v2 but returns/prints chunks only.
#     """
#     logger.info(f"Retrieving for query: '{query}'")
    
#     # helper for default args
#     if group_by is None:
#         group_by = ['meeting_id', 'doc_type']

#     try:
#         results = query_knowledge_base_retrieval_only(
#             query=query,
#             bm25_service=bm25_service,
#             group_by=group_by,
#             filters=filters
#         )
        
#         # Simple print of chunks
#         print("\n" + "="*60)
#         print(f"RETRIEVAL RESULTS FOR: {query}")
#         print("="*60)
        
#         if results['status'] == 'success':
#             import json
#             # Print the entire JSON structure raw
#             print(json.dumps(results, indent=2, default=str))
#         else:
#             print(f"❌ No results or error: {results.get('message')}")

#     except Exception as e:
#         logger.exception("Error during retrieval execution")

# def main():
#     # 1. Initialize BM25 (Required for hybrid search)
#     index_path = os.path.join("data", "bm25_index", "bm25_index.pkl")
#     logger.info(f"Loading BM25 index from {index_path}...")
#     try:
#         # Check if file exists first to avoid confusing error
#         abs_index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', index_path))
#         if os.path.exists(abs_index_path):
#              bm25_service = BM25Service(index_path=abs_index_path)
#              logger.info("BM25 Loaded.")
#         else:
#              logger.warning(f"BM25 index not found at {abs_index_path}. Using None.")
#              bm25_service = None
#     except Exception as e:
#         logger.warning(f"Could not load BM25 index: {e}. Proceeding without BM25.")
#         bm25_service = None

#     # 2. Ensure Qdrant Connection
#     try:
#         ensure_collection_exists()
#     except Exception as e:
#         logger.error(f"Cannot connect to Qdrant: {e}. Exiting.")
#         return

#     # 3. Define Query
#     query_str = "Q4 SOX compliance dashboard"
    
#     # 4. Run
#     get_raw_chunks(
#         query=query_str,
#         filters={'meeting_id': 990},
#         group_by=['doc_type']
#     )

# if __name__ == "__main__":
#     main()
import sys
import os
import logging
from typing import Optional, Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.bm25_service import BM25Service
from backend.services.query_engine import query_knowledge_base_retrieval_only
from backend.services.qdrant_service import ensure_collection_exists

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_raw_chunks(
    query: str, 
    group_by: Optional[List[str]] = None, 
    filters: Optional[Dict[str, Any]] = None
):
    """
    Retrieves and prints raw chunks for a given query.
    Matches signature of query_knowledge_base_v2 but returns/prints chunks only.
    """
    logger.info(f"Retrieving for query: '{query}'")
    
    # helper for default args
    if group_by is None:
        group_by = ['doc_type']

    try:
        results = query_knowledge_base_retrieval_only(
            query=query,
            group_by=group_by,
            filters=filters
        )
        
        # Simple print of chunks
        print("\n" + "="*60)
        print(f"RETRIEVAL RESULTS FOR: {query}")
        print("="*60)
        
        if results['status'] == 'success':
            import json
            # Print the entire JSON structure raw
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"❌ No results or error: {results.get('message')}")

    except Exception as e:
        logger.exception("Error during retrieval execution")

def main():

    try:
        ensure_collection_exists()
    except Exception as e:
        logger.error(f"Cannot connect to Qdrant: {e}. Exiting.")
        return

   
    
    # 4. Run
    get_raw_chunks(
        query='Review of key governance outcomes for FY 2025 and regulatory updates.',
                filters={'meeting_id': 1050,'doc_type':'MOM'},
        group_by=['doc_type']
    )

if __name__ == "__main__":
    main()
