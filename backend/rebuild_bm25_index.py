"""
Rebuild BM25 index from existing Qdrant collection
This ensures BM25 and Vector search are in sync
"""
import sys
import os
import pickle
from dotenv import load_dotenv

# Add project root to path (parent of this file's directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from backend.services.qdrant_service import get_qdrant_client
from backend.services.bm25_service import BM25Service
from qdrant_client.http import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")

def get_bm25_path():
    # Base dir is project root (parent of this file's directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.getenv("BM25_INDEX_PATH")
    if env_path:
        if os.path.isabs(env_path):
            return env_path
        else:
            return os.path.join(base_dir, env_path)
    return os.path.join(base_dir, "data", "bm25_index", "bm25_index.pkl")

BM25_INDEX_PATH = get_bm25_path()


def rebuild_bm25_index():
    """Rebuild BM25 index from Qdrant collection"""
    try:
        logger.info("Connecting to Qdrant...")
        client = get_qdrant_client()
        
        # Get all points from collection
        logger.info(f"Fetching all documents from collection '{QDRANT_COLLECTION}'...")
        
        # Scroll through all points
        all_points = []
        offset = None
        batch_size = 100
        
        while True:
            result = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, offset = result
            all_points.extend(points)
            
            logger.info(f"Fetched {len(all_points)} chunks so far...")
            
            if offset is None:
                break
        
        logger.info(f"Total chunks fetched: {len(all_points)}")
        
        if not all_points:
            logger.warning("No documents found in Qdrant collection. Please process documents first.")
            return False
        
        # Build document list for BM25
        docs = []
        for point in all_points:
            doc_id = f"{point.payload.get('document_id', '')}:{point.payload.get('chunk_index', 0)}"
            doc_text = point.payload.get('text', '')
            
            if doc_text:
                docs.append({
                    'id': doc_id,
                    'text': doc_text,
                    'document_name': point.payload.get('document_name', ''),
                    'document_id': point.payload.get('document_id', ''),
                    'chunk_index': point.payload.get('chunk_index', 0),

                #     other base metadata vinayak
                    'meeting_id': point.payload.get('meeting_id', ''),
                    'doc_id': point.payload.get('doc_id', ''),
                    'doc_type':point.payload.get('doc_type', ''),
                    'department':point.payload.get('department', ''),
                    'group':point.payload.get('group', ''),
                    'meeting_date':point.payload.get('meeting_date', ''),
                    'meeting_name':point.payload.get('meeting_name', '')

                })
        
        logger.info(f"Building BM25 index with {len(docs)} document chunks...")
        
        # Delete old index if exists
        if os.path.exists(BM25_INDEX_PATH):
            os.remove(BM25_INDEX_PATH)
            logger.info(f"Removed old index file")
        
        # Create BM25 service and build index
        # We pass index_path to __init__ so it knows where to save when we call build()
        # Note: BM25Service.build calls _save(self.index_path) if self.index_path is set.
        bm25_service = BM25Service(docs=None, index_path=BM25_INDEX_PATH)
        bm25_service.build(docs)
        
        logger.info(f"✓ BM25 index rebuilt successfully!")
        logger.info(f"  Index saved to: {BM25_INDEX_PATH}")
        logger.info(f"  Total documents indexed: {len(docs)}")
        
        # Verify the index
        logger.info("\nVerifying index with test query...")
        test_results = bm25_service.query("test", top_k=3)
        logger.info(f"  Test query returned {len(test_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Error rebuilding BM25 index: {str(e)}")
        logger.exception(e)
        return False


# if __name__ == "__main__":
#     print("\n" + "="*80)
#     print("BM25 INDEX REBUILD UTILITY")
#     print("="*80 + "\n")
    
#     success = rebuild_bm25_index()
    
#     if success:
#         print("\n✅ BM25 index rebuild completed successfully!")
#         print("   You can now use the enhanced hybrid search.\n")
#     else:
#         print("\n❌ BM25 index rebuild failed. Check logs above.\n")
