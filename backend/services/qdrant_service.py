
import os
import subprocess
import time
import uuid
import logging
import platform
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.services.embedding_service import generate_embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant configuration
# QDRANT_PATH = os.getenv("QDRANT_PATH", r"/mnt/dev/qdrant")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")

client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client (starts service if needed)."""
    global client
    if client is None:
        # ensure_qdrant_running()
        client = QdrantClient(host='172.17.177.78', port='6333', timeout=10.0)
    return client

# def ensure_qdrant_running():
#     """Start Qdrant if not already running."""
#     try:
#         test_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
#         test_client.get_collections()
#         logger.info("Qdrant is running.")
#     except Exception as e:
#         logger.info(f"Qdrant not reachable ({e}). Attempting to start it from {QDRANT_PATH}...")
#         qdrant_exe = os.path.join(QDRANT_PATH, "qdrant.exe" if platform.system() == "Windows" else "qdrant")
#         creation_flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
#         try:
#             subprocess.Popen([qdrant_exe], cwd=QDRANT_PATH, creationflags=creation_flags)
#         except Exception as start_err:
#             logger.error(f"Failed to start Qdrant: {start_err}")
#             raise
#         for i in range(20):
#             time.sleep(2)
#             try:
#                 test_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
#                 test_client.get_collections()
#                 logger.info("✅ Qdrant started successfully.")
#                 return
#             except Exception:
#                 logger.info("Waiting for Qdrant to become available...")
#         raise RuntimeError("Unable to start Qdrant; please start it manually.")


def ensure_collection_exists():
    """Ensure the main collection exists and is indexed."""
    global client
    client = get_qdrant_client()
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            logger.info(f"Creating collection {QDRANT_COLLECTION}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
            )
        create_payload_indices()
        logger.info(f"Collection '{QDRANT_COLLECTION}' is ready.")
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}")
        raise


def create_payload_indices():
    """Create payload indices for searchable metadata fields."""
    try:
        indices = [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("document_name", models.PayloadSchemaType.KEYWORD),
            ("source_type", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("section_title", models.PayloadSchemaType.KEYWORD),
        #     new metadata fields vinayak
            ("doc_id", models.PayloadSchemaType.INTEGER),
            ("meeting_id", models.PayloadSchemaType.INTEGER),
            ("department", models.PayloadSchemaType.KEYWORD),
            ("group", models.PayloadSchemaType.KEYWORD),
            ("meeting_date", models.PayloadSchemaType.KEYWORD),
            ("meeting_name", models.PayloadSchemaType.KEYWORD)
        ]
        for field_name, schema in indices:
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name=field_name,
                    field_schema=schema
                )
            except UnexpectedResponse as e:
                logger.debug(f"Index {field_name} already exists or skipped: {e}")
    except Exception as e:
        logger.error(f"Failed to create payload indices: {e}")
        raise


def store_document_embeddings(document_path: str, document_name: str, chunks: List[str],
                              embeddings: Optional[List[List[float]]] = None,
                              metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Store embeddings and metadata for document chunks."""
    try:
        client = get_qdrant_client()

        if embeddings is None:
            logger.info("Generating embeddings inside qdrant_service...")
            embeddings = generate_embeddings(chunks)

        if not embeddings or len(embeddings) != len(chunks):
            logger.error("Embeddings length mismatch or empty.")
            return False

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                # change doc_id from path to doc_id passed from function vinayak
                "document_id": metadatas[i]['doc_id'],
                "document_name": document_name,
                "chunk_index": i,
                "text": chunk,
                "page_number": None,
                "sheet": None,
                "is_table": False,
            }
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])
            points.append(models.PointStruct(id=point_id, vector=emb, payload=payload))

        batch_size = 256
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            logger.info(f"Upserted points {i}-{i + len(batch) - 1}")

        logger.info(f"✅ Stored {len(points)} chunks for document: {document_name}")
        return True
    except Exception as e:
        logger.exception(f"Error storing document embeddings: {e}")
        return False


def delete_document(document_path: str,doc_id: int) -> bool:
    """Delete all embeddings for a given document."""
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=doc_id))]
                )
            )
        )
        logger.info(f"Deleted embeddings for document: {document_path}")
        return True
    except Exception as e:
        logger.exception(f"Error deleting document embeddings: {e}")
        return False


def search_similar_chunks(query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search for most similar document chunks with optional metadata filtering."""
    try:
        client = get_qdrant_client()
        
        # Build Qdrant filter conditions
        query_filter = None
        if filters:
            logger.info(f"🔍 Applying Qdrant filters: {filters}")
            must_conditions = []
            if filters.get('meeting_id') is not None:
                logger.info(f"  ➤ Filtering by meeting_id = {filters['meeting_id']}")
                must_conditions.append(
                    models.FieldCondition(
                        key="meeting_id",
                        match=models.MatchValue(value=filters['meeting_id'])
                    )
                )
            if filters.get('doc_type'):
                logger.info(f"  ➤ Filtering by doc_type = '{filters['doc_type']}'")
                must_conditions.append(
                    models.FieldCondition(
                        key="doc_type",
                        match=models.MatchValue(value=filters['doc_type'])
                    )
                )
            
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)
        else:
            logger.info("🔍 Qdrant search: No filters applied (searching all documents)")
        
        search_response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
            score_threshold=None,
        )
        search_results = search_response.points

        if not search_results:
            logger.warning("⚠️ No search results returned from Qdrant.")
            return []

        formatted_results = []
        for r in search_results:
            payload = r.payload or {}
            formatted_results.append({
                "id": getattr(r, "id", None),
                "score": getattr(r, "score", 0.0),
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "chunk_index": payload.get("chunk_index", -1),
                "text": payload.get("text", ""),
                "page_number": payload.get("page_number"),
                "section_title": payload.get("section_title"),
                "payload": payload,  # full raw payload for traceability
            })

        # Normalize vector scores 0–1
        scores = [r.get("score", 0.0) for r in formatted_results]
        if scores:
            mn, mx = min(scores), max(scores)
            for r in formatted_results:
                raw = r["score"]
                r["vec_norm"] = 0.0 if mx == mn else (raw - mn) / (mx - mn)
        else:
            for r in formatted_results:
                r["vec_norm"] = 0.0

        if filters:
            logger.info(f"✅ Retrieved {len(formatted_results)} filtered chunks (filters: {filters})")
        else:
            logger.info(f"🔍 Retrieved {len(formatted_results)} similar chunks (no filters)")
        return formatted_results

    except Exception as e:
        logger.exception(f"Error searching similar chunks: {e}")
        return []


def count_documents() -> int:
    """Check if any documents exist in the collection."""
    try:
        client = get_qdrant_client()
        result = client.scroll(collection_name=QDRANT_COLLECTION, limit=1, with_payload=False)
        return 1 if result and result[0] else 0
    except Exception as e:
        logger.exception(f"Error counting documents: {e}")
        return 0


def get_all_document_names() -> Dict[str, Dict[str, Any]]:
    """
    Get all unique document names from Qdrant collection with their metadata.
    
    Returns:
        Dict mapping document_name to metadata dict containing doc_id, meeting_id, etc.
    """
    try:
        client = get_qdrant_client()
        documents = {}
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
            
            for point in points:
                doc_name = point.payload.get('document_name', '')
                if doc_name and doc_name not in documents:
                    # Store first occurrence metadata for this document
                    documents[doc_name] = {
                        'document_name': doc_name,
                        'doc_id': point.payload.get('doc_id'),
                        'meeting_id': point.payload.get('meeting_id'),
                        'doc_type': point.payload.get('doc_type', ''),
                        'group': point.payload.get('group', ''),
                        'department': point.payload.get('department', ''),
                        'meeting_date': point.payload.get('meeting_date', ''),
                        'meeting_name': point.payload.get('meeting_name', '')
                    }
            
            if offset is None:
                break
        
        logger.info(f"Found {len(documents)} unique documents in Qdrant")
        return documents
        
    except Exception as e:
        logger.exception(f"Error getting document names from Qdrant: {e}")
        return {}


def get_document_info(document_name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific document.
    
    Args:
        document_name: Name of the document
        
    Returns:
        Dictionary with document metadata or None if not found
    """
    try:
        client = get_qdrant_client()
        
        # Search for any point with this document_name
        result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_name",
                        match=models.MatchValue(value=document_name)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        points, _ = result
        
        if points:
            payload = points[0].payload
            return {
                'document_name': payload.get('document_name', ''),
                'doc_id': payload.get('doc_id'),
                'meeting_id': payload.get('meeting_id'),
                'doc_type': payload.get('doc_type', ''),
                'group': payload.get('group', ''),
                'department': payload.get('department', ''),
                'meeting_date': payload.get('meeting_date', ''),
                'meeting_name': payload.get('meeting_name', '')
            }
        
        return None
        
    except Exception as e:
        logger.exception(f"Error getting document info: {e}")
        return None

def delete_documents_by_metadata(meeting_id: int, doc_type: str) -> bool:
    """Delete all embeddings for a given meeting_id and doc_type."""
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="meeting_id", match=models.MatchValue(value=meeting_id)),
                        models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))
                    ]
                )
            )
        )
        logger.info(f"Deleted embeddings for Meeting {meeting_id}, DocType {doc_type}")
        return True
    except Exception as e:
        logger.exception(f"Error deleting document embeddings by metadata: {e}")
        return False




def get_existing_doc_ids(meeting_id: int, doc_type: str) -> set:
    """
    Get set of existing document IDs for a given meeting and doc type.
    """
    try:
        client = get_qdrant_client()
        doc_ids = set()
        offset = None
        
        # Filter by meeting_id and doc_type
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(key="meeting_id", match=models.MatchValue(value=meeting_id)),
                models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))
            ]
        )
        
        while True:
            # We only need payload fields 'doc_id' or 'document_id'
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=["document_id", "doc_id"], 
                with_vectors=False
            )
            
            for point in points:
                # Prefer 'doc_id' (int) but fallback to 'document_id'
                did = point.payload.get("doc_id")
                if did is None:
                    did = point.payload.get("document_id")
                
                if did is not None:
                    doc_ids.add(did)
            
            if offset is None:
                break
                
        return doc_ids
    except Exception as e:
        logger.exception(f"Error getting existing doc IDs: {e}")
        return set()
