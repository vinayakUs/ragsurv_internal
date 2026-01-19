import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import requests
import oracledb
import uvicorn
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)
from fastapi import FastAPI, HTTPException, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tools.doc_type import DocType
from fastapi.responses import StreamingResponse
import io
import base64
from pathlib import Path
from backend.rebuild_bm25_index import rebuild_bm25_index


# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Imports from api.py ---
from tools.oracle_client import OracleClient

# --- Imports from main.py ---
from backend.services.document_processor import process_document, fetch_documents_from_service , save_single_document_to_disk
from backend.services.hybrid_search import hybrid_search
from backend.services.bm25_service import BM25Service
from backend.services.query_engine import query_knowledge_base_v2
from backend.services.llm_service import ensure_model_loaded
from backend.services.qdrant_service import ensure_collection_exists
from backend.services.parallel_query_service import process_batch_queries, validate_batch_request
from backend.services.query_engine import retrieve_chunks_for_streaming
from backend.services.streaming_query_service import create_streaming_response
from backend.services.input_guardrail import validate_query, validate_query_batch as validate_queries_batch
from backend.services.intent_validator import validate_query_intent, validate_query_intent_batch
from backend.services.structured_logger import (
    get_structured_logger,
    generate_request_id,
    set_request_context,
    clear_request_context,
    Timer,
    SERVER_ID
)

# Structured logger instance
slog = get_structured_logger("server")

# Import caching service
try:
    from backend.services.cache_service import (
        cache_query_result, 
        get_cached_query_result, 
        get_cache_stats,
        clear_all_caches
    )
    CACHING_ENABLED = True
except ImportError:
    CACHING_ENABLED = False

# Import session-based caching service
try:
    from backend.services.session_cache_service import (
        cache_for_session,
        get_from_session,
        get_session_stats,
        get_all_sessions_stats,
        clear_session_cache,
        cleanup_expired_sessions
    )
    SESSION_CACHING_ENABLED = True
except ImportError:
    SESSION_CACHING_ENABLED = False

# ---------------------------------------------
# Logging setup
# ---------------------------------------------
# ---------------------------------------------
# Logging setup
# ---------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(LOGS_DIR, "server.log"), encoding="utf-8")]
)
logger = logging.getLogger("server")

# ---------------------------------------------
# Query Logger - logs ALL user queries to logs/query.log
# ---------------------------------------------
query_logger = logging.getLogger("query_log")
query_logger.setLevel(logging.INFO)
query_logger.propagate = False  # Don't send to root logger

query_file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "query.log"), encoding="utf-8")
query_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
query_logger.addHandler(query_file_handler)

# ---------------------------------------------
# Static and Data directory setup
# ---------------------------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Initialize BM25 Index
bm25_service = BM25Service(index_path=os.path.join("data", "bm25_index", "bm25_index.pkl"))

# ─────────────────────────────────────────────
# ✅ FastAPI app init
# ─────────────────────────────────────────────
app = FastAPI(title="Unified Knowledge Base & Meeting API", version="1.0")

# ─────────────────────────────────────────────
# ✅ CORS setup
# ─────────────────────────────────────────────
origins = ["*"]  # Replace with specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------
# Request Tracking Middleware
# ---------------------------------------------
from starlette.requests import Request

@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Extract session ID, generate request ID, and set logging context."""
    # Generate unique request ID
    request_id = generate_request_id()
    session_id = request.headers.get("X-Session-ID", "anonymous")
    
    # Set context for structured logging
    set_request_context(request_id=request_id, session_id=session_id)
    
    # Attach to request state
    request.state.session_id = session_id
    request.state.request_id = request_id
    
    # Log request start
    slog.info("REQUEST_START", method=request.method, path=request.url.path)
    
    try:
        response = await call_next(request)
        slog.info("REQUEST_END", status_code=response.status_code)
        return response
    except Exception as e:
        slog.exception("REQUEST_ERROR", error=str(e)[:50])
        raise
    finally:
        clear_request_context()

# ---------------------------------------------
# Startup event
# ---------------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing Qdrant collection...")
        ensure_collection_exists()

        logger.info("Checking LLM model availability...")
        ensure_model_loaded()
        logger.info("[OK] LLM model loaded successfully.")

        # Pre-load embedding model at startup for faster first request
        logger.info("Pre-loading embedding model...")
        from backend.services.embedding_service import get_embedding_model
        get_embedding_model()
        logger.info("[OK] Embedding model pre-loaded successfully.")


        from backend.services.event_scheduler import start_scheduler
        logger.info("Starting Event Scheduler...")
        start_scheduler()

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

# ─────────────────────────────────────────────
# ✅ Static files mount
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─────────────────────────────────────────────
# ✅ Helper: Execute Oracle Query (MOVED TO TOOLS)
# ─────────────────────────────────────────────

# execute_query and execute_write have been moved to tools.oracle_client.OracleClient
# normalize_row kept here as it's a specific helper for kpis


def normalize_row(row):
    """Convert all keys to uppercase for consistent access."""
    return {k.upper(): v for k, v in row.items()}

# ─────────────────────────────────────────────
# ✅ Models (Merged)
# ─────────────────────────────────────────────

# --- Meeting Models ---
class Meeting(BaseModel):
    meeting_id: int
    meeting_name: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    group_name: Optional[str] = None
    department: Optional[str] = None
    venue: Optional[str] = None
    stage: Optional[str] = None
    role: Optional[str] = None
    created_date: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_date: Optional[datetime] = None
    updated_by: Optional[str] = None
    is_active: Optional[str] = None

class MeetingDocument(BaseModel):
    id: int
    meeting_id: Optional[int] = None
    document_name: Optional[str] = None
    document_path: Optional[str] = None
    is_archive: Optional[str] = None
    created_date: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_date: Optional[datetime] = None
    updated_by: Optional[str] = None
    doc_type: Optional[str] = None

class AgendaStatusModel(BaseModel):
    meeting_id: str
    status: List[dict]   # list of dictionaries

class DgmUser(BaseModel):
    id: int
    role_id: Optional[int] = None
    organization_id: Optional[int] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[int] = None
    email: Optional[str] = None
    passkey: Optional[str] = None
    is_archive: Optional[str] = None
    created_date: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_date: Optional[datetime] = None
    updated_by: Optional[str] = None
    otp: Optional[int] = None

# --- RAG/KB Models ---
class QueryRequest(BaseModel):
    query: str
    meeting_id: Optional[int] = None
    doc_type: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[str]
    source:str
    title:str
    score:str
    
class LocalDocumentInfo(BaseModel):
    filename: str
    path: str
    last_modified: Optional[str] = None

class DocumentRequest(BaseModel):
    meeting_id: int
    doc_type: DocType

class BatchQueryRequest(BaseModel):
    queries: List[str]
    meeting_id: Optional[int] = None
    doc_type: Optional[str] = None
    per_query_filters: Optional[List[dict]] = None

class BatchQueryResponse(BaseModel):
    results: List[dict]
    total_queries: int
    successful: int
    failed: int
    processing_time: float

# ---------------------------------------------
# Meeting / Oracle APIs
# ---------------------------------------------


@app.get("/meetings", response_model=List[Meeting])
def get_meetings(page: int = Query(None), limit: int = Query(None)):
    """
    Fetch meeting records from Oracle.
    
    If page and limit are provided, returns paginated results.
    If not provided, returns ALL meetings (for client-side pagination).
    """
    if page is not None and limit is not None:
        # Server-side pagination
        offset = (page - 1) * limit
        query = """
            SELECT * FROM MEETING_DTLS_SURV 
            ORDER BY scheduled_at DESC
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
        """
        return execute_query(query, {"offset": offset, "limit": limit})
    else:
        # Return all meetings (for client-side pagination)
        query = """
            SELECT * FROM MEETING_DTLS_SURV 
            ORDER BY scheduled_at DESC
        """
        return execute_query(query)

@app.get("/meetings/search", response_model=List[Meeting])
def search_meetings(
    meeting_id: Optional[int] = Query(None, description="Filter by meeting ID"),
    name: Optional[str] = Query(None, description="Search by meeting name (partial match)"),
    stage: Optional[str] = Query(None, description="Filter by stage"),
    group_name: Optional[str] = Query(None, description="Filter by group name"),
    department: Optional[str] = Query(None, description="Filter by department"),
    date: Optional[str] = Query(None, description="Search by scheduled date (YYYY-MM-DD)")
):
    """
    Search meetings by meeting_id, name, stage, group_name, department, or date.
    """
    query = "SELECT * FROM MEETING_DTLS_SURV WHERE 1=1"
    params = {}

    if meeting_id:
        query += " AND meeting_id = :meeting_id"
        params["meeting_id"] = meeting_id
    if name:
        query += " AND LOWER(meeting_name) LIKE :name"
        params["name"] = f"%{name.lower()}%"
    if stage:
        query += " AND LOWER(stage) = :stage"
        params["stage"] = stage.lower()
    if group_name:
        query += " AND LOWER(group_name) LIKE :group_name"
        params["group_name"] = f"%{group_name.lower()}%"
    if department:
        query += " AND LOWER(department) LIKE :department"
        params["department"] = f"%{department.lower()}%"
    if date:
        query += " AND TRUNC(scheduled_at) = TO_DATE(:date, 'YYYY-MM-DD')"
        params["date"] = date

    query += " ORDER BY scheduled_at DESC"
    try:
        return OracleClient().execute_query_formatted(query, params)
    except Exception as e:
        logger.error(f"Error searching meetings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meetings/{meeting_id}", response_model=Meeting)
def get_meeting_by_id(meeting_id: int):
    """
    Get details of a single meeting by ID.
    """
    query = "SELECT * FROM MEETING_DTLS_SURV WHERE meeting_id = :meeting_id"
    try:
        results = OracleClient().execute_query_formatted(query, {"meeting_id": meeting_id})
        if not results:
            raise HTTPException(status_code=404, detail="Meeting not found")
        return results[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching meeting {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/users", response_model=List[DgmUser])
# def get_all_users():
#     """
#     Fetch all users (only limited columns from tbl_dgm_usr)
#     """
#     query = """
#         SELECT id, role_id, organization_id, first_name, last_name,
#                phone, email, passkey, is_archive, created_date,
#                created_by, updated_date, updated_by, otp
#         FROM tbl_users
#         ORDER BY id
#     """
#     try:
#         return OracleClient().execute_query_formatted(query)
#     except Exception as e:
#         logger.error(f"Error fetching users: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/users/search", response_model=List[DgmUser])
# def search_users(
#     id: Optional[int] = Query(None, description="Search by user ID"),
#     first_name: Optional[str] = Query(None, description="Search by first name"),
#     last_name: Optional[str] = Query(None, description="Search by last name"),
#     email: Optional[str] = Query(None, description="Search by email"),
#     role_id: Optional[int] = Query(None, description="Filter by role ID"),
#     organization_id: Optional[int] = Query(None, description="Filter by organization ID")
# ):
#     """
#     Search tbl_users by ID, name, email, role, or organization.
#     Only limited fields are returned.
#     """
#     query = """
#         SELECT id, role_id, organization_id, first_name, last_name,
#                phone, email, passkey, is_archive, created_date,
#                created_by, updated_date, updated_by, otp
#         FROM tbl_users
#         WHERE 1=1
#     """
#     params = {}

#     if id:
#         query += " AND id = :id"
#         params["id"] = id
#     if first_name:
#         query += " AND LOWER(first_name) LIKE :first_name"
#         params["first_name"] = f"%{first_name.lower()}%"
#     if last_name:
#         query += " AND LOWER(last_name) LIKE :last_name"
#         params["last_name"] = f"%{last_name.lower()}%"
#     if email:
#         query += " AND LOWER(email) LIKE :email"
#         params["email"] = f"%{email.lower()}%"
#     if role_id:
#         query += " AND role_id = :role_id"
#         params["role_id"] = role_id
#     if organization_id:
#         query += " AND organization_id = :organization_id"
#         params["organization_id"] = organization_id

#     query += " ORDER BY created_date DESC"
#     try:
#         return OracleClient().execute_query_formatted(query, params)
#     except Exception as e:
#         logger.error(f"Error searching users: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/users/{user_id}", response_model=DgmUser)
# def get_user_by_id(user_id: int):
#     """
#     Get a single user by ID (only limited fields)
#     """
#     query = """
#         SELECT id, role_id, organization_id, first_name, last_name,
#                phone, email, passkey, is_archive, created_date,
#                created_by, updated_date, updated_by, otp
#         FROM tbl_users
#         WHERE id = :id
#     """
#     try:
#         results = OracleClient().execute_query_formatted(query, {"id": user_id})
#         if not results:
#             raise HTTPException(status_code=404, detail="User not found")
#         return results[0]
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching user {user_id}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# --- Documents (DB) ---

# @app.get("/documents", response_model=List[MeetingDocument])
# def get_all_documents():
#     """
#     Fetch all documents from the documents table.
#     """
#     query = """
#         SELECT 
#             ID as id, 
#             MEETING_ID as meeting_id, 
#             DOCUMENT_NAME as document_name, 
#             DOCUMENT_PATH as document_path,
#             IS_ARCHIVE as is_archive, 
#             CREATED_DATE as created_date, 
#             CREATED_BY as created_by, 
#             UPDATED_DATE as updated_date, 
#             UPDATED_BY as updated_by, 
#             CATEGORY_NAME as doc_type
#         FROM agenda_dtls_surv
#         ORDER BY created_date DESC
#     """
#     try:
#         return OracleClient().execute_query_formatted(query)
#     except Exception as e:
#         logger.error(f"Error fetching documents: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/search", response_model=List[MeetingDocument])
def search_documents(
    id: Optional[int] = Query(None, description="Search by document ID"),
    meeting_id: Optional[int] = Query(None, description="Search by meeting ID"),
    category: Optional[int] = Query(None, description="Filter by category"),
    name: Optional[str] = Query(None, description="Search by document name (partial match)"),
    doc_type: Optional[str] = Query(None, description="Search by document type"),
    is_archive: Optional[str] = Query(None, description="Filter by archive flag (0 or 1)")
):
    """
    Search documents by ID, meeting ID, category ID, or document name.
    """
    query = """
        SELECT 
            ID as id, 
            MEETING_ID as meeting_id, 
            DOCUMENT_NAME as document_name, 
            DOCUMENT_PATH as document_path,
            IS_ARCHIVE as is_archive, 
            CREATED_DATE as created_date, 
            CREATED_BY as created_by, 
            UPDATED_DATE as updated_date, 
            UPDATED_BY as updated_by, 
            CATEGORY_NAME as doc_type
        FROM agenda_dtls_surv
        WHERE 1=1
    """
    params = {}

    if id:
        query += " AND id = :id"
        params["id"] = id
    if meeting_id:
        query += " AND meeting_id = :meeting_id"
        params["meeting_id"] = meeting_id
    if name:
        query += " AND LOWER(document_name) LIKE :name"
        params["name"] = f"%{name.lower()}%"
    if doc_type:
        query += " AND LOWER(CATEGORY_NAME) LIKE :doc_type"
        params["doc_type"] = f"%{doc_type.lower()}%"
    if is_archive:
        query += " AND is_archive = :is_archive"
        params["is_archive"] = is_archive

    query += " ORDER BY created_date DESC"
    try:
        return OracleClient().execute_query_formatted(query, params)
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}", response_model=MeetingDocument)
def get_document_by_id(doc_id: int):
    """
    Get a single document by ID.
    """
    query = """
        SELECT id, meeting_id, category_id, category_name, stage, document_name, document_path,
               is_archive, created_date, created_by, updated_date, updated_by
        FROM agenda_dtls_surv
        WHERE id = :id
    """
    try:
        results = OracleClient().execute_query_formatted(query, {"id": doc_id})
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        return results[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/documents/download/")
# def download_document(meeting_id: int, doc_type: str):

#     response = requests.get(
#         "https://survportaluat.nse.co.in/surv/document-service/document/getdocument",
#         params={"meetingId": meeting_id, "categoryId": doc_type},
#         verify=False,
#         timeout=60
#     )

#     if response.status_code != 200:
#         raise HTTPException(
#             status_code=502,
#             detail=f"Document service failed ({response.status_code})"
#         )

#     payload = response.json()

#     pdf_content = payload["result"][0]["base64"]

#     return StreamingResponse(
#         io.BytesIO(base64.b64decode(pdf_content)),
#         media_type="application/pdf",
#         headers={
#             "Content-Disposition": f'attachment; filename="{payload["result"][0]["documentName"]}"',
#             "X-Doc-Id": str(payload["result"][0]["id"])
#         }
#     )

@app.get("/downloadDocument")
def download_document_get(meeting_id: int, doc_type: str):
    doc = fetch_document_bytes(meeting_id, doc_type)

    return StreamingResponse(
        io.BytesIO(doc["bytes"]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{doc["filename"]}"',
            "X-Doc-Id": str(doc["doc_id"]),
        }
    )
def fetch_document_bytes(meeting_id: int, doc_type: str) -> dict:
    response = requests.get(
        "https://survportaluat.nse.co.in/surv/document-service/document/getdocument",
        params={"meetingId": meeting_id, "categoryId": doc_type},
        timeout=60,verify=False
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail="Document service failed"
        )

    result = response.json()["result"][0]

    return {
        "bytes": base64.b64decode(result["base64"]),
        "filename": result["documentName"],
        "doc_id": result["id"],
    }

@app.get("/participants/{meeting_id}")
def get_participant_by_id(meeting_id: int):
    """
    Get a List of participants by meeting id.
    """
    query = """select * from user_dtls_surv m join tbl_users u on m.user_id = u.id  where m.meeting_id = :meeting_id"""
    try:
        results = OracleClient().execute_query_formatted(query, {"meeting_id": meeting_id})
        if not results:
            raise HTTPException(status_code=404, detail="Participants not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching participants for {meeting_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agenda_status/{meeting_id}")
def save_agenda_status(meeting_id: str, payload: AgendaStatusModel):
    """
    Save agenda status list (list of dicts) as JSON in Oracle CLOB.
    """
    status_json = json.dumps(payload.status)

    query = """
        MERGE INTO agenda_status tgt
        USING (SELECT :meeting_id AS meeting_id FROM dual) src
        ON (tgt.meeting_id = src.meeting_id)
        WHEN MATCHED THEN 
            UPDATE SET status = :status
        WHEN NOT MATCHED THEN
            INSERT (meeting_id, status)
            VALUES (:meeting_id, :status)
    """
    try:
        OracleClient().execute_write(query, {"meeting_id": meeting_id, "status": status_json})
    except Exception as e:
        logger.error(f"Error saving agenda status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Agenda status saved", "meeting_id": meeting_id}

def execute_query(query: str, params: dict = None):
    try:
        client = OracleClient()
        # Use new method that returns dicts with lowercase keys
        rows = client.execute_query_dicts(query, params)
        
        # Post-process for datetime isoformat if needed 
        for record in rows:
            for k, v in record.items():
                if hasattr(v, "isoformat"):
                    record[k] = v.isoformat()
        return rows

    except Exception as e:
        logger.error(f"Database query error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/agenda_status/{meeting_id}")
def get_agenda_status(meeting_id: str):
    """
    Retrieve all agenda items with their status.
    """
    query = """
        SELECT id, meeting_id, item_text, status, created_at, updated_at ,summary
        FROM MEETING_AGENDA_ITEMS
        WHERE meeting_id = :meeting_id
        ORDER BY id ASC
    """
    # Note: assumed 'id' column exists for ordering, if not use created_at
    
    try:
        rows = execute_query(query, {"meeting_id": meeting_id})
        
        # Post-process CLOBs if necessary (OracleClient usually handles it but let's be safe)
        results = []
        for row in rows:
            item = {
                "id": row.get("id"),
                "meeting_id": row.get("meeting_id"),
                "item_text": row.get("item_text"),
                "status": row.get("status"),
                "summary": row.get("summary"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }
            # Handle CLOBs
            if hasattr(item["item_text"], "read"):
                item["item_text"] = item["item_text"].read()
            if hasattr(item["status"], "read"):
                item["status"] = item["status"].read()
            
            results.append(item)

        return results
        
    except Exception as e:
        logger.error(f"Get status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get('/dashboard/home/kpis')
def get_dashboard_kpis():
    """
    Retrieve dashboard card data for the admin dashboard.
    """
    queries = {
        "quarter_meeting": """
            WITH cnt_cte AS (
                SELECT 
                    TO_CHAR(TRUNC(SCHEDULED_AT, 'Q'), 'YYYY-Q') AS quarter,
                    COUNT(DISTINCT MEETING_ID) AS cnt
                FROM MEETING_DTLS_SURV
                GROUP BY TO_CHAR(TRUNC(SCHEDULED_AT, 'Q'), 'YYYY-Q')
                ORDER BY quarter DESC
            )
            SELECT 
                cnt AS met_cnt_curr, 
                cnt - LAG(cnt) OVER (ORDER BY quarter) AS prv_diff
            FROM cnt_cte
            ORDER BY quarter DESC
            FETCH FIRST 1 ROW ONLY
        """,
        "atr_count": """
            SELECT COUNT(DOCUMENT_NAME) AS tot_atr_cnt 
            FROM AGENDA_DTLS_SURV
            WHERE UPPER(CATEGORY_NAME) = 'ATR'
        """,
        "committee_member_count": """
            SELECT COUNT(DISTINCT id) AS cmte_mem_cnt
            FROM tbl_users
        """,
        "committee_total_count": """
            SELECT COUNT(DISTINCT GROUP_NAME) AS cmte_tot_cnt
            FROM MEETING_DTLS_SURV
        """,
        "completed_total_count": """
            SELECT COUNT(DOCUMENT_NAME) AS cmpl_tot_cnt
            FROM AGENDA_DTLS_SURV
        """
    }

    def run_query(key):
        try:
            result = OracleClient().execute_query_formatted(queries[key])
        except Exception as e:
            logger.error(f"Error processing KPI {key}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
        if not result or len(result) == 0:
            raise HTTPException(404, f"Info unavailable for {key}")
        return result[0]

    q1 = normalize_row(run_query("quarter_meeting"))
    q2 = normalize_row(run_query("atr_count"))
    q3 = normalize_row(run_query("committee_member_count"))
    q4 = normalize_row(run_query("committee_total_count"))
    q5 = normalize_row(run_query("completed_total_count"))

    def safe(value):
        return value if value is not None else 0

    return {
        "quarter_meeting": {
            "current_quarter_count": safe(q1.get("MET_CNT_CURR")),
            "previous_quarter_diff": safe(q1.get("PRV_DIFF"))
        },
        "atr_summary": {
            "total_atr_count": safe(q2.get("TOT_ATR_CNT"))
        },
        "committee_summary": {
            "member_count": safe(q3.get("CMTE_MEM_CNT")),
            "total_committees": safe(q4.get("CMTE_TOT_CNT"))
        },
        "documents_summary": {
            "completed_total_count": safe(q5.get("CMPL_TOT_CNT"))
        }
    }

@app.get("/meeting/monthly-count")
def get_monthly_meeting_count():
    query = """
        SELECT
            TO_CHAR(SCHEDULED_AT, 'YYYY-MM') AS MONTH_KEY,
            COUNT(*) AS MEETING_COUNT
        FROM
            MEETING_DTLS_SURV
        WHERE
            SCHEDULED_AT >= ADD_MONTHS(TRUNC(SYSDATE, 'MM'), -6)
           
        GROUP BY
            TO_CHAR(SCHEDULED_AT, 'YYYY-MM')
        ORDER BY
            MONTH_KEY
    """
    try:
        rows = OracleClient().execute_query_formatted(query)
    except Exception as e:
        logger.error(f"Error fetching monthly meeting count: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    # print("DEBUG ROW KEYS:", rows[0].keys()) 
    return rows


from fastapi import Query, HTTPException

@app.get("/meetings/summary")
def get_meeting_summaries(
    page: int | None = Query(None, ge=1),
    limit: int | None = Query(None, ge=1),
):
    """
    Fetch meeting summaries from meeting_summary_surv.

    Response format:
    {
      "page": 1,
      "limit": 10,
      "count": 10,
      "total": 123,
      "data": [...]
    }
    """

    # Enforce correct pagination usage
    if (page is None) != (limit is None):
        raise HTTPException(
            status_code=400,
            detail="Both page and limit must be provided together",
        )

    try:
        # Total count query (always executed)
        count_query = """
            SELECT COUNT(*) AS total
            FROM meeting_summary_surv
        """
        total_rows = execute_query(count_query)
        logger.info(total_rows)
        total = total_rows[0]["total"]

        if page is not None and limit is not None:
            offset = (page - 1) * limit
            data_query = """
                SELECT meeting_id, MEETING_NAME, created_at, updated_at
                FROM meeting_summary_surv
                ORDER BY meeting_id ASC
                OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
            """
            rows = execute_query(
                data_query,
                {"offset": offset, "limit": limit},
            )
        else:
            # No pagination → return all
            data_query = """
                SELECT meeting_id, MEETING_NAME, created_at, updated_at
                FROM meeting_summary_surv
                ORDER BY meeting_id ASC
            """
            rows = execute_query(data_query)
            page = 1
            limit = len(rows)

        data = [
            {
                "meeting_id": row.get("meeting_id"),
                "meeting_name": row.get("MEETING_NAME"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            }
            for row in rows
        ]

        return {
            "page": page,
            "limit": limit,
            "count": len(data),
            "total": total,
            "data": data,
        }

    except Exception as e:
        logger.error(f"Error fetching meeting summaries: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch meeting summaries")


@app.get("/meetings/summary/{meeting_id}")
def get_monthly_meeting_count(meeting_id:str):
    query = """
        select meeting_id , summary_json , created_at , updated_at from meeting_summary_surv 
        WHERE meeting_id = :meeting_id
        ORDER BY meeting_id ASC
    """
    # Note: assumed 'id' column exists for ordering, if not use created_at
    
    try:
        rows = execute_query(query, {"meeting_id": meeting_id})
        
        # Post-process CLOBs if necessary (OracleClient usually handles it but let's be safe)
        results = []
        for row in rows:
            item = {
                "meeting_id": row.get("meeting_id"),
                "summary_json": json.loads(row.get("summary_json")),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }
            # Handle CLOBs
            if hasattr(item["summary_json"], "read"):
                item["summary_json"] = item["summary_json"].read()
             
            results.append(item)

        return results[0]
        
    except Exception as e:
        logger.error(f"Get status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ─────────────────────────────────────────────
# ✅ RAG / Knowledge Base APIs
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "✅ Unified Knowledge Base & Meeting API is running successfully."}

@app.get("/health")
async def health():
    """Enhanced health check with cache and session stats."""
    ok = ensure_model_loaded()
    health_info = {"llm_status": "ok" if ok else "error"}
    
    if CACHING_ENABLED:
        health_info["global_cache_stats"] = get_cache_stats()
    
    if SESSION_CACHING_ENABLED:
        health_info["session_stats"] = get_all_sessions_stats()
    
    return health_info

@app.get("/session/stats")
async def session_stats(req: Request):
    """Get cache statistics for the current session."""
    if not SESSION_CACHING_ENABLED:
        raise HTTPException(status_code=503, detail="Session caching not enabled")
    
    session_id = getattr(req.state, "session_id", "anonymous")
    return get_session_stats(session_id)

@app.delete("/session/cache")
async def clear_session(req: Request):
    """Clear the cache for the current session."""
    if not SESSION_CACHING_ENABLED:
        raise HTTPException(status_code=503, detail="Session caching not enabled")
    
    session_id = getattr(req.state, "session_id", "anonymous")
    clear_session_cache(session_id)
    return {"message": "Session cache cleared", "session_id": session_id[:8] + "..."}

@app.post("/validate-query")
async def validate_query_endpoint(request: QueryRequest):
    """
    Test a query against all validation layers WITHOUT executing RAG.
    
    Useful for:
    - Testing queries before submission
    - Debugging blocked queries
    - Understanding why a query was rejected
    
    Returns validation status for each layer:
    - security_check: SQL injection, XSS, etc.
    - intent_check: Organizational vs off-topic
    """
    result = {
        "query": request.query,
        "security_check": {"passed": True, "error": None},
        "intent_check": {"passed": True, "error": None, "intent": None},
        "overall": "VALID"
    }
    
    # Check 1: Empty query
    if not request.query or not request.query.strip():
        result["overall"] = "BLOCKED"
        result["security_check"] = {"passed": False, "error": "Query cannot be empty"}
        return result
    
    # Check 2: Security guardrail
    is_safe, guardrail_error = validate_query(request.query)
    if not is_safe:
        result["security_check"] = {"passed": False, "error": guardrail_error}
        result["overall"] = "BLOCKED"
        return result
    
    # Check 3: Intent validation
    is_valid_intent, intent_error, intent = validate_query_intent(request.query)
    result["intent_check"]["intent"] = intent
    if not is_valid_intent:
        result["intent_check"]["passed"] = False
        result["intent_check"]["error"] = intent_error
        result["overall"] = "BLOCKED"
        return result
    
    return result


@app.post("/query")
async def query(request: QueryRequest, req: Request):
    """
    Query knowledge base with real-time streaming of results.
    Uses Server-Sent Events (SSE) for progressive delivery.
    
    Each chunk is processed in parallel and streamed immediately upon completion.
    Events:
    - 'start': Initial event with total_chunks count
    - 'chunk': Individual chunk result with chunk_index for ordering
    - 'done': Final event indicating all chunks processed
    
    Optional filters: meeting_id, doc_type
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
        # Input guardrail - validate query for security threats
        is_safe, guardrail_error = validate_query(request.query)
        if not is_safe:
            logger.warning(f"[GUARDRAIL] Blocked query: {request.query[:50]}...")
            raise HTTPException(status_code=400, detail=guardrail_error)
        
        # Intent validation - ensure query is relevant to organizational content
        is_valid_intent, intent_error, intent = validate_query_intent(request.query)
        if not is_valid_intent:
            logger.warning(f"[INTENT] Rejected query ({intent}): {request.query[:50]}...")
            raise HTTPException(status_code=400, detail=intent_error)
        
        # Get context from middleware
        session_id = getattr(req.state, "session_id", "anonymous")
        request_id = getattr(req.state, "request_id", "-")
        
        # Build filters dict from request
        filters = {}
        if request.meeting_id is not None:
            filters['meeting_id'] = request.meeting_id
        if request.doc_type:
            filters['doc_type'] = request.doc_type
        
        # Structured logging
        slog.info("QUERY_START", query=request.query[:50], filters=str(filters) if filters else "none")
        
        # Log query to dedicated query log file
        query_logger.info(
            f"SESSION={session_id[:8]}... | "
            f"MEETING={request.meeting_id or 'N/A'} | "
            f"DOC_TYPE={request.doc_type or 'ALL'} | "
            f"QUERY={request.query}"
        )
        
        # Retrieve and group chunks (fast - no LLM yet)
        with Timer() as t:
            chunk_groups, metadata = retrieve_chunks_for_streaming(
                query=request.query,
                bm25_service=bm25_service,
                filters=filters
            )
        
        if not chunk_groups:
            slog.info("QUERY_NO_RESULTS", duration_ms=t.duration_ms)
            # Return error/no_results as SSE
            import json
            error_event = f"event: error\ndata: {json.dumps(metadata)}\n\n"
            return StreamingResponse(
                iter([error_event]),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        
        slog.info("RETRIEVAL_COMPLETE", chunks=len(chunk_groups), duration_ms=t.duration_ms)
        
        # Create streaming response with parallel chunk processing
        return StreamingResponse(
            create_streaming_response(chunk_groups, request.query, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        slog.exception("QUERY_ERROR", error=str(e)[:50])
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query/batch")
async def query_batch(request: BatchQueryRequest):
    """
    Process multiple queries in parallel.
    Returns results in the same order as queries were provided.
    
    Example request:
    {
        "queries": ["What is ESM?", "What is ATR?", "List committees"],
        "meeting_id": 123,  // optional global filter
        "doc_type": "AGENDA"  // optional global filter
    }
    """
    try:
        # Validate request structure
        is_valid, error_msg = validate_batch_request(request.queries)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Input guardrail - validate all queries for security threats
        is_safe, guardrail_error, failed_idx = validate_queries_batch(request.queries)
        if not is_safe:
            logger.warning(f"[GUARDRAIL] Blocked batch query at index {failed_idx}")
            raise HTTPException(status_code=400, detail=guardrail_error)
        
        # Intent validation - ensure all queries are relevant
        is_valid_intent, intent_error, intent_idx, intent = validate_query_intent_batch(request.queries)
        if not is_valid_intent:
            logger.warning(f"[INTENT] Rejected batch query at index {intent_idx} ({intent})")
            raise HTTPException(status_code=400, detail=intent_error)
        
        # Build filters
        global_filters = {}
        if request.meeting_id is not None:
            global_filters['meeting_id'] = request.meeting_id
        if request.doc_type:
            global_filters['doc_type'] = request.doc_type
        
        # Use per-query filters if provided, otherwise use global filters for all
        if request.per_query_filters:
            filters_list = request.per_query_filters
        else:
            filters_list = [global_filters] * len(request.queries)
        
        logger.info(f"[BATCH] Batch query request: {len(request.queries)} queries")
        if global_filters:
            logger.info(f"Global filters: {global_filters}")
        
        # Process queries in parallel
        import time
        start_time = time.time()
        results = await process_batch_queries(
            queries=request.queries,
            bm25_service=bm25_service,
            filters=filters_list
        )
        processing_time = time.time() - start_time
        
        # Count successes and failures
        successful = sum(1 for r in results if r.get("status") == "success")
        failed = len(results) - successful
        
        logger.info(
            f"[OK] Batch query completed: {successful}/{len(results)} successful "
            f"in {processing_time:.2f}s"
        )
        
        return {
            "results": results,
            "total_queries": len(request.queries),
            "successful": successful,
            "failed": failed,
            "processing_time": processing_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch query error")
        raise HTTPException(status_code=500, detail=str(e))


# RENAME: /documents -> /kb/documents to avoid conflict with DB documents
@app.get("/kb/documents", response_model=List[LocalDocumentInfo])
async def get_kb_documents():
    """
    List documents from information on local filesystem (Knowledge Base).
    """
    doc_dir = r"D:\knowledge base\Document for test" # Keep specific path from main.py
    try:
        documents = []
        if os.path.exists(doc_dir):
            for filename in os.listdir(doc_dir):
                file_path = os.path.join(doc_dir, filename)
                if os.path.isfile(file_path):
                    last_modified = os.path.getmtime(file_path)
                    documents.append(
                        LocalDocumentInfo(
                            filename=filename,
                            path=file_path,
                            last_modified=str(last_modified)
                        )
                    )
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_kb_document(filename: str):
    """
    Download a document from the Knowledge Base (Local Filesystem).
    """
    file_path = os.path.join(STATIC_DIR, filename)
    doc_dir = r"D:\knowledge base\Document for test"
    try:
        if not os.path.exists(file_path):
            alt_path = os.path.join(doc_dir, filename)
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/view/{filename}")
# async def view_kb_document(filename: str):
#     """
#     View a document from the Knowledge Base.
#     """
#     file_path = os.path.join(STATIC_DIR, filename)
#     doc_dir = r"D:\knowledge base\Document for test"
#     try:
#         if not os.path.exists(file_path):
#             alt_path = os.path.join(doc_dir, filename)
#             if os.path.exists(alt_path):
#                 file_path = alt_path
#             else:
#                 raise HTTPException(status_code=404, detail="File not found")

#         return FileResponse(path=file_path, filename=filename)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingestDocument")
async def ingest_document(request: DocumentRequest):
    try:
        logger.info(f"Ingest request: {request.dict()}")

        # 1. Fetch meeting metadata
        query = """
            SELECT meeting_id, meeting_name, group_name, department, scheduled_at
            FROM meeting_dtls_surv
            WHERE meeting_id = :meeting_id
        """
        results = execute_query(query, {"meeting_id": request.meeting_id})

        if not results:
            raise HTTPException(status_code=404, detail="Meeting not found")

        meeting = results[0]

        additional_metadata = {
            "doc_type": request.doc_type.name,
            "group": meeting["group_name"],
            "department": meeting["department"],
            "meeting_date": meeting["scheduled_at"],
            "meeting_name": meeting["meeting_name"],
        }

        logger.info(f"Metadata: {additional_metadata}")

        # 2. Fetch all documents from service
        docs = fetch_documents_from_service(
            meeting_id=request.meeting_id,
            doc_type=str(request.doc_type.value),
        )

        if not docs:
            logger.warning("No documents returned from service.")
            return {
                "status": "warning",
                "message": "No documents found to ingest",
                "processed_count": 0
            }

        processed_ids = []
        
        # 3. Process each document
        for doc in docs:
            # Save to disk
            file_path, doc_id = save_single_document_to_disk(doc)
            
            # Process
            success = process_document(
                meeting_id=request.meeting_id,
                doc_type=request.doc_type,
                file_path=file_path,
                doc_id=doc_id,
                additional_metadata=additional_metadata,
            )
            
            if success:
                processed_ids.append(doc_id)
            else:
                logger.error(f"Failed to process document {doc_id}")

        bm25success = rebuild_bm25_index()
    
        if bm25success:
            print("\n✅ BM25 index rebuild completed successfully!")
            print("   You can now use the enhanced hybrid search.\n")
        else:
            print("\n❌ BM25 index rebuild failed. Check logs above.\n")


        if not processed_ids:
             # If we had docs but none processed successfully
            raise HTTPException(
                status_code=500,
                detail="Document processing failed for all documents"
            )

        return {
            "status": "success",
            "message": f"Ingested {len(processed_ids)} documents successfully",
            "document_ids": processed_ids,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingest document failed")
        raise HTTPException(status_code=500, detail=str(e))
 

@app.get("/all-external-documents/{meeting_id}")
def get_all_external_documents_combined(meeting_id: int):
    """
    Fetch ALL documents (AGENDA, SUPPORTING, MOM, ATR, SUBMISSION) from the external SURV portal
    for a given meeting, including base64 content.
    """
    try:
        combined_docs = []
        # Iterate over all defined DocTypes
        for dt in DocType:
            # fetch_documents_from_service returns list of dicts with 'bytes', 'filename', 'doc_id'
            docs = fetch_documents_from_service(meeting_id, str(dt.value))
            
            # Add doc_type info to each doc so frontend knows what it is
            for doc in docs:
                # logger.info(f"dd-- {doc["filename"]}")
                # Encode bytes to base64 string for JSON response
                if isinstance(doc["bytes"], bytes):
                    doc["base64"] = base64.b64encode(doc["bytes"]).decode('utf-8')
                    del doc["bytes"] # Remove raw bytes to avoid serialization error
                
                doc["doc_type"] = dt.name
                doc["meeting_id"] = meeting_id
                combined_docs.append(doc)

        return combined_docs
    except Exception as e:
        logger.error(f"Error fetching all external documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/external-documents/{doc_id}/download")
def download_external_document(doc_id: int):
    """
    Download a specific document by ID from the external SURV portal.
    1. Look up meeting_id and doc_type from DB using doc_id.
    2. Fetch content from external service.
    3. Return the specific document with base64 content.
    """
    try:
        # 1. Get metadata from DB
        query = """
            SELECT MEETING_ID, CATEGORY_NAME 
            FROM agenda_dtls_surv 
            WHERE ID = :doc_id
        """
        results = OracleClient().execute_query_dicts(query, {"doc_id": doc_id})
        
        if not results:
             raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in database")
        
        meeting_id = results[0]['meeting_id']
        doc_type_name = results[0]['category_name']
        
        # Map DB category name to DocType enum value if needed, 
        # or rely on the fact that fetch_documents_from_service takes a string value.
        # But wait, fetch_documents_from_service takes 'doc_type' which is usually the ID (e.g. "10", "13").
        # The DB 'CATEGORY_NAME' acts as the descriptive name (e.g. 'AGENDA').
        # We need to map CATEGORY_NAME back to the ID expected by the service.
        
        # Let's check DocType enum to map Name -> Value
        try:
            # Case-insensitive matching for robustness
            dt_enum = next(dt for dt in DocType if dt.name.upper() == doc_type_name.upper())
            doc_type_value = str(dt_enum.value)
        except StopIteration:
             # Fallback: maybe the DB column actually stores the ID? 
             # Previous conversation implied CATEGORY_NAME has 'MOM', 'ATR'.
             # If mapping fails, log warning.
             logger.warning(f"Could not map category '{doc_type_name}' to DocType enum. Trying to use it directly.")
             doc_type_value = str(doc_type_name)

        # 2. Fetch documents from service
        docs = fetch_documents_from_service(meeting_id, doc_type_value)
        
        # 3. Find the specific document
        target_doc = next((d for d in docs if str(d.get("doc_id")) == str(doc_id)), None)
        
        if not target_doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} found in DB but not returned by external service")

        # Encode bytes to base64 for JSON response
        if isinstance(target_doc.get("bytes"), bytes):
            target_doc["base64"] = base64.b64encode(target_doc["bytes"]).decode('utf-8')
            del target_doc["bytes"]
            
        target_doc["meeting_id"] = meeting_id
        target_doc["doc_type"] = doc_type_name
            
        return target_doc

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ─────────────────────────────────────────────
# ✅ Run server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8081))
    logger.info(f"port {port}" )
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
