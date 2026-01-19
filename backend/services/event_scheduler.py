
import threading
import time
import logging
import traceback
from datetime import datetime
import os # Added for path handling and env vars
from typing import Dict, Any

from tools.oracle_client import OracleClient
from tools.doc_type import DocType
from backend.services.document_processor import fetch_documents_from_service, save_single_document_to_disk, process_document
from backend.services.qdrant_service import delete_document, get_existing_doc_ids
from backend.rebuild_bm25_index import rebuild_bm25_index

# Imports moved to inside methods to prevent top-level crashes
# from workflow.agenda_ingestion.main import run_agenda_extraction
# from workflow.agenda_analysis.main import run_analysis_workflow 
# from workflow.meeting_summary.main import run_summary_workflow

# Configure logging for event scheduler
# This ensures logs appear even if workflow modules (which carry their own config) haven't been imported yet.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # os imported above
# Adjust path to find logs directory relative to this file
# This file is in backend/services/
# We want logs to be in <project_root>/logs/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_file = os.path.join(project_root, "logs", "event_scheduler.log")

os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

logger = logging.getLogger("event_scheduler")

class EventScheduler:
    def __init__(self, polling_interval: int = 5):
        self.polling_interval = polling_interval
        self._stop_event = threading.Event()
        self._thread = None
        self.client = OracleClient()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("EventScheduler started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            logger.info("EventScheduler stopped.")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._poll_and_process()
            except Exception as e:
                logger.error(f"Error in EventScheduler loop: {e}")
            
            # Wait for polling_interval *after* the previous processing completes.
            # This ensures that we don't overlap executions and there is always
            # a 5-second gap between the end of one process and the start of the next.
            time.sleep(self.polling_interval)

    def _poll_and_process(self):
        # Fetch pending events ordered by creation time and then by stage priority
        # Priority: SUBMISSION (1) > MOM (2) > ATR (3)
        # This global ordering guarantees that for any specific MEETING_ID, 
        # events are processed in creation order (and stage order as fallback).
        query = """
            SELECT MEETING_ID, STAGE, STATUS 
            FROM TABLE_AUDIT_SURV 
            WHERE STATUS = 'PENDING'
            ORDER BY 
                CREATED_DT ASC,
                CASE STAGE 
                    WHEN 'SUBMISSION' THEN 1 
                    WHEN 'MOM' THEN 2 
                    WHEN 'ATR' THEN 3 
                    ELSE 4 
                END ASC
            FETCH FIRST 1 ROWS ONLY
        """
        # Note: Oracle 12c+ syntax for FETCH FIRST
        
        events = self.client.execute_query_dicts(query)
        if not events:
            return

        for event in events:
            self._process_event(event)

    def _process_event(self, event: dict):
        meeting_id = event['meeting_id'] # execute_query_dicts returns lowercase keys
        stage = event['stage']
        logger.info(f"Processing event: Meeting {meeting_id}, Stage {stage}")

        # Update status to PROCESSING
        self._update_status(meeting_id, stage, 'PROCESSING')

        try:
            if stage == "SUBMISSION":
                self._handle_submission(meeting_id)
            elif stage == "MOM":
                self._handle_mom(meeting_id)
            elif stage == "ATR":
                self._handle_atr(meeting_id)
            else:
                raise ValueError(f"Unknown stage: {stage}")

            # Update status to COMPLETED
            self._update_status(meeting_id, stage, 'COMPLETED')
            logger.info(f"Successfully processed event: Meeting {meeting_id}, Stage {stage}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Failed to process event {meeting_id}: {error_msg}")
            self._update_status(meeting_id, stage, 'FAILED', error_msg)

    def _update_status(self, meeting_id, stage, status, error_msg=None):
        query = """
            UPDATE TABLE_AUDIT_SURV 
            SET STATUS = :status, 
                ERROR_MSG = :error_msg,
                UPDATED_DT = SYSTIMESTAMP 
            WHERE MEETING_ID = :meeting_id AND STAGE = :stage
        """
        self.client.execute_write(query, {
            "status": status, 
            "error_msg": error_msg,
            "meeting_id": meeting_id,
            "stage": stage
        })

    def _ingest_docs(self, meeting_id, doc_type_enum):
        logger.info(f"Ingesting {doc_type_enum.name} for meeting {meeting_id}")
        
        # 1. Fetch current documents from service
        docs = fetch_documents_from_service(meeting_id, str(doc_type_enum.value))
        current_doc_ids = {doc["doc_id"] for doc in docs}

        # 2. Fetch existing doc_ids from Qdrant
        existing_doc_ids = get_existing_doc_ids(meeting_id, doc_type_enum.name)
        
        # 3. Determine what to add and what to delete
        to_add_ids = current_doc_ids - existing_doc_ids
        to_delete_ids = existing_doc_ids - current_doc_ids
        
        if not to_add_ids and not to_delete_ids:
            logger.info(f"No changes for {doc_type_enum.name} (Meeting {meeting_id})")
            return

        # Delete removed documents
        for doc_id in to_delete_ids:
            logger.info(f"Deleting removed document {doc_id}")
            delete_document(str(doc_id), doc_id)
            
        if not docs:
            logger.warning(f"No documents found (or all deleted) for {doc_type_enum.name} (Meeting {meeting_id})")
            return
        if docs:
            for doc in docs:
                # Only process if it's in the add list
                if doc["doc_id"] in to_add_ids:
                    file_path, doc_id = save_single_document_to_disk(doc)
                    
                    # Fetch metadata
                    meta_query = "SELECT meeting_name, group_name, department, scheduled_at FROM meeting_dtls_surv WHERE meeting_id = :mid"
                    meta_res = self.client.execute_query_dicts(meta_query, {"mid": meeting_id})
                    additional_metadata = {}
                    if meta_res:
                        m = meta_res[0]
                        additional_metadata = {
                            "doc_type": doc_type_enum.name,
                            "group_name": m.get("group_name"),
                            "department": m.get("department"),
                            "meeting_date": m.get("scheduled_at"),
                            "meeting_name": m.get("meeting_name"),
                        }

                    process_document(
                        meeting_id=meeting_id,
                        doc_type=doc_type_enum,
                        file_path=file_path,
                        doc_id=doc_id, 
                        additional_metadata=additional_metadata,
                        update=False # We handled deletions manually above
                    )
        logger.info(f"Rebuilding BM25 index after changes in {doc_type_enum.name} ...")
        try:
            rebuild_bm25_index()
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")

    def _handle_submission(self, meeting_id):
        # Ingest AGENDA and SUPPORTING
        self._ingest_docs(meeting_id, DocType.AGENDA)
        self._ingest_docs(meeting_id, DocType.SUPPORTING)
        
        # Run Workflow
        logger.info(f"Running Agenda Ingestion Workflow for {meeting_id}")
        from workflow.agenda_ingestion.main import run_agenda_extraction
        run_agenda_extraction(meeting_id)

    def _handle_mom(self, meeting_id):
        # Ingest SUBMISSION
        self._ingest_docs(meeting_id, DocType.SUBMISSION)
        
        # No workflow for MOM stage as per requirements
        logger.info(f"MOM Stage: Ingested SUBMISSION for {meeting_id}. No workflow scheduled.")

    def _handle_atr(self, meeting_id):
        # Ingest MOM and ATR if existed
        self._ingest_docs(meeting_id, DocType.MOM)
        self._ingest_docs(meeting_id, DocType.ATR)
        
        # Run Workflows
        logger.info(f"Running Agenda Analysis and Meeting Summary Workflows for {meeting_id}")
        
        from workflow.agenda_analysis.main import run_analysis_workflow
        run_analysis_workflow(meeting_id)

        from workflow.meeting_summary.main import run_summary_workflow
        run_summary_workflow(meeting_id)

_scheduler = None

def start_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = EventScheduler()
        _scheduler.start()
    return _scheduler