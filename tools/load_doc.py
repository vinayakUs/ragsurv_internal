# tools/load_doc.py
import os
import uuid
from pathlib import Path
import re
import logging

import requests
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import logging
logger = logging.getLogger("load_doc")

def load_doc(file_path: str) -> str:
    """
    Load a document (agenda or MoM) and return its text.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if p.suffix.lower() == ".pdf":
        docs = PyPDFLoader(file_path).load()
    elif p.suffix.lower() == ".docx":
        docs = Docx2txtLoader(file_path).load()
    else:
        docs = TextLoader(file_path, encoding="utf-8").load()

    text = "\n".join(d.page_content for d in docs)
    return text


def load_doc_from_api(meeting_id: int, doc_type:str) -> dict:
    """
    Fetches documents using shared logic, saves them, and helps extract text.
    Handles multiple documents by concatenating their text.
    """
    from backend.services.document_processor import fetch_documents_from_service, save_single_document_to_disk
    try:
        docs = fetch_documents_from_service(meeting_id, doc_type)
        if not docs:
            logger.warning(f"No documents found for Meeting {meeting_id} Type {doc_type}")
            return {"text": "", "file_name": ""}

        combined_text = []
        file_names = []
        
        for doc in docs:
            # Save locally using shared logic
            local_path, _ = save_single_document_to_disk(doc)
            
            file_names.append(doc['filename'])

            # Extract text
            # Note: doc['filename'] extension logic
            ext = os.path.splitext(doc['filename'])[1].lower()
            
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(local_path)
                    pages = loader.load()
                    text = "\n".join(page.page_content for page in pages)
                elif ext == ".docx":
                    loader = Docx2txtLoader(local_path)
                    data = loader.load()
                    text = "\n".join(d.page_content for d in data)
                else:
                    # Fallback for text
                    loader = TextLoader(local_path, encoding="utf-8")
                    data = loader.load()
                    text = "\n".join(d.page_content for d in data)
                
                combined_text.append(text)
            except Exception as extract_err:
                logger.error(f"Failed to extract text from {doc['filename']}: {extract_err}")

        return {
            "text": "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(combined_text), 
            "file_name": ", ".join(file_names)
        }

    except Exception as e:
        logger.error(f"Error in load_doc_from_api: {e}")
        return {"text": "", "file_name": ""}
