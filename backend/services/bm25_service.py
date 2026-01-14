
from rank_bm25 import BM25Okapi
import pickle
import os
import re
import logging
import threading

logger = logging.getLogger(__name__)

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r'[\W_]+', ' ', text)
    return text.split()

class BM25Service:
    def __init__(self, docs=None, index_path=None):
        self.index_path = index_path
        self.docs = docs or []
        self._bm25 = None
        self._tokenized_texts = None
        self._lock = threading.RLock()  # Thread-safe access
        print(f"[BM25] Initialized with index path: {index_path}")
        if index_path and os.path.exists(index_path):
            self._load(index_path)
        elif self.docs:
            self.build(self.docs)

    def build(self, docs):
        self.docs = docs
        corpus = [d['text'] for d in docs]
        self._tokenized_texts = [simple_tokenize(t) for t in corpus]
        self._bm25 = BM25Okapi(self._tokenized_texts)
        if self.index_path:
            self._save(self.index_path)

    def _save(self, index_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "docs": self.docs
            }, f)
        print(f"[BM25] Saved index with {len(self.docs)} docs at {index_path}")

    def _load(self, index_path):
        if not os.path.exists(index_path):
            print(f"[BM25] No index found at {index_path}. Skipping load.")
            return

        with open(index_path, "rb") as f:
            payload = pickle.load(f)

        # Support multiple formats
        if isinstance(payload, dict):
            if "bm25" in payload:
                # New format: serialized BM25 object and docs
                self._bm25 = payload.get("bm25")
                self.docs = payload.get("docs", [])
                self._tokenized_texts = None
                print(f"[BM25] Loaded index with {len(self.docs)} docs.")
            else:
                # Older format in this codebase: tokenized + docs
                self.docs = payload.get('docs', [])
                self._tokenized_texts = payload.get('tokenized')
                if self._tokenized_texts:
                    self._bm25 = BM25Okapi(self._tokenized_texts)
                print(f"[BM25] Loaded legacy index with {len(self.docs)} docs (rebuilt BM25 from tokens).")
        else:
            # Legacy support (payload is a BM25Okapi object)
            self._bm25 = payload
            self.docs = []
            self._tokenized_texts = None
            print("[BM25] Loaded legacy BM25 model without docs.")

    def query(self, query_text, top_k=5, filters=None):
        """
        Query BM25 index with optional metadata filtering (thread-safe).
        
        Args:
            query_text: Search query
            top_k: Number of results to return
            filters: Optional dict with 'meeting_id' and/or 'doc_type'
        """
        with self._lock:  # Ensure thread-safe access
            if not self._bm25:
                return []
            
            # Apply metadata filters if provided
            if filters and self.docs:
                logger.info(f"?? Applying BM25 filters: {filters}")
                filtered_docs = []
                filtered_indices = []
                
                for idx, doc in enumerate(self.docs):
                    # Check meeting_id filter
                    if filters.get('meeting_id') is not None:
                        if doc.get('meeting_id') != filters['meeting_id']:
                            continue
                    
                    # Check doc_type filter
                    if filters.get('doc_type'):
                        if doc.get('doc_type') != filters['doc_type']:
                            continue
                    
                    filtered_docs.append(doc)
                    filtered_indices.append(idx)
                
                if not filtered_docs:
                    logger.info(f"  No documents matched filters: {filters}")
                    return []
                
                logger.info(f"? Filtered {len(self.docs)} docs down to {len(filtered_docs)} matching documents")
                
                # Get scores for all documents
                tokens = simple_tokenize(query_text)
                all_scores = self._bm25.get_scores(tokens)
                
                # Extract scores only for filtered documents
                filtered_scores = [all_scores[idx] for idx in filtered_indices]
                
                # Get top K from filtered results
                top_n_idx = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)[:top_k]
                
                results = []
                for i in top_n_idx:
                    doc = filtered_docs[i]
                    results.append({
                        "id": doc.get('id'),
                        "text": doc.get('text', '').strip(),
                        "bm25_score": float(filtered_scores[i]),
                        "document_name": doc.get('document_name', '')
                    })
                logger.info(f" Returning top {len(results)} BM25 results from filtered documents")
                return results
            
            # No filters - use original logic
            logger.info(f"?? BM25 search: No filters applied (searching {len(self.docs)} total documents)")
            tokens = simple_tokenize(query_text)
            scores = self._bm25.get_scores(tokens)
            if scores is None or len(scores) == 0:
                return []
            top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for idx in top_n_idx:
                document_name = ""
                if 0 <= idx < len(self.docs):
                    doc = self.docs[idx]
                    doc_id = doc.get('id')
                    doc_text = doc.get('text', '').strip()
                    document_name = doc.get('document_name', '')
                    if not doc_text:
                        continue  # Skip empty chunks
                else:
                    # Fallback when docs are unavailable (legacy BM25 without docs)
                    doc_id = str(idx)
                    doc_text = ""
                results.append({
                    "id": doc_id,
                    "text": doc_text,
                    "bm25_score": float(scores[idx]),
                    "document_name": document_name
                })
            return results

    def get_indexed_documents(self):
        """
        Get all unique document names in the BM25 index.
        
        Returns:
            set: Set of unique document names
        """
        if not self.docs:
            return set()
        
        document_names = set()
        for doc in self.docs:
            doc_name = doc.get('document_name', '')
            if doc_name:
                document_names.add(doc_name)
        
        return document_names

        
    _instance = None

    @staticmethod
    def get_client():
        if BM25Service._instance is None:
             # Try to find default path relative to this file
             base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
             
             env_path = os.getenv("BM25_INDEX_PATH")
             if env_path:
                 if os.path.isabs(env_path):
                     index_path = env_path
                 else:
                     index_path = os.path.join(base_dir, env_path)
                 print(f"[BM25] Loading index from env config: {index_path}")
             else:
                 index_path = os.path.join(base_dir, "data", "bm25_index", "bm25_index.pkl")
                 print(f"[BM25] Attempting to load default index from: {index_path}")

             if os.path.exists(index_path):
                 BM25Service._instance = BM25Service(index_path=index_path)
             else:
                 print("[BM25] Index not found.")
                 return None
        return BM25Service._instance