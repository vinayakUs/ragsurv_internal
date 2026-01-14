from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import gc
import torch
import numpy as np
import logging
import time
try:
    import psutil
except ImportError:
    psutil = None
from functools import wraps

# Import caching service
try:
    from backend.services.cache_service import cache_embedding, get_cached_embedding
    CACHING_ENABLED = True
except ImportError:
    CACHING_ENABLED = False
    logging.warning("Cache service not available, running without caching")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the BAAI Large embedding model
MODEL_NAME = os.getenv("EMBEDDING_MODEL_PATH", "/mnt/dev/backend/HF_bge_large")
model = None
embedding_dimension = 1024  # BGE-large-en-v1.5 has 1024 dimensions
BATCH_SIZE = 32  # Process texts in batches to reduce memory usage


def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_embedding_model():
    """
    Get or initialize the embedding model with memory optimization
    
    Returns:
        SentenceTransformer: The embedding model
    """
    global model
    if model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        try:
            # Force garbage collection before loading model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load model with optimized settings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(MODEL_NAME, device=device)
            
            # Log memory usage after loading
            if psutil:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)
                logger.info(f"Embedding model loaded. Memory usage: {memory_usage_mb:.2f} MB")
            else:
                logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    return model

@timer_decorator
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks with batching for memory efficiency
    
    Args:
        texts: List of text chunks
        
    Returns:
        List[List[float]]: List of embeddings (as lists of floats)
    """
    if not texts:
        logger.warning("Empty text list provided for embedding generation")
        return []
    
    try:
        model = get_embedding_model()
        all_embeddings = []
        
        # Process in batches to reduce memory usage
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE + 1} with {len(batch_texts)} texts")
            
            # Generate embeddings for the current batch
            batch_embeddings = model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=len(batch_texts) > 10,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            # Add batch embeddings to the result
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Force garbage collection between batches
            if i + BATCH_SIZE < len(texts):
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"Generated {len(all_embeddings)} embeddings successfully")
        return all_embeddings
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        # Return empty list or partial results if available
        return all_embeddings if 'all_embeddings' in locals() else []

@timer_decorator
def generate_query_embedding(query: str) -> Optional[List[float]]:
    """
    Generate embedding for a query with error handling and caching
    
    Args:
        query: The query text
        
    Returns:
        Optional[List[float]]: The query embedding or None if error
    """
    if not query or not query.strip():
        logger.warning("Empty query provided for embedding generation")
        return None
    
    # Check cache first
    if CACHING_ENABLED:
        cached = get_cached_embedding(query)
        if cached is not None:
            return cached
    
    try:
        model = get_embedding_model()
        
        # Generate embedding
        embedding = model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for better similarity search
        )
        
        # Convert numpy array to list for JSON serialization
        result = embedding.tolist()
        
        # Cache the result
        if CACHING_ENABLED:
            cache_embedding(query, result)
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        return None

def batch_process_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """
    Process a large number of texts in batches to avoid memory issues
    
    Args:
        texts: List of text chunks
        batch_size: Size of each batch
        
    Returns:
        List[List[float]]: List of embeddings
    """
    return generate_embeddings(texts)  # Uses batching internally now
