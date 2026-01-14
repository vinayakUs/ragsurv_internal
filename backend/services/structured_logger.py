"""
Structured Logger Service

Provides production-grade structured logging with:
- SERVER_ID for cluster identification
- request_id for end-to-end tracing
- Stage-based logging with duration tracking
- Consistent key=value format
"""

import os
import uuid
import time
import logging
from typing import Optional, Dict, Any
from contextvars import ContextVar
from functools import wraps

# Context variables for request tracking
_request_id: ContextVar[str] = ContextVar('request_id', default='')
_session_id: ContextVar[str] = ContextVar('session_id', default='')
_batch_id: ContextVar[str] = ContextVar('batch_id', default='')

# Server identity from environment
SERVER_ID = os.getenv("SERVER_ID", "llm-0")


class StructuredLogger:
    """Structured logger with context tracking."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.server_id = SERVER_ID
    
    def _format_message(self, stage: str, status: str, extra: Dict[str, Any], duration_ms: Optional[float] = None) -> str:
        """Format log message in structured key=value format."""
        parts = [
            f"server={self.server_id}",
            f"req={_request_id.get() or '-'}",
        ]
        
        session = _session_id.get()
        if session:
            parts.append(f"session={session[:8]}")
        
        batch = _batch_id.get()
        if batch:
            parts.append(f"batch={batch[:8]}")
        
        parts.append(f"stage={stage}")
        
        for key, value in extra.items():
            parts.append(f"{key}={value}")
        
        if duration_ms is not None:
            parts.append(f"ms={duration_ms:.0f}")
        
        parts.append(f"status={status}")
        
        return " | ".join(parts)
    
    def info(self, stage: str, status: str = "OK", duration_ms: Optional[float] = None, **extra):
        """Log info level with structured format."""
        msg = self._format_message(stage, status, extra, duration_ms)
        self.logger.info(msg)
    
    def warning(self, stage: str, status: str = "WARN", duration_ms: Optional[float] = None, **extra):
        """Log warning level with structured format."""
        msg = self._format_message(stage, status, extra, duration_ms)
        self.logger.warning(msg)
    
    def error(self, stage: str, status: str = "ERROR", duration_ms: Optional[float] = None, **extra):
        """Log error level with structured format."""
        msg = self._format_message(stage, status, extra, duration_ms)
        self.logger.error(msg)
    
    def exception(self, stage: str, status: str = "EXCEPTION", **extra):
        """Log exception with structured format and traceback."""
        msg = self._format_message(stage, status, extra)
        self.logger.exception(msg)


# Singleton loggers
_loggers: Dict[str, StructuredLogger] = {}


def get_structured_logger(name: str = "server") -> StructuredLogger:
    """Get or create a structured logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex[:12]


def set_request_context(request_id: str = None, session_id: str = None, batch_id: str = None):
    """Set request context for logging."""
    if request_id:
        _request_id.set(request_id)
    if session_id:
        _session_id.set(session_id)
    if batch_id:
        _batch_id.set(batch_id)


def get_request_id() -> str:
    """Get current request ID."""
    return _request_id.get()


def clear_request_context():
    """Clear request context after request completes."""
    _request_id.set('')
    _session_id.set('')
    _batch_id.set('')


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.duration_ms = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.duration_ms = (time.time() - self.start_time) * 1000


def timed_stage(stage: str, logger: StructuredLogger = None):
    """Decorator for timing function execution and logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_structured_logger()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                logger.info(stage, duration_ms=duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(stage, error=str(e)[:50])
                raise
        return wrapper
    return decorator
