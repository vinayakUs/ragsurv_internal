"""
Gunicorn Configuration for NSE RAG Application
Production deployment with multiple workers for concurrent user support
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8081"
backlog = 4096  # Increased for 70 concurrent users

# Worker processes - optimized for 70 concurrent users
workers = int(os.getenv("GUNICORN_WORKERS", "12"))  # 12 workers for high concurrency
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 2000  # Increased for more concurrent connections
threads = 4  # Threads per worker for blocking operations
max_requests = 10000  # Restart workers after handling this many requests
max_requests_jitter = 1000  # Add randomness to avoid all workers restarting simultaneously

# Timeouts
timeout = 120  # Worker timeout in seconds
graceful_timeout = 30  # Time to wait for workers to finish during shutdown
keepalive = 5  # Keep-alive connections

# Logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "nse_rag_backend"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

def on_starting(server):
    """Called when the server is starting."""
    print("?? Starting NSE RAG Backend with Gunicorn")
    print(f"?? Workers: {workers}")
    print(f"?? Binding to: {bind}")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    print("??  Reloading workers...")

def when_ready(server):
    """Called just after the server is started."""
    print("? NSE RAG Backend is ready to handle requests")

def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal."""
    print(f"??  Worker {worker.pid} received interrupt signal")

def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    print(f"?? Worker {worker.pid} was aborted")