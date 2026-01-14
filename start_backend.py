import os
import sys
import uvicorn

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8081))

    print("Starting Knowledge Base RAG System Backend...")
    
    # Check if running in production mode
    production_mode = os.getenv("PRODUCTION", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    if production_mode and workers > 1:
        print(f"?? Starting in PRODUCTION mode with {workers} workers...")
        print("   Use: python -c \"import os; os.environ['PRODUCTION']='true'; os.environ['WORKERS']='10'; exec(open('start_backend.py').read())\"")
        print("   Or better: gunicorn -c gunicorn.conf.py backend.main:app")
        
        # For now, just use uvicorn with workers
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )
    else:
        print(f"?? Starting in DEVELOPMENT mode...")
        uvicorn.run(
            "backend.main:app",       # Path to your FastAPI instance
            host="0.0.0.0",
            port=port,
            reload=True,              # Auto reload on file changes
            reload_dirs=["backend"],  # (optional) Limit reload to your backend directory
            log_level="debug",        # Debug logs
            access_log=True
        )

