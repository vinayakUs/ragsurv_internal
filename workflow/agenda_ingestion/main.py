import logging
import sys
import os
from dotenv import load_dotenv
import sys
import os
import logging
import argparse
import json

load_dotenv()

# Ensure parent directory is in path to allow imports from tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenda_ingestion.graph import create_agenda_graph

# Setup logging
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_FILE = os.path.join(ROOT_DIR, "logs", "agenda_extraction.log")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agenda_runner")

def run_agenda_extraction(meeting_id: str):
    logger.info(f"🚀 Starting agenda extraction for meeting {meeting_id}")
    
    graph = create_agenda_graph()
    
    try:
        result = graph.invoke({"meeting_id": meeting_id})
        
        if result.get("error"):
            logger.error(f"❌ Workflow finished with error: {result['error']}")
        else:
            logger.info(f"✅ Agenda extraction completed successfully for {meeting_id}")
            logger.info(f"Extracted Items: {result.get('agenda_items')}")
            
        return result
        
    except Exception as e:
        logger.exception(f"❌ Critical failure for {meeting_id}: {e}")
        return None

if __name__ == "__main__":
    # Default to a sample ID if not provided
    mid = "990"
    parser = argparse.ArgumentParser(description="Generate Deep Meeting Summary")
    parser.add_argument("meeting_id", type=int, help="Meeting ID to summarize")
    args = parser.parse_args()
    
    meeting_id = args.meeting_id
    
        
    run_agenda_extraction(meeting_id)



