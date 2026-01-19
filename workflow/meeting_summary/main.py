import sys
import os
import logging
import argparse
import json

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
workflow_dir = os.path.dirname(current_dir) # application/workflow
root_path = os.path.dirname(workflow_dir)   # application

sys.path.append(workflow_dir)
sys.path.append(root_path)

from dotenv import load_dotenv

# Load .env variables
load_dotenv()

from meeting_summary.graph import create_summary_graph

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file = os.getenv("LOG_FILE", os.path.join(root_path, "logs", "meeting_summary.log"))
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger("summary_runner")

def run_summary_workflow(meeting_id: int):
    logger.info(f"Starting DEEP summary generation for meeting {meeting_id}")
    
    try:
        graph = create_summary_graph()
        result = graph.invoke({"meeting_id": meeting_id})
        
        final_output = result.get("final_json_output", {})
        
        if not final_output:
            raise Exception("No output generated!")

        # Save report to JSON file - REMOVED as per user request
        # filename = f"meeting_{meeting_id}_deep_summary.json"
        
        # logger.info(f"Deep Summary saved to {filename}")
        return final_output
        
    except Exception as e:
        logger.error(f"Critical error in summary runner: {e}", exc_info=True)
        raise e

def main():
    parser = argparse.ArgumentParser(description="Generate Deep Meeting Summary")
    parser.add_argument("meeting_id", type=int, help="Meeting ID to summarize")
    args = parser.parse_args()
    
    meeting_id = args.meeting_id
    
    try:
        output = run_summary_workflow(meeting_id)
        
        if output:
            print("\n" + "="*50)
            print("GENERATION SUCCESSFUL")
            print("General Section Preview:")
            print(output.get("general_section", "")[:300] + "...")
            print("="*50 + "\n")
    except Exception as e:
        print(f"Summary generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()