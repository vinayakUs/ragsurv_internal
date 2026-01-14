import os
import sys
import argparse
import logging

# Add project root to sys.path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
workflow_dir = os.path.dirname(os.path.dirname(current_dir)) # application/workflow - No, wait.
# main.py is in application/workflow/agenda_analysis
# dirname(main) = agenda_analysis
# dirname(agenda_analysis) = workflow
# dirname(workflow) = application

workflow_path = os.path.dirname(current_dir) # .../workflow
root_path = os.path.dirname(workflow_path)   # .../application

sys.path.append(workflow_path)
sys.path.append(root_path)


from agenda_analysis.graph import create_analysis_graph

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file = os.getenv("LOG_FILE", os.path.join(root_path, "logs", "agenda_analysis.log"))

os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

logger = logging.getLogger("analysis_runner")

from dotenv import load_dotenv

# Load .env variables
load_dotenv()

def run_analysis_workflow(meeting_id):
    logger.info(f"Starting agenda analysis for meeting {meeting_id}")

    try:
        graph = create_analysis_graph()
        
        initial_state = {"meeting_id": meeting_id}
        
        result = graph.invoke(initial_state)
        
        if result.get("error"):
            logger.error(f"Analysis failed: {result['error']}")
        else:
            logger.info(f"Analysis completed successfully for {meeting_id}")
            
        return result
            
    except Exception as e:
        logger.error(f"Critical error in runner: {e}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description="Run Agenda Analysis Workflow")
    parser.add_argument("meeting_id", help="The ID of the meeting to analyze")
    args = parser.parse_args()

    meeting_id = args.meeting_id
    run_analysis_workflow(meeting_id)

if __name__ == "__main__":
    main()
