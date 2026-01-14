import sys
import os
import logging
from dotenv import load_dotenv

# Add project root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env vars
load_dotenv()

from agenda_analysis.graph import create_analysis_graph

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def main():
    meeting_id = "990"
    print(f"--- Triggering Agenda Analysis (Real Calls) for Meeting {meeting_id} ---")
    
    try:
        graph = create_analysis_graph()
        result = graph.invoke({"meeting_id": meeting_id})
        
        print("\n--- Execution Finished ---")
        print("Final State:")
        print(result)
        
        if result.get("error"):
            print(f"❌ Error encountered: {result['error']}")
        elif not result.get("agenda_items"):
            print("⚠️ No agenda items were found to analyze.")
        else:
            print(f"✅ Analyzed {len(result['agenda_items'])} items.")
            
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    main()
