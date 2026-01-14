import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from backend.services.query_engine import query_knowledge_base_retrieval_only

from agenda_ingestion.store import fetch_agenda_items, update_agenda_status
from agenda_analysis.reasoning import analyze_agenda_status

logger = logging.getLogger("analysis_graph")

class AnalysisState(TypedDict, total=False):
    meeting_id: str
    agenda_items: List[str]
    error: str

def fetch_items(state: AnalysisState) -> AnalysisState:
    meeting_id = state.get("meeting_id")
    logger.info(f"[{meeting_id}] Fetching agenda items from DB...")
    
    try:
        items = fetch_agenda_items(meeting_id)
        logger.info(f"[{meeting_id}] Fetched {len(items)} agenda items from DB.")
        if not items:
            logger.warning(f"[{meeting_id}] No items found in DB to analyze.")
        
        state["agenda_items"] = items
        return state
    except Exception as e:
        logger.error(f"[{meeting_id}] Error fetching items: {e}")
        state["error"] = str(e)
        return state

def analyze_items(state: AnalysisState) -> AnalysisState:
    if state.get("error"): return state
    
    meeting_id = state["meeting_id"]
    items = state.get("agenda_items", [])
    
    if not items:
        return state
        
    logger.info(f"[{meeting_id}] Starting analysis for {len(items)} agenda items...")
    
    for item in items:
        try:
            # 1. Retrieve context (MOM chunks)
            # Using internal query engine directly
            data = query_knowledge_base_retrieval_only(
                query=item,
                group_by=['doc_type'],
                filters={'meeting_id': int(meeting_id), 'doc_type': 'MOM'}
            )
            
            chunks = []
            if data['status'] == 'success':
                # Navigate: results -> doc_type -> MOM -> items
                mom_group = data.get('results', {}).get('doc_type', {}).get('MOM', {})
                if 'items' in mom_group and mom_group['items']:
                    chunks = [res_item['text'] for res_item in mom_group['items'] if res_item.get('text')]
            
            if not chunks:
                logger.info(f"[{meeting_id}] No MOM chunks found for item: {item[:30]}...")
                chunks.append("No data found")

            # 2. Analyze status using LLM
            result = analyze_agenda_status(item, chunks)
            status = result.get("status", "Unknown")
            summary = result.get("summary", "")
            
            # 3. Update DB
            update_agenda_status(meeting_id, item, status, summary)
            logger.info(f"[{meeting_id}] Updated item '{item[:20]}...' -> {status}")
            
        except Exception as e:
            logger.error(f"[{meeting_id}] Error analyzing item '{item[:20]}...': {e}")
            # Continue to next item even if one fails
            
    return state

def create_analysis_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("fetch_items", fetch_items)
    graph.add_node("analyze_items", analyze_items)

    graph.add_edge("fetch_items", "analyze_items")
    graph.add_edge("analyze_items", END)

    graph.set_entry_point("fetch_items")
    
    return graph.compile()


