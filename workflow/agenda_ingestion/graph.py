import logging
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

import os
from tools.doc_type import DocType
from tools.load_doc import load_doc_from_api
from agenda_ingestion.extract import extract_agenda_items
from agenda_ingestion.store import store_agenda_items

logger = logging.getLogger("agenda_graph") 

class AgendaState(TypedDict, total=False):
    meeting_id: str
    meeting_name: str
    agenda_doc_id: int
    agenda_text: str
    agenda_items: List[str]
    error: str

def load_agenda(state: AgendaState) -> AgendaState:
    if state.get("error"): return state
    
    try:
        # doc_id = state["agenda_doc_id"]
        logger.info(f"Loading agenda doc for meeting id {state['meeting_id']} and {DocType.AGENDA}...")
        res = load_doc_from_api(state['meeting_id'],str(DocType.AGENDA.value))
        state["agenda_text"] = res['text']
        logger.info(f'''text extracted : {state["agenda_text"][:200]}''')
        return state
    except Exception as e:
        logger.error(f"Error loading agenda: {e}")
        state["error"] = str(e)
        return state

def extract_step(state: AgendaState) -> AgendaState:
    if state.get("error"): return state
    
    try:
        items = extract_agenda_items(state["agenda_text"], state["meeting_id"])
        state["agenda_items"] = items
        return state
    except Exception as e:
        logger.error(f"Error extracting agenda: {e}")
        state["error"] = str(e)
        return state

def store_step(state: AgendaState) -> AgendaState:
    if state.get("error"): return state
    
    try:
        logger.info(f"[{state['meeting_id']}] Storing In DB...")
        store_agenda_items(state["meeting_id"], state["agenda_items"])
        logger.info(f"[{state['meeting_id']}] Storing In DB Completed...")
        return state
    except Exception as e:
        logger.error(f"Error storing agenda: {e}")
        state["error"] = str(e)
        return state

def create_agenda_graph():
    graph = StateGraph(AgendaState)

    graph.add_node("load_agenda", load_agenda)
    graph.add_node("extract_step", extract_step)
    graph.add_node("store_step", store_step)

    graph.add_edge("load_agenda", "extract_step")
    graph.add_edge("extract_step", "store_step")
    graph.add_edge("store_step", END)

    graph.set_entry_point("load_agenda")
    
    return graph.compile()
