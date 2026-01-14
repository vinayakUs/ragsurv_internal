
# from typing import TypedDict, List, Dict, Any
# from langgraph.graph import StateGraph, END
# from tools.oracle_client import OracleClient
# from agenda_ingestion.store import fetch_agenda_items
# from meeting_summary.retrieval import RetrievalService
# from meeting_summary.generator import SummaryGenerator
# import logging
# import json

# logger = logging.getLogger("summary_graph")

# class SummaryState(TypedDict):
#     meeting_id: int
#     meeting_data: Dict[str, Any]
#     participants: List[Dict[str, Any]]
#     agenda_items: List[str]
    
#     # New Output Fields
#     general_section: str
#     deep_analysis_list: List[Dict[str, str]]
    
#     final_json_output: Dict[str, Any]

# def fetch_real_data_step(state: SummaryState) -> SummaryState:
#     meeting_id = state["meeting_id"]
#     logger.info(f"[{meeting_id}] Fetching REAL data from Oracle...")
#     client = OracleClient()
    
#     # 1. Fetch Meeting Details
#     logger.info("Fetching Meeting Details...")
#     q_meet = "SELECT * FROM MEETING_DTLS_SURV WHERE meeting_id = :1"
#     rows_meet = client.execute_query_dicts(q_meet, [meeting_id])
#     meeting_data = rows_meet[0] if rows_meet else {}
    
#     # 2. Fetch Participants
#     logger.info("Fetching Participants...")
#     q_part = """select u.first_name, u.last_name, u.email, u.phone from user_dtls_surv m 
#                 join tbl_users u on m.user_id = u.id  
#                 where m.meeting_id = :1"""
#     participants = client.execute_query_dicts(q_part, [meeting_id])
    
#     # 3. Fetch Agenda Items (Using existing store function or raw query)
#     # The existing function returns List[str] (text only). This is fine for now.
#     logger.info("Fetching Agenda Items...")
#     agenda_items = fetch_agenda_items(str(meeting_id))
#     data = {
#         "meeting_data": meeting_data,
#         "participants": participants,
#         "agenda_items": agenda_items
#     }

#     logger.info(f"Meeting Raw INFO from db : {data}")
    
#     return data

# def generate_general_section_step(state: SummaryState) -> SummaryState:
#     logger.info("Generating General Section Markdown...")

#     meeting_info = """
# # Meeting Title: test_di_02

# ## Meeting Information
# - Date: 2025-11-19 15:43:00
# - Group: Test_committee_01
# - Department: Investigation
# - Venue: Equinox

# ## Attendees
# | Name          | Role/Designation                   | Organization/Department        | Contact Information                |
# |---------------|------------------------------------|---------------------------------|------------------------------------|
# | Sachin Shigwan | N/A                                | National Stock Exchange (NSE)  | [sshigwan@nse.co.in](mailto:sshigwan@nse.co.in), +91 8655648872 |
# | Vinayak Paste | N/A                                | National Stock Exchange (NSE)  | [nseit_vinayakp@nse.co.in](mailto:nseit_vinayakp@nse.co.in), +91 8308008442 |
# | Lata Bajare   | N/A                                | National Stock Exchange (NSE)  | [nseit_lbajare@nse.co.in](mailto:nseit_lbajare@nse.co.in), +91 9673990211 |
# """
#     return  {"general_section": meeting_info}


#     generator = SummaryGenerator()
#     markdown = generator.generate_general_section(state["meeting_data"], state["participants"])
#     logger.info(f"Mardown from LLM : {markdown}")

#     return {"general_section": markdown}

# def generate_deep_analysis_step(state: SummaryState) -> SummaryState:
#     agenda_items = state["agenda_items"]
#     meeting_id = state["meeting_id"]
    
#     retriever = RetrievalService()
#     generator = SummaryGenerator()
    
#     analysis_list = []
    
#     for i, item in enumerate(agenda_items):
#         logger.info(f"Processing Deep Analysis for Item {i+1}/{len(agenda_items)}")
        
#         # 1. Retrieve Context (Grouped by Doc Type)
#         context_map = retriever.get_context_for_agenda(meeting_id, item)
        
#         # 2. Generate Analysis Markdown
#         markdown_analysis = generator.generate_deep_analysis(item, context_map)
        
#         analysis_list.append({
#             "agenda_title": item,
#             "analysis_content": markdown_analysis
#         })
        
#     return {"deep_analysis_list": analysis_list}

# def compile_final_json_step(state: SummaryState) -> SummaryState:
#     logger.info("Compiling Final JSON Structure...")
    
#     output = {
#         "meeting_id": state["meeting_id"],
#         "general_section": state["general_section"],
#         "agenda_analysis": state["deep_analysis_list"]
#     }
    
#     return {"final_json_output": output}



# def save_to_db_step(state: SummaryState) -> SummaryState:
#     logger.info("Saving Final Summary to DB...")
#     save_meeting_summary(state["meeting_id"], state["final_json_output"])
#     return state


# def create_summary_graph():
#     graph = StateGraph(SummaryState)
    
#     graph.add_node("fetch_real_data", fetch_real_data_step)
#     graph.add_node("generate_general", generate_general_section_step)
#     graph.add_node("generate_deep", generate_deep_analysis_step)
#     graph.add_node("save_to_db_step", save_to_db_step)
#     # graph.add_node("save_to_db_step", save_to_db_step)
    
#     graph.add_edge("fetch_real_data", "generate_general")
#     graph.add_edge("fetch_real_data", "generate_deep") # Data fetch feeds both parallel branches if we wanted, but let's keep it sequential for simplicity
    
#     # Sequential Flow: Data -> General -> Deep -> Compile
#     # Actually, General and Deep are independent. Let's make them parallel for speed? 
#     # LangGraph supports parallel execution if they share a common ancestor.
    
#     # Let's do strict sequential for now to avoid complexity in this edit
#     # fetch -> generate_general -> generate_deep -> compile
    
#     # Wait, I set edges above. Let me reset.
#     pass # Ignoring previous thought process, enforcing simple sequence.
    
#     graph = StateGraph(SummaryState)
#     graph.add_node("fetch_real_data", fetch_real_data_step)
#     graph.add_node("generate_general", generate_general_section_step)
#     graph.add_node("generate_deep", generate_deep_analysis_step)
#     graph.add_node("compile_json", compile_final_json_step)

#     graph.set_entry_point("fetch_real_data")
#     graph.add_edge("fetch_real_data", "generate_general")
#     graph.add_edge("generate_general", "generate_deep")
#     graph.add_edge("generate_deep", "compile_json")
#     graph.add_edge("compile_json", "save_to_db_step")
#     graph.add_edge("save_to_db_step", END)
    
#     return graph.compile()





from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from tools.oracle_client import OracleClient
from agenda_ingestion.store import fetch_agenda_items
from meeting_summary.retrieval import RetrievalService
from meeting_summary.generator import SummaryGenerator
import logging
import json

logger = logging.getLogger("summary_graph")


# =====================================================
# STATE DEFINITION
# =====================================================
class SummaryState(TypedDict, total=False):
    meeting_id: int

    meeting_data: Dict[str, Any]
    participants: List[Dict[str, Any]]
    agenda_items: List[str]

    general_section: str
    deep_analysis_list: List[Dict[str, str]]

    final_json_output: Dict[str, Any]


# =====================================================
# STEP 1: FETCH REAL DATA
# =====================================================
def fetch_real_data_step(state: SummaryState) -> dict:
    meeting_id = state["meeting_id"]
    logger.info(f"[{meeting_id}] Fetching REAL data from Oracle...")

    client = OracleClient()

    # Meeting details
    q_meet = "SELECT * FROM MEETING_DTLS_SURV WHERE meeting_id = :1"
    rows_meet = client.execute_query_dicts(q_meet, [meeting_id])
    meeting_data = rows_meet[0] if rows_meet else {}

    # Participants
    q_part = """
        SELECT u.first_name, u.last_name, u.email, u.phone
        FROM user_dtls_surv m
        JOIN tbl_users u ON m.user_id = u.id
        WHERE m.meeting_id = :1
    """
    participants = client.execute_query_dicts(q_part, [meeting_id])

    # Agenda items
    agenda_items = fetch_agenda_items(str(meeting_id)) or []

    logger.info(f"[{meeting_id}] Data fetched successfully")

    return {
        "meeting_data": meeting_data,
        "participants": participants,
        "agenda_items": agenda_items
    }


# =====================================================
# STEP 2: GENERATE GENERAL SECTION
# =====================================================
def generate_general_section_step(state: SummaryState) -> dict:
    logger.info("Generating General Section using LLM...")

    generator = SummaryGenerator()

    logger.info(f"raw data from db {state["meeting_data"]} , {state["participants"]}")
    markdown = generator.generate_general_section(
        state["meeting_data"],
        state["participants"]
    )
    logger.info(f"general summary {markdown}")

    return {
        "general_section": markdown
    }


# =====================================================
# STEP 3: GENERATE DEEP ANALYSIS
# =====================================================
def generate_deep_analysis_step(state: SummaryState) -> dict:
    logger.info("Generating Deep Agenda Analysis...")

    meeting_id = state["meeting_id"]
    agenda_items = state.get("agenda_items", [])

    retriever = RetrievalService()
    generator = SummaryGenerator()

    analysis_list: List[Dict[str, str]] = []

    for idx, agenda in enumerate(agenda_items, start=1):
        logger.info(f"[{meeting_id}] Processing agenda {idx}/{len(agenda_items)}")

        context_map = retriever.get_context_for_agenda(meeting_id, agenda)
        analysis_md = generator.generate_deep_analysis(agenda, context_map)

        analysis_list.append({
            "agenda_title": agenda,
            "analysis_content": analysis_md
        })

    return {
        "deep_analysis_list": analysis_list
    }


# =====================================================
# STEP 4: COMPILE FINAL JSON
# =====================================================
def compile_final_json_step(state: SummaryState) -> dict:
    logger.info("Compiling final JSON output...")

    return {
        "final_json_output": {
            "meeting_id": state["meeting_id"],
            "general_section": state["general_section"],
            "agenda_analysis": state["deep_analysis_list"]
        }
    }


# =====================================================
# DB SAVE HELPER
# =====================================================
def save_meeting_summary(meeting_id: int, summary_payload: dict) -> None:
    client = OracleClient()
    summary_json = json.dumps(summary_payload, ensure_ascii=False)

    merge_sql = """
        MERGE INTO MEETING_SUMMARY_SURV tgt
        USING (
            SELECT :meeting_id AS meeting_id,
                   :summary_json AS summary_json
            FROM dual
        ) src
        ON (tgt.meeting_id = src.meeting_id)
        WHEN MATCHED THEN
            UPDATE SET
                tgt.summary_json = src.summary_json,
                tgt.updated_at = SYSTIMESTAMP
        WHEN NOT MATCHED THEN
            INSERT (
                meeting_id,
                summary_json,
                created_at,
                updated_at
            )
            VALUES (
                src.meeting_id,
                src.summary_json,
                SYSTIMESTAMP,
                SYSTIMESTAMP
            )
    """

    logger.info(f"[{meeting_id}] Saving meeting summary to DB...")
    client.execute_write(
        merge_sql,
        {
            "meeting_id": meeting_id,
            "summary_json": summary_json
        }
    )
    logger.info(f"[{meeting_id}] Meeting summary saved successfully")


# =====================================================
# STEP 5: SAVE TO DB
# =====================================================
def save_to_db_step(state: SummaryState) -> SummaryState:
    final_json = state.get("final_json_output")
    logger.info(f"final output {final_json}")

    if not final_json:
        raise RuntimeError("final_json_output missing before DB save")

    save_meeting_summary(state["meeting_id"], final_json)
    return state


# =====================================================
# GRAPH BUILDER
# =====================================================
def create_summary_graph():
    graph = StateGraph(SummaryState)

    graph.add_node("fetch_real_data", fetch_real_data_step)
    graph.add_node("generate_general", generate_general_section_step)
    graph.add_node("generate_deep", generate_deep_analysis_step)
    graph.add_node("compile_json", compile_final_json_step)
    graph.add_node("save_to_db", save_to_db_step)

    graph.set_entry_point("fetch_real_data")

    graph.add_edge("fetch_real_data", "generate_general")
    graph.add_edge("generate_general", "generate_deep")
    graph.add_edge("generate_deep", "compile_json")
    graph.add_edge("compile_json", "save_to_db")
    graph.add_edge("save_to_db", END)

    return graph.compile()
