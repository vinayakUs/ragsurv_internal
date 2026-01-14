
import logging
import json
import re
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from llm_core.factory import LLMFactory

logger = logging.getLogger("agenda_reasoning")

def analyze_agenda_status(agenda_item: str, chunks: List[str]) -> Dict:
    """
    Analyzes the status of an agenda item based on MoM chunks using an LLM via the Abstraction Layer.
    Returns a dictionary with 'status' and 'summary'.
    """
    logger.info(f"Analyzing status for item: '{agenda_item[:50]}...' with {len(chunks)} chunks.")

    # Get the LLM provider from the factory
    llm_provider = LLMFactory.get_provider()
    llm = llm_provider.get_base_model() # Get base model for LangChain compatibility
    
    # Combine chunks into a single context string
    context_text = "\n\n".join([f"- {chunk}" for chunk in chunks])

    prompt = ChatPromptTemplate.from_template("""
    You are a meeting analyst.
    
    AGENDA ITEM: "{agenda_item}"
    
    MEETING MINUTES CHUNKS:
    {context_text}
    
    Task:
    Determine the status of the agenda item based on the minutes.
    Status options: "Not Discussed", "Partial", "Completed".
    Provide a brief summary (1-2 sentences) explaining the status.
    
    Output JSON format:
    {{
        "status": "...",
        "summary": "..."
    }}
    """)

    chain = prompt | llm

    try:
        response = chain.invoke({
            "agenda_item": agenda_item,
            "context_text": context_text
        })
        
        content = response.content.strip()
        
        # Clean up potential markdown code blocks
        content = re.sub(r"^```json", "", content).strip()
        content = re.sub(r"^```", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
        
        result = json.loads(content)
        return result

    except Exception as e:
        logger.error(f"Failed to analyze status: {e}", exc_info=True)
        return {"status": "Error", "summary": f"Failed to analyze status due to an internal error: {e}"}