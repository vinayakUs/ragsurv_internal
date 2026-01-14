import logging
import json
import re
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from llm_core.factory import LLMFactory

logger = logging.getLogger("agenda_extractor")

def extract_agenda_items(text: str, meeting_id: str) -> List[str]:
    """
    Extracts agenda items from the provided text using an LLM via the Abstraction Layer.
    Returns a list of strings, where each string is an agenda item.
    """
    if not text:
        logger.warning(f"[{meeting_id}] Empty document text provided for agenda extraction.")
        return []

    logger.info(f"[{meeting_id}] Extracting agenda items from text ({len(text)} chars)...")

    # Get the LLM provider from the factory
    # This decouples the extraction logic from the specific LLM implementation (Groq)
    llm_provider = LLMFactory.get_provider()
    llm = llm_provider.get_base_model() # Get base model for LangChain compatibility

    prompt = ChatPromptTemplate.from_template("""
    You are an expert secretary. Extract the agenda items from the following meeting document text.
    Return the output as a JSON object with a single key "agenda_items" which is a list of strings.
    Each string should be a distinct agenda item.
    Do not include sub-items as separate entries unless they are major points.
    
    Document Text:
    {text}
    
    Output Format:
    {{
        "agenda_items": [
            "1. First item",
            "2. Second item"
        ]
    }}

    ##Rules
    1. Do not start agenda with numbering like 1. 2. 
    """
    
    )

    chain = prompt | llm

    try:
        response = chain.invoke({"text": text})
        content = response.content.strip()
        
        # Clean up potential markdown code blocks
        content = re.sub(r"^```json", "", content).strip()
        content = re.sub(r"^```", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
        
        data = json.loads(content)
        items = data.get("agenda_items", [])
        
        logger.info(f"[{meeting_id}] Extracted {len(items)} agenda items.")
        return items

    except Exception as e:
        logger.error(f"[{meeting_id}] Failed to extract agenda items: {e}", exc_info=True)
        return []
