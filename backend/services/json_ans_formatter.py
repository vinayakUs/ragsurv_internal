import json
import re

def extract_final_json_from_llm(response_text: str):
    """
    Extracts the first valid JSON object from a messy LLM response.
    """

    # 1. Find first {...} block (greedy).
    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        return None

    json_str = match.group(0).strip()

    # 2. Try parsing as JSON
    try:
        data = json.loads(json_str,strict=False)
        return {
            "title": data.get("title", ""),
            "answer": data.get("answer", "")
        }
    except json.JSONDecodeError:
        pass

    # 3. Auto-fix common issues (bad escape chars, trailing commas, unescaped newlines)
    try:
        fixed = re.sub(r",\s*}", "}", json_str)   # remove trailing commas
        # Attempt to fix unescaped newlines inside strings.
        # This is tricky without a full parser, but a simple \n -> \\n can help if the LLM output raw newlines
        # fixed = fixed.replace("\n", "\\n") 
        # data = json.loads(fixed)
        data = json.loads(fixed,strict=False)
        return {
            "title": data.get("title", ""),
            "answer": data.get("answer", "")
        }
    except Exception:
        pass

    return {
        "title": "Parsing Error",
        "answer": json_str
    }
