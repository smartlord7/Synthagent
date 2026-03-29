import json
import re
from utils.logging_config import logger

def robust_json_loads(text: str):
    """
    Attempts to parse JSON from text, handling conversational padding,
    markdown blocks, and common minor malformations.
    """
    if not text:
        return None

    # Helper to attempt parsing with various cleanups
    def try_parse(candidate):
        candidate = candidate.strip()
        if not candidate:
            return None
            
        # 1. Direct parse
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # 2. Manual "fixed manner" fixes
        # Replace NaN/inf/-inf with null
        fixed = re.sub(r'\b(NaN|inf|-inf)\b', 'null', candidate)
        # Remove trailing commas in objects: , } -> }
        fixed = re.sub(r',\s*\}', '}', fixed)
        # Remove trailing commas in arrays: , ] -> ]
        fixed = re.sub(r',\s*\]', ']', fixed)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
            
        return None

    # Step 1: Try raw text
    result = try_parse(text)
    if result is not None:
        return result

    # Step 2: Strip markdown blocks and try
    cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    result = try_parse(cleaned)
    if result is not None:
        return result

    # Step 3: Find the first occurrence of '{' or '[' and the last occurrence of '}' or ']'
    start_match = re.search(r'[\{\[]', text)
    if start_match:
        start_idx = start_match.start()
        
        # Find the last closing bracket/brace by searching backwards
        end_idx = -1
        for i in range(len(text) - 1, start_idx, -1):
            if text[i] in ['}', ']']:
                end_idx = i + 1
                break
        
        # If no closing bracket found, we take the rest of the string from start_idx
        if end_idx == -1:
            potential_json = text[start_idx:].strip()
        else:
            potential_json = text[start_idx:end_idx].strip()
            
        result = try_parse(potential_json)
        if result is not None:
            return result
            
        # If it still fails, raise RobustJSONError with context.
        try:
            json.loads(potential_json)
        except json.JSONDecodeError as e:
            lines = potential_json.splitlines()
            err_line = e.lineno
            # Handle potential negative indices or empty lines
            start = max(0, err_line - 3)
            end = min(len(lines), err_line + 2)
            context_snippet = "\n".join(lines[start:end]) if lines else "Empty or truncated JSON"
            
            error_data = {
                "error": str(e),
                "line": err_line,
                "column": e.colno,
                "context": context_snippet,
                "full_text": potential_json
            }
            logger.warning(f"JSON parsing failed after all attempts. Error: {e}")
            raise RobustJSONError(f"JSON parsing failed: {e}", error_data)

    logger.error("Failed to extract any JSON-like structure.")
    raise json.JSONDecodeError("No JSON-like structure found", text, 0)

class RobustJSONError(Exception):
    """Custom error carrying context for LLM repair."""
    def __init__(self, message, error_data):
        super().__init__(message)
        self.error_data = error_data

def to_compact_json(data: list, cols: list) -> dict:
    """Converts a list of dicts to a compact columnar format."""
    if not data:
        return {"cols": cols, "data": []}
    return {
        "cols": cols,
        "data": [[row.get(col) for col in cols] for row in data]
    }

def from_compact_json(compact_data: dict, fallback_cols: list = None) -> list:
    """Converts a compact columnar format back to a list of dicts."""
    if not isinstance(compact_data, dict):
        return []
    cols = compact_data.get("cols", fallback_cols)
    if not cols:
        return []
    rows = compact_data.get("data", [])
    if not isinstance(rows, list):
        # Fallback if the dict IS the list of rows (unlikely but safe)
        if isinstance(compact_data, list):
             rows = compact_data
        else:
             return []
    return [dict(zip(cols, row)) for row in rows]
