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
    def try_parse(candidate, stage_name):
        candidate = candidate.strip()
        if not candidate:
            return None
            
        # 1. Direct parse
        try:
            val = json.loads(candidate)
            logger.debug(f"JSON Parse Success: Stage='{stage_name}' (Raw Match)")
            return val
        except json.JSONDecodeError:
            pass

        # 2. Manual "fixed manner" fixes
        fixed = re.sub(r'\b(NaN|inf|-inf)\b', 'null', candidate)
        
        # Remove trailing commas causing JSONDecodeError
        fixed = re.sub(r',\s*\}', '}', fixed)
        fixed = re.sub(r',\s*\]', ']', fixed)
        
        # Inject missing structural commas (e.g. LLM forgot comma between arrays or objects)
        fixed = re.sub(r'\}\s*\{', '}, {', fixed)
        fixed = re.sub(r'\]\s*\[', '], [', fixed)
        
        # Auto-close truncated JSON gracefully using a bracket stack
        stripped = fixed.rstrip()
        if stripped.endswith(','): 
            stripped = stripped[:-1]
            
        stack = []
        in_string = False
        escape = False
        for char in stripped:
            if escape:
                escape = False
                continue
            if char == '"':
                in_string = not in_string
            elif char == '\\':
                escape = True
            elif not in_string:
                if char in '[{':
                    stack.append(char)
                elif char in ']}':
                    if stack: stack.pop()
        
        closer = ''
        while stack:
            c = stack.pop()
            closer += ']' if c == '[' else '}'
            
        fixed = stripped + closer

        if fixed != candidate:
            try:
                val = json.loads(fixed)
                logger.debug(f"JSON Parse Success: Stage='{stage_name}' (After Aggressive Regex Fixes)")
                return val
            except json.JSONDecodeError:
                pass
                
        # 3. Ultimate Fallback: Python AST evaluation handles single quotes and trailing syntax natively
        try:
            import ast
            ast_str = fixed.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            val = ast.literal_eval(ast_str)
            if isinstance(val, (list, dict)):
                logger.debug(f"JSON Parse Success: Stage='{stage_name}' (AST Literal Eval Match)")
                return val
        except Exception:
            pass
            
        return None

    # Step 1: Try raw text
    result = try_parse(text, "Raw Text")
    if result is not None:
        return result

    # Step 2: Strip markdown blocks and try
    if "```" in text:
        cleaned = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
        result = try_parse(cleaned, "Markdown Strip")
        if result is not None:
            return result

    # Step 3: Find the first occurrence of '{' or '[' and the last occurrence of '}' or ']'
    start_match = re.search(r'[\{\[]', text)
    if start_match:
        start_idx = start_match.start()
        end_idx = -1
        for i in range(len(text) - 1, start_idx, -1):
            if text[i] in ['}', ']']:
                end_idx = i + 1
                break
        
        potential_json = text[start_idx:].strip() if end_idx == -1 else text[start_idx:end_idx].strip()
        result = try_parse(potential_json, "Bracket Extraction")
        if result is not None:
            return result
            
        # If it still fails, raise RobustJSONError with context.
        try:
            json.loads(potential_json)
        except json.JSONDecodeError as e:
            lines = potential_json.splitlines()
            err_line = e.lineno
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
            logger.warning(f"JSON parsing failed after all stages. Error: {e}")
            raise RobustJSONError(f"JSON parsing failed: {e}", error_data)

    logger.error("Failed to extract any JSON-like structure from response.")
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
    logger.debug(f"JSON Transform: Packing {len(data)} rows into Compact JSON.")
    return {
        "cols": cols,
        "data": [[row.get(col) for col in cols] for row in data]
    }

def from_compact_json(compact_data: dict, fallback_cols: list = None) -> list:
    """Converts a compact columnar format back to a list of dicts."""
    if not isinstance(compact_data, dict):
        # Fallback if list returned directly
        if isinstance(compact_data, list):
             logger.debug(f"JSON Transform: Response is already a raw list of {len(compact_data)} rows.")
             return compact_data
        return []
        
    cols = compact_data.get("cols", fallback_cols)
    if not cols:
        logger.warning("JSON Transform: No columns found for Unpacking.")
        return []
        
    rows = compact_data.get("data", [])
    if not isinstance(rows, list):
        return []
    
    logger.info(f"JSON Transform: Successfully unpacked {len(rows)} rows from Compact JSON.")
    return [dict(zip(cols, row)) for row in rows]
