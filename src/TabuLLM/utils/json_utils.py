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
                # Try wrapping in brackets if it looks like a sequence of objects
                try:
                    val = json.loads("[" + fixed + "]")
                    logger.debug(f"JSON Parse Success: Stage='{stage_name}' (Bracket Wrap Fallback)")
                    return val
                except:
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
    error_data = {
        "error": "No JSON-like structure (brackets) found in text.", 
        "full_text": text,
        "context": "No JSON brackets detected",
        "line": 0,
        "column": 0
    }
    raise RobustJSONError("JSON extraction failed: No brackets found", error_data)

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

def from_compact_json(compact_data: dict, fallback_cols: list, h_to_f: dict = None) -> list:
    """
    Unpacks a compact JSON object into a list of dicts.
    fallback_cols: The list of Field IDs (primary keys) to ensure result dicts are keyed by these.
    h_to_f: (Optional) Map of Header Name -> Field ID to bridge AI names back to internal IDs.
    """
    if not isinstance(compact_data, dict):
        # Fallback if list returned directly
        if isinstance(compact_data, list):
             logger.debug(f"JSON Transform: Response is already a raw list of {len(compact_data)} rows.")
             return compact_data
        return []
        
    # If the LLM returned its own 'cols' list, we try to match them to the fallback_cols
    # safely (handling case-insensitivity or small typos).
    actual_cols = compact_data.get("cols")
    mapping_keys = []
    
    if actual_cols and fallback_cols:
        # 1. Try Strict Ordinal mapping (Gold Standard for robustness)
        if len(actual_cols) == len(fallback_cols):
            mapping_keys = fallback_cols
            logger.debug(f"JSON Transform: Using strict ordinal mapping into {len(mapping_keys)} unique fields.")
        else:
            # 2. Try Header-to-Field Bridge (Silver Standard for mismatched counts)
            logger.warning(f"JSON Transform: Length mismatch ({len(actual_cols)} vs {len(fallback_cols)}). Using Fuzzy Bridge.")
            # Build normalized maps
            norm_to_canonical = {str(c).strip().lower(): c for c in fallback_cols}
            bridge_norm = {str(h).strip().lower(): f for h, f in (h_to_f.items() if h_to_f else {})}
            
            for ac in actual_cols:
                ac_norm = str(ac).strip().lower()
                # Priority 1: Check if it's already a Field ID (fallback_cols)
                if ac_norm in norm_to_canonical:
                    mapping_keys.append(norm_to_canonical[ac_norm])
                # Priority 2: Check if it's a known Header that maps to a Field ID
                elif ac_norm in bridge_norm:
                    mapping_keys.append(bridge_norm[ac_norm])
                # Priority 3: Keep original name (last resort)
                else:
                    mapping_keys.append(ac)
    else:
        mapping_keys = fallback_cols or actual_cols

    if not mapping_keys:
        logger.warning("JSON Transform: No columns found for Unpacking.")
        return []
        
    rows = compact_data.get("data", [])
    if not isinstance(rows, list):
        return []
    
    logger.info(f"JSON Transform: Successfully unpacked {len(rows)} rows from Compact JSON using {len(mapping_keys)} identifying keys.")
    
    # Standardize result into a list of dicts using canonical mapping_keys
    final_output = []
    for row in rows:
        if isinstance(row, dict):
            norm_item = {str(k).strip().lower(): v for k, v in row.items()}
            new_item = {}
            # Use the derived mapping_keys to pull from the normalized AI dict
            # mapping_keys are Field IDs if the bridge succeeded
            for target_col in mapping_keys:
                new_item[target_col] = norm_item.get(str(target_col).lower(), row.get(target_col))
            final_output.append(new_item)
        elif isinstance(row, list):
            # Zip it with the derived mapping_keys (positional alignment)
            final_output.append(dict(zip(mapping_keys, row)))
        else:
            final_output.append(row)
            
    return final_output
