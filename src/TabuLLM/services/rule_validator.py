from typing import Any, List, Dict, Optional, Union
from pydantic import create_model, validator, BaseModel, ValidationError, Field
from utils.logging_config import logger

class RuleValidator:
    """Service for dynamically generating Pydantic models to enforce dataset rules."""

    @staticmethod
    def create_pydantic_model(rules: List[Dict[str, Any]], model_name: str = "GenerationRules"):
        """
        Dynamically creates a Pydantic model based on hierarchical rules.
        """
        fields = {}
        validators_dict = {}

        for rule in rules:
            col_name = rule.get('name')
            if not col_name:
                continue

            rule_type = rule.get('type')
            level = rule.get('level', 3) # Default to User Rule level

            # 1. Handle Field Typing (Level 1 - Integrity)
            # This ensures the LLM returns the correct primitive type
            if rule_type == 'number':
                fields[col_name] = (Union[float, int], Field(..., description=f"Numeric rule for {col_name}"))
            else:
                fields[col_name] = (Any, Field(..., description=f"Categorical rule for {col_name}"))

            # 2. Handle Constraint Validation (Level 2/3 - Boundaries & User Rules)
            # These are only added if the rule dict contains the specific constraints
            if rule_type == 'number':
                r_min = rule.get('min')
                r_max = rule.get('max')
                
                if r_min is not None or r_max is not None:
                    def make_range_validator(min_val, max_val):
                        def validate_range(cls, v):
                            if min_val is not None and v < min_val:
                                raise ValueError(f"Value {v} is less than minimum permitted {min_val}")
                            if max_val is not None and v > max_val:
                                raise ValueError(f"Value {v} is greater than maximum permitted {max_val}")
                            return v
                        return validate_range
                    validators_dict[f"validate_{col_name}_range_{level}"] = validator(col_name, allow_reuse=True)(make_range_validator(r_min, r_max))

            elif rule_type == 'category':
                constraints = rule.get('text', '')
                if constraints:
                    allowed_vals = [s.strip() for s in constraints.split(',')]
                    def make_enum_validator(allowed):
                        def validate_enum(cls, v):
                            if str(v) not in allowed:
                                raise ValueError(f"Value '{v}' not in permitted categories: {allowed}")
                            return v
                        return validate_enum
                    validators_dict[f"validate_{col_name}_enum_{level}"] = validator(col_name, allow_reuse=True)(make_enum_validator(allowed_vals))

        if not fields:
            return None

        # Build the dynamic model
        DynamicModel = create_model(model_name, __validators__=validators_dict, **fields)
        return DynamicModel

    @staticmethod
    def validate_batch(data: List[Dict[str, Any]], model: Optional[Any]) -> List[Dict[str, Any]]:
        """
        Validates a list of many rows against the generated Pydantic model.
        Returns a list of violation reports.
        """
        if model is None or not data:
            return []

        violations = []
        for i, row in enumerate(data):
            try:
                model(**row)
            except ValidationError as e:
                for error in e.errors():
                    violations.append({
                        "row_index": i,
                        "field": error['loc'][0],
                        "message": error['msg'],
                        "type": error['type'],
                        "input": row.get(error['loc'][0])
                    })
            except Exception as ex:
                violations.append({
                    "row_index": i,
                    "field": "global",
                    "message": str(ex),
                    "type": "runtime_error"
                })
        
        return violations

    @staticmethod
    def format_violations_for_llm(violations: List[Dict[str, Any]]) -> str:
        """Converts structured violations into a concise feedback string for the LLM."""
        if not violations:
            return ""
        
        lines = ["RULE VIOLATIONS DETECTED:"]
        # Group by row_index to keep feedback compact
        by_row = {}
        for v in violations:
            idx = v['row_index']
            if idx not in by_row: by_row[idx] = []
            by_row[idx].append(f"- Field '{v['field']}': {v['message']} (Received: {v['input']})")
        
        for idx, msgs in by_row.items():
            lines.append(f"Row {idx}:")
            lines.extend(msgs)
        
        return "\n".join(lines)
