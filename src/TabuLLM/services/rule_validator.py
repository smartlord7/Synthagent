from typing import Any, List, Dict, Optional, Union
from pydantic import create_model, validator, BaseModel, ValidationError, Field
from utils.logging_config import logger

class RuleValidator:
    """Service for dynamically generating Pydantic models to enforce dataset rules."""

    @staticmethod
    def create_pydantic_model(rules: List[Dict[str, Any]], model_name: str = "GenerationRules"):
        """
        Dynamically creates a Pydantic model based on hierarchical rules.

        All fields are Optional (default None) so that a missing or null value
        in a generated row does not trigger a spurious violation.  Constraint
        validators (range / enum) silently skip None values; they only fire
        when a concrete value is present but wrong.
        """
        fields = {}
        validators_dict = {}

        for rule in rules:
            col_name = rule.get('name')
            if not col_name:
                continue

            rule_type = rule.get('type')
            level = rule.get('level', 3)  # Default to User Rule level

            # 1. Field typing — always Optional so null/missing rows aren't flagged
            if rule_type == 'number':
                fields[col_name] = (
                    Optional[Union[float, int]],
                    Field(default=None, description=f"Numeric field: {col_name}")
                )
            else:
                fields[col_name] = (
                    Optional[Any],
                    Field(default=None, description=f"Categorical field: {col_name}")
                )

            # 2. Constraint validators — only when the rule carries concrete constraints
            if rule_type == 'number':
                r_min = rule.get('min')
                r_max = rule.get('max')

                if r_min is not None or r_max is not None:
                    def make_range_validator(min_val, max_val):
                        def validate_range(cls, v):
                            if v is None:
                                return v          # null is allowed; skip range check
                            try:
                                v = float(v)      # coerce string "23" → 23.0
                            except (TypeError, ValueError):
                                return v          # unparseable — not a range issue
                            if min_val is not None and v < min_val:
                                raise ValueError(
                                    f"Value {v} is below minimum {min_val}"
                                )
                            if max_val is not None and v > max_val:
                                raise ValueError(
                                    f"Value {v} is above maximum {max_val}"
                                )
                            return v
                        return validate_range

                    validators_dict[f"validate_{col_name}_range_{level}"] = validator(
                        col_name, allow_reuse=True
                    )(make_range_validator(r_min, r_max))

            elif rule_type == 'category':
                constraints = rule.get('text', '')
                if constraints:
                    allowed_vals = [s.strip() for s in constraints.split(',')]
                    # Normalise once — comparisons are case-insensitive & whitespace-stripped
                    allowed_norm = {a.lower() for a in allowed_vals}

                    def make_enum_validator(allowed, allowed_lower):
                        def validate_enum(cls, v):
                            if v is None:
                                return v          # null is allowed; skip enum check
                            if str(v).strip().lower() not in allowed_lower:
                                raise ValueError(
                                    f"Value '{v}' not in permitted categories: {allowed}"
                                )
                            return v
                        return validate_enum

                    validators_dict[f"validate_{col_name}_enum_{level}"] = validator(
                        col_name, allow_reuse=True
                    )(make_enum_validator(allowed_vals, allowed_norm))

        if not fields:
            return None

        DynamicModel = create_model(model_name, __validators__=validators_dict, **fields)
        return DynamicModel

    @staticmethod
    def validate_batch(data: List[Dict[str, Any]], model: Optional[Any]) -> List[Dict[str, Any]]:
        """
        Validates a list of rows against the generated Pydantic model.
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
        """
        Converts structured violations into a compact, field-centric feedback string
        for the LLM.

        Groups violations by field (not row) so the model sees "column X has 15
        bad values" rather than 15 near-identical row-level messages.  Only the
        first example value is shown per field to keep the prompt short.
        """
        if not violations:
            return ""

        # Aggregate by field
        by_field: Dict[str, Dict] = {}
        for v in violations:
            field = v['field']
            if field not in by_field:
                by_field[field] = {
                    "count": 0,
                    "message": v['message'],
                    "example": v.get('input'),
                }
            by_field[field]["count"] += 1

        lines = [
            f"RULE VIOLATIONS DETECTED — {len(violations)} violation(s) across "
            f"{len(by_field)} field(s).  Regenerate ALL rows fixing every instance:"
        ]
        for field, info in by_field.items():
            lines.append(
                f"  • '{field}' ({info['count']} row(s)): {info['message']}  "
                f"[example bad value: {info['example']!r}]"
            )

        return "\n".join(lines)
