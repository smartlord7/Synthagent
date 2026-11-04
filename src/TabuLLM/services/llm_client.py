import os
import time
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv
from services.logger import log_interaction
from utils.logging_config import logger
from utils.json_utils import robust_json_loads, RobustJSONError, to_compact_json, from_compact_json
from services.prompt_manager import prompt_manager
from services.rule_validator import RuleValidator

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def get_dataset_description(self, df_sample, provided_metadata=None, algorithm=None) -> str:
        pass

    @abstractmethod
    def generate_bulk_data(self, rows_payload, context, all_columns, all_fields, algorithm, instructions, temperature, num_empty=0, chunk_size=30, rules_metadata=None, enforce_rules=True) -> Optional[dict]:
        pass

    @abstractmethod
    def detect_target_columns(self, df_sample, single_target=False, algorithm=None) -> list:
        pass

    @abstractmethod
    def get_available_models(self) -> list:
        pass

    @abstractmethod
    def call_llm(self, messages: list, model_id: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
        """Standard interface for raw chat completions."""
        pass

    @abstractmethod
    def can_handle(self, model_id: Optional[str]) -> bool:
        """Check if this provider can handle the given model ID."""
        pass



class LocalOpenAIProvider(BaseLLMProvider):
    """Provider connecting to an OpenAI-compatible Local Server (e.g. LM Studio)."""
    
    def __init__(self, endpoint_url="http://127.0.0.1:1234/v1/chat/completions", model_name="google/gemma-3-4b"):
        self.endpoint_url = endpoint_url
        self.default_model = model_name
        # Set timeout to 600s (10 minutes) to prevent local model timeouts
        try:
            self.timeout = int(os.getenv("LOCAL_LLM_TIMEOUT", 600))
        except (ValueError, TypeError):
            self.timeout = 600
        # Cache the model context window after the first lookup — avoids a
        # redundant HTTP round-trip to LM Studio on every generate_bulk_data call.
        self._model_ctx_cache: dict[str, int] = {}

    def get_available_models(self):
        """Query local server for available models and context metadata."""
        # --- Stage 1: Try Native LM Studio API first (best metadata) ---
        try:
            native_url = self.endpoint_url.replace("/v1/chat/completions", "/api/v1/models")
            req = urllib.request.Request(native_url)
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = []
                for m in data.get("data", []):
                    ctx = None
                    for inst in m.get("loaded_instances", []):
                        cfg = inst.get("config", {})
                        ctx = cfg.get("context_length") or cfg.get("max_context_length")
                        if ctx: break
                    if not ctx:
                        name_lower = m["id"].lower()
                        if "gemma" in name_lower: ctx = 8192
                        elif "llama-3" in name_lower: ctx = 8192 if "-8b" in name_lower else 32768
                        else: ctx = 4096
                    models.append({"id": m["id"], "max_context": ctx})
                if models: return models
        except Exception: pass

        # --- Stage 2: Fallback to standard OpenAI /v1/models ---
        try:
            models_url = self.endpoint_url.replace("/chat/completions", "/models")
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = []
                for m in data.get("data", []):
                    ctx = m.get("context_window") or m.get("max_context") or m.get("context_length")
                    if not ctx:
                        name_lower = m["id"].lower()
                        if "gemma" in name_lower: ctx = 8192
                        elif "llama-3" in name_lower: ctx = 8192 if "-8b" in name_lower else 32768
                        else: ctx = 4096
                    models.append({"id": m["id"], "max_context": ctx})
                if models: return models
        except Exception as e:
            logger.warning(f"Both Native and OpenAI fallback failed to fetch models: {e}")

        # --- Stage 3: Final hardcoded safety default ---
        return [{"id": self.default_model, "max_context": 4096}, {"id": "meta/llama-3.1-8b", "max_context": 8192}]

    def _get_headers(self) -> dict:
        """Request headers for the OpenAI-compatible endpoint. Overridden by
        subclasses (e.g. OpenAIProvider) that need an Authorization header."""
        return {'Content-Type': 'application/json'}

    def _call_local_llm(self, messages, core_model=None, temperature=0.7, json_mode=False, response_format=None):
        active_model = core_model if core_model else self.default_model
        payload = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature
        }
        # Structured-output baseline: schema-constrained decoding on endpoints that
        # support OpenAI's response_format (json_schema, strict).
        if response_format is not None:
            payload["response_format"] = response_format

        logger.info(f"--- [SERVICE] CALLING LOCAL LLM: {active_model} (Timeout: {self.timeout}s) ---")
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.endpoint_url, data=data, headers=self._get_headers())

        max_retries = self._max_retries()
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    raw_response = response.read()
                    result = json.loads(raw_response.decode('utf-8'))
                    return result
            except (OSError, ConnectionResetError) as socket_err:
                delay = self._retry_delay(socket_err, attempt)
                if delay is not None and attempt < max_retries:
                    attempt += 1
                    logger.warning(f"Transient LLM error ({socket_err}); retry {attempt}/{max_retries} in {delay:.0f}s.")
                    time.sleep(delay)
                    continue
                if "[WinError 10038]" in str(socket_err):
                    logger.error(f"Critical Socket Failure (WinError 10038). The local server connection was dropped. Error: {socket_err}")
                else:
                    logger.error(f"Local LLM Socket Connection Reset: {socket_err}")
                raise socket_err
            except Exception as e:
                logger.error(f"Unexpected error during LLM request: {e}")
                raise e

    @staticmethod
    def _max_retries() -> int:
        try:
            return max(0, int(os.getenv("TABULLM_LLM_MAX_RETRIES", 8)))
        except (ValueError, TypeError):
            return 8

    @staticmethod
    def _retry_delay(err: Exception, attempt: int) -> Optional[float]:
        """Seconds to wait before retrying `err`, or None if it isn't transient.

        Retried: 429 and 5xx (rate limits / server hiccups, honouring Retry-After) and network-level
        failures (DNS lookup failed, connection reset). NOT retried: any other HTTP error -- notably
        400/422, which generate_bulk_data() handles itself by shrinking the context -- and timeouts,
        since a request that already hung for the full timeout shouldn't be repeated up to N times.
        Backoff is exponential (2s, 4s, ... capped at 120s), so the default 8 retries ride out
        roughly 6 minutes of rate limiting or a brief network drop.
        """
        backoff = min(2 ** (attempt + 1), 120)
        if isinstance(err, urllib.error.HTTPError):
            if err.code in (429, 500, 502, 503, 504):
                try:
                    return max(float(err.headers.get("Retry-After", 0)), backoff)
                except (TypeError, ValueError, AttributeError):
                    return backoff
            return None
        if isinstance(err, TimeoutError) or isinstance(getattr(err, "reason", None), TimeoutError):
            return None
        if isinstance(err, (urllib.error.URLError, ConnectionError)):
            return backoff
        return None

    def call_llm(self, messages: list, model_id: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
        """Public wrapper for OpenAI-style completions."""
        response = self._call_local_llm(messages, core_model=model_id, temperature=temperature)
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return "Error: No response from local LLM."

    def can_handle(self, model_id: Optional[str]) -> bool:
        """LocalOpenAI is the fallback or handles specific models."""
        return True

    def get_dataset_description(self, df_sample, provided_metadata=None, algorithm=None):
        df_sample_str = df_sample.to_string(index=False) if not df_sample.empty else "NO DATA ROWS"
        
        prompt = prompt_manager.get_prompt(
            "dataset_description",
            provided_metadata=provided_metadata,
            df_sample_str=df_sample_str,
            is_empty=df_sample.empty
        )
        
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._call_local_llm(messages, core_model=algorithm, temperature=0.3)
            content = response["choices"][0]["message"]["content"]
            log_interaction(f"Metadata Analysis [{algorithm or self.default_model}]", prompt, content)
            return content
        except Exception as e:
            logger.error(f"Error calling local LLM for description: {e}")
            return "Could not generate description from local model."

    def detect_target_columns(self, df_sample, single_target=False, algorithm=None):
        # 1. Generate Metadata-Only Schema String
        internal_cols = ['_row_id', 'is_context', 'is_target', 'ai_audit']
        col_meta = []
        for col in df_sample.columns:
            if col in internal_cols or col.startswith('is_ai_'):
                continue
            dtype = str(df_sample[col].dtype)
            col_meta.append(f"- {col} ({dtype})")
        
        column_metadata = "\n".join(col_meta)
        
        prompt = prompt_manager.get_prompt(
            "target_detection",
            column_metadata=column_metadata,
            single_target=single_target
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._call_local_llm(messages, core_model=algorithm, temperature=0.1, json_mode=True)
            content = response["choices"][0]["message"]["content"]
            logger.info(f"--- [SERVICE] TARGET DETECTION RAW RESPONSE: {content} ---")
            log_interaction(f"Target Detection [{algorithm or self.default_model}]", prompt, content)
            
            # Robust JSON parsing
            targets = robust_json_loads(content)
            if isinstance(targets, list):
                return targets
            if isinstance(targets, dict):
                # If LLM returns {"target": "name"} or {"targets": ["a", "b"]}
                if "targets" in targets and isinstance(targets["targets"], list):
                    return targets["targets"]
                if "target" in targets:
                    return [targets["target"]] if isinstance(targets["target"], str) else targets["target"]
                # Fallback: return all values if it's a simple dict
                return [str(v) for v in targets.values() if isinstance(v, (str, int))]
            return []
        except Exception as e:
            logger.error(f"Error calling local LLM for target detection: {e}")
            return []

    @staticmethod
    def _raw_rows(content, all_columns, stats):
        """No-controller parse: strict json.loads, rows zipped with whatever column names
        the model returned (no bridge to the declared schema). Unparseable output yields
        no rows."""
        try:
            parsed = json.loads(content)
        except Exception:
            stats["parse_failures"] += 1
            return []
        if isinstance(parsed, dict) and "records" in parsed:
            cols, data = None, parsed.get("records", [])
        elif isinstance(parsed, dict):
            cols, data = parsed.get("cols"), parsed.get("data", [])
        else:
            cols, data = None, parsed
        if not isinstance(data, list):
            stats["parse_failures"] += 1
            return []
        if cols is not None and list(cols) != list(all_columns):
            stats["schema_mismatches"] += 1
        rows = []
        for row in data:
            if isinstance(row, dict):
                rows.append(row)
            elif isinstance(row, list) and cols is not None:
                rows.append(dict(zip(cols, row)))
        return rows

    @staticmethod
    def _count_invalid_rows(rows, rules_metadata, stats):
        """Counts (but never corrects or drops) rows that violate the active rules."""
        if not rules_metadata or not rows:
            return
        if isinstance(rules_metadata, dict) and "integrity" in rules_metadata:
            active_rules = list(rules_metadata.get("integrity", []))
            if rules_metadata.get("enforce_boundaries"):
                active_rules.extend(rules_metadata.get("boundaries", []))
            if rules_metadata.get("enforce_user"):
                active_rules.extend(rules_metadata.get("user", []))
        else:
            active_rules = rules_metadata
        model = RuleValidator.create_pydantic_model(active_rules)
        violations = RuleValidator.validate_batch(rows, model)
        stats["rule_violations"] += len(violations)
        stats["invalid_rows"] += len({v["row_index"] for v in violations})

    def generate_bulk_data(self, rows_payload, context, all_columns, all_fields, algorithm, instructions, temperature, num_empty=0, chunk_size=30, rules_metadata=None, enforce_rules=True, raw_mode=False, response_format=None):
        # response_format (with raw_mode) is the structured-output baseline: the same
        # no-controller path, but the endpoint constrains decoding to a JSON schema.
        # raw_mode is the no-controller ablation: the same prompts and context, but strict
        # json.loads with no repair, no schema bridge, rule violations counted but never
        # corrected, no context-overflow retry, no top-up and no overshoot truncation.
        # 1. Parse dictionary context securely
        meta = context.get("metadata", "Not provided") if isinstance(context, dict) else str(context)
        manual_ctx = context.get("manual_ctx", "[]") if isinstance(context, dict) else "[]"
        global_ctx = context.get("global_ctx", []) if isinstance(context, dict) else []

        # 2. Add Background Metadata & Reference Context
        base_messages = []
        
        # Compact Manual Context if available
        manual_ctx_data = []
        if manual_ctx and manual_ctx != "[]":
            try:
                manual_ctx_data = json.loads(manual_ctx)
            except: pass
        
        # We use HEADERS for the prompt (so LLM understands) but we will use FIELDS for the response mapping
        compact_manual = to_compact_json(manual_ctx_data, all_columns)
        
        init_content = prompt_manager.get_prompt(
            "generation_init",
            meta=meta,
            compact_manual_json=json.dumps(compact_manual)
        )
        base_messages.append({"role": "user", "content": init_content})
        base_messages.append({"role": "assistant", "content": "I understand the dataset background context. Please feed global pattern subsets next."})

        # 3. Global Context is now streamed turn-by-turn in the chunk loop below
        # to ensure it stays within the model's token budget.

        # --- CHUNKED TARGET PROCESSING ---
        chunk_sz = chunk_size if chunk_size else 30
        subsets = [rows_payload[i:i+chunk_sz] for i in range(0, len(rows_payload), chunk_sz)] if rows_payload else []
        logger.info(f"Starting LocalOpenAI generation... Total payload: {len(rows_payload)}. Num empty requested: {num_empty}. Chunk limit: {chunk_sz}. Derived subsets: {max(1, len(subsets))}")
        
        if not subsets and num_empty > 0:
            # 3b. Force Zero-Shot Mode for empty payload
            n_gen_chunks = max(1, (num_empty + chunk_sz - 1) // chunk_sz)
            subsets = [[] for _ in range(n_gen_chunks)]
            logger.info(f"Zero-Shot Mode Activated: Generating {num_empty} rows across {n_gen_chunks} chunks.")

        # 4. Token Budgeting Setup  (context window cached after first lookup)
        if algorithm not in self._model_ctx_cache:
            models_info = self.get_available_models()
            for m in models_info:
                self._model_ctx_cache[str(m["id"])] = int(m["max_context"])
        model_ctx_limit = self._model_ctx_cache.get(algorithm, 4096)
        
        # Heuristic: 2.8 chars per token for dense data JSON
        TOKEN_RATIO = 2.8 
        SAFETY_MARGIN = 0.85 # Use 85% of window
        MAX_SAFE_TOKENS = int(model_ctx_limit * SAFETY_MARGIN)
        
        # Estimate fixed overhead (Prompt init + metadata + instructions)
        fixed_overhead_chars = len(init_content) + len(instructions) + len(str(all_columns)) * 2
        fixed_overhead_tokens = int(fixed_overhead_chars / TOKEN_RATIO) + 300 # Buffer for framing
        
        # Estimate tokens per row for context
        sample_row_tokens = 50 # Default fallback
        if global_ctx:
            sample_size = min(5, len(global_ctx))
            sample_json = json.dumps(to_compact_json(global_ctx[:sample_size], all_columns))
            sample_row_tokens = int((len(sample_json) / sample_size) / TOKEN_RATIO)

        final_data = []
        schema_updates = {}
        accumulated_stats = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0, "rule_violations": 0, "repair_attempts": 0,
                             "requested_rows": int(num_empty or len(rows_payload or [])), "delivered_rows": 0}
        if raw_mode:
            accumulated_stats.update({"parse_failures": 0, "schema_mismatches": 0, "invalid_rows": 0, "request_errors": 0})
        logger.info(f"Budgeting generation context for model {algorithm} (Max: {model_ctx_limit} tok).")

        for subset_idx, subset in enumerate(subsets):
            # A. Calculate Budget for this turn
            target_json = json.dumps(to_compact_json(subset, all_columns)) if subset else "{}"
            target_tokens = int(len(target_json) / TOKEN_RATIO)
            
            # If generating empty rows, reserve space for them in output
            output_reserve = 100 * (chunk_sz if not subset else 1) 
            
            used_tokens = fixed_overhead_tokens + target_tokens + output_reserve
            remaining_budget = MAX_SAFE_TOKENS - used_tokens
            
            # B. Smart Context Slicing & Rotation
            rows_allowed = max(1, int(remaining_budget / sample_row_tokens)) if sample_row_tokens > 0 else 20
            
            # Rotate context so different chunks see different rows if we have plenty
            start_row = (subset_idx * rows_allowed) % len(global_ctx) if global_ctx else 0
            current_global_ctx = global_ctx[start_row : start_row + rows_allowed]
            # Wrap around if slicing reached the end but didn't fill allowed count
            if len(current_global_ctx) < rows_allowed and len(global_ctx) > rows_allowed:
                current_global_ctx += global_ctx[0 : rows_allowed - len(current_global_ctx)]

            logger.info(f"Turn {subset_idx+1}: Budgeting {len(current_global_ctx)} context rows out of {len(global_ctx)} requested.")

            attempt_reduction = 0
            max_reductions = 3

            while attempt_reduction <= max_reductions:
                messages = list(base_messages)
                
                # C. Feed Dynamic Global Context
                if current_global_ctx:
                    compact_global = to_compact_json(current_global_ctx, all_columns)
                    global_content = prompt_manager.get_prompt(
                        "generation_global_ctx",
                        compact_global_json=json.dumps(compact_global)
                    )
                    messages.append({"role": "user", "content": global_content})
                    messages.append({"role": "assistant", "content": "Received global context references."})

                # C. Feed Tasks
                rows_to_gen = 0
                if num_empty > 0:
                    if not subset:
                        rows_to_gen = min(chunk_sz, num_empty - subset_idx * chunk_sz)
                tasks = []
                if subset:
                    compact_subset = to_compact_json(subset, all_columns)
                    tasks.append(f"- Complete these specific PARTIAL TARGET ROWS. Fill ALL null/empty fields with statistically plausible values based on context (Chunk {subset_idx+1}):\n{json.dumps(compact_subset)}")
                if num_empty > 0:
                    if not subset:
                        # Pure generation chunk — ask for this chunk's share only
                        rows_this = min(chunk_sz, num_empty - subset_idx * chunk_sz)
                        if rows_this > 0:
                            tasks.append(f"- Generate {rows_this} entirely NEW rows. Ensure every single column is populated and consistent with the reference data.")
                    elif subset_idx == len(subsets) - 1:
                        rows_to_gen = num_empty

                compact_payload_json = json.dumps(to_compact_json(subset, all_columns)) if subset else "[]"
                
                logger.info(f"Processing chunk {subset_idx + 1}/{len(subsets)}. Prompt generation...")
                final_prompt = prompt_manager.get_prompt(
                    "generation_final",
                    instructions=instructions,
                    all_columns=all_columns,
                    is_empty=(not subset and not global_ctx and not manual_ctx_data),
                    num_rows_to_generate=max(0, rows_to_gen),
                    compact_payload_json=compact_payload_json
                )
                messages.append({"role": "user", "content": final_prompt})

                try:
                    response = self._call_local_llm(messages, core_model=algorithm, temperature=temperature, json_mode=True,
                                                    response_format=response_format if raw_mode else None)
                    content = response["choices"][0]["message"]["content"]
                    usage = response.get("usage", {})

                    # Accumulate stats
                    accumulated_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    accumulated_stats["candidates_tokens"] += usage.get("completion_tokens", 0)
                    accumulated_stats["total_tokens"] += usage.get("total_tokens", 0)

                    if raw_mode:
                        raw_rows = self._raw_rows(content, all_columns, accumulated_stats)
                        self._count_invalid_rows(raw_rows, rules_metadata if enforce_rules else None, accumulated_stats)
                        final_data.extend(raw_rows)
                        break

                    # Use unified robust_json_loads with repair step
                    try:
                        parsed_content = robust_json_loads(content)
                    except RobustJSONError as re:
                        logger.info(f"Initial JSON parsing failed. Attempting focused repair... Error: {re.error_data['error']}")
                        accumulated_stats["repair_attempts"] += 1
                        repair_prompt = prompt_manager.get_prompt(
                            "json_repair",
                            line=re.error_data['line'],
                            column=re.error_data['column'],
                            error=re.error_data['error'],
                            context=re.error_data['context']
                        )
                        messages_repair = list(messages)
                        messages_repair.append({"role": "assistant", "content": content})
                        messages_repair.append({"role": "user", "content": repair_prompt})
                        
                        response_repair = self._call_local_llm(messages_repair, core_model=algorithm, temperature=0.1, json_mode=True)
                        content_repair = response_repair["choices"][0]["message"]["content"]
                        parsed_content = robust_json_loads(content_repair)

                    if isinstance(parsed_content, dict) and "schema_updates" in parsed_content and not schema_updates:
                        schema_updates = parsed_content["schema_updates"]

                    # Implicit Schema Discovery: Always scan for changes in the 'cols' list
                    returned_cols = parsed_content.get("cols", []) if isinstance(parsed_content, dict) else []
                    if returned_cols and len(returned_cols) == len(all_columns):
                        for i, rc in enumerate(returned_cols):
                            if str(rc).strip().lower() != str(all_columns[i]).strip().lower():
                                schema_updates[all_fields[i]] = rc

                    # Create Fuzzy Bridge Map for robustness against count mismatch
                    h_to_f_bridge = dict(zip(all_columns, all_fields))

                    if isinstance(parsed_content, dict) and "cols" in parsed_content and "data" in parsed_content:
                        subset_data = from_compact_json(parsed_content, all_fields, h_to_f=h_to_f_bridge)
                    elif isinstance(parsed_content, list):
                        subset_data = from_compact_json({"data": parsed_content}, all_fields, h_to_f=h_to_f_bridge) if len(parsed_content)>0 and isinstance(parsed_content[0], list) else parsed_content
                    else:
                        subset_data = parsed_content.get("data", [])

                    # --- RULE ENFORCEMENT VALIDATION LOOP ---
                    if enforce_rules and rules_metadata:
                        # Hierarchical filtering
                        if isinstance(rules_metadata, dict) and "integrity" in rules_metadata:
                            active_rules = list(rules_metadata.get("integrity", []))
                            if rules_metadata.get("enforce_boundaries"):
                                active_rules.extend(rules_metadata.get("boundaries", []))
                            if rules_metadata.get("enforce_user"):
                                active_rules.extend(rules_metadata.get("user", []))
                        else:
                            active_rules = rules_metadata

                        pydantic_model = RuleValidator.create_pydantic_model(active_rules)
                        if pydantic_model:
                            max_retries = 2
                            for retry in range(max_retries):
                                violations = RuleValidator.validate_batch(subset_data, pydantic_model)
                                if not violations:
                                    break

                                accumulated_stats["rule_violations"] += len(violations)
                                logger.warning(f"Chunk {subset_idx+1}: {len(violations)} rule violations found. Triggering self-correction (Retry {retry+1}/{max_retries}).")
                                feedback = RuleValidator.format_violations_for_llm(violations)
                                
                                # Prepare correction prompt
                                correction_messages = list(messages)
                                correction_messages.append({"role": "assistant", "content": content})
                                correction_messages.append({
                                    "role": "user", 
                                    "content": f"{feedback}\n\n" + prompt_manager.get_prompt("generation_fix")
                                })
                                
                                try:
                                    response_corr = self._call_local_llm(correction_messages, core_model=algorithm, temperature=0.1, json_mode=True)
                                    content = response_corr["choices"][0]["message"]["content"]
                                    # Use robust_json_loads directly here
                                    parsed_corr = robust_json_loads(content)
                                    h_to_f_bridge = dict(zip(all_columns, all_fields))
                                    if isinstance(parsed_corr, dict) and "data" in parsed_corr:
                                        subset_data = from_compact_json(parsed_corr, all_fields, h_to_f=h_to_f_bridge)
                                    elif isinstance(parsed_corr, list):
                                        subset_data = from_compact_json({"data": parsed_corr}, all_fields, h_to_f=h_to_f_bridge) if len(parsed_corr)>0 and isinstance(parsed_corr[0], list) else parsed_corr
                                    else:
                                        subset_data = parsed_corr.get("data", [])
                                except Exception as e:
                                    logger.error(f"Self-correction failed: {e}")
                                    break # Give up on correction for this chunk if it crashes

                    final_data.extend(subset_data)
                    break # Success! Continue onto next chunk subset

                except urllib.error.HTTPError as e:
                    if raw_mode:
                        accumulated_stats["request_errors"] += 1
                        logger.warning(f"[raw] Chunk {subset_idx+1} request failed ({e}); no retry.")
                        break
                    if e.code in [400, 422] and attempt_reduction < max_reductions:
                        attempt_reduction += 1
                        reduction_pct = 0.4 # Reduce context by 40%
                        new_len = int(len(current_global_ctx) * (1 - reduction_pct))
                        logger.warning(f"400 Context Overflow on chunk {subset_idx+1}. Reducing Global Context from {len(current_global_ctx)} to {new_len}. (Attempt {attempt_reduction})")
                        current_global_ctx = current_global_ctx[:new_len]
                        if not current_global_ctx and attempt_reduction == max_reductions:
                            raise e
                        continue
                    else:
                        raise e
                except Exception as e:
                    if raw_mode:
                        accumulated_stats["request_errors"] += 1
                        logger.warning(f"[raw] Chunk {subset_idx+1} failed ({e}); no retry.")
                        break
                    logger.error(f"Fatally failed chunk index {subset_idx+1}: {e}")
                    raise e

        # --- TOP-UP GUARANTEE ---
        # The LLM may return fewer rows than requested per chunk.
        # Keep requesting the shortfall until we reach num_empty or exhaust retries.
        if not raw_mode and num_empty > 0 and len(final_data) < num_empty:
            logger.info(f"Initial bulk generation fell short. Have {len(final_data)}/{num_empty} rows. Transitioning to Top-Up Guarantee loop...")
            max_topups = 50 # High limit to run "until dataset is completed" securely
            topup_round = 0
            consecutive_zeros = 0
            
            while len(final_data) < num_empty and topup_round < max_topups:
                shortfall = num_empty - len(final_data)
                to_request = min(chunk_sz, shortfall)
                logger.info(f"[Top-up Loop {topup_round + 1}] Exact deficit: {shortfall}. Requesting {to_request} missing rows.")
                try:
                    t_messages = list(base_messages)
                    if global_ctx:
                        ctx_slice = global_ctx[:min(20, len(global_ctx))]
                        compact_topup_ctx = to_compact_json(ctx_slice, all_columns)
                        t_messages.append({"role": "user", "content": prompt_manager.get_prompt("generation_global_ctx", compact_global_json=json.dumps(compact_topup_ctx))})
                        t_messages.append({"role": "assistant", "content": "Received global context references."})
                    t_messages.append({"role": "user", "content": prompt_manager.get_prompt(
                        "generation_final",
                        instructions=instructions,
                        all_columns=all_columns,
                        is_empty=(not final_data and not global_ctx),
                        num_rows_to_generate=to_request,
                        compact_payload_json="[]"
                    )})
                    
                    # Log truncated prompt for debugging
                    p_trunc = t_messages[-1]["content"][:200] + "..." if len(t_messages[-1]["content"]) > 200 else t_messages[-1]["content"]
                    logger.info(f"[LLM] [{algorithm}] Dispatching Prompt: {p_trunc}")
                    
                    t_response = self._call_local_llm(t_messages, core_model=algorithm, temperature=temperature, json_mode=True)
                    t_content = t_response["choices"][0]["message"]["content"]
                    
                    # Log truncated response
                    r_trunc = t_content[:200] + "..." if len(t_content) > 200 else t_content
                    logger.info(f"[LLM] [{algorithm}] Received Raw Response: {r_trunc}")
                    
                    try:
                        t_parsed = robust_json_loads(t_content)
                    except RobustJSONError:
                        consecutive_zeros += 1
                        logger.warning(f"Top-up {topup_round + 1}: JSON parse critically failed. Skipping round... (Consecutive failures: {consecutive_zeros})")
                        if consecutive_zeros >= 5:
                            logger.error(f"Top-up Loop halting. JSON parsing consistently failed {consecutive_zeros} times.")
                            break
                        topup_round += 1
                        continue
                        
                    if isinstance(t_parsed, dict) and "cols" in t_parsed and "data" in t_parsed:
                        topup_rows = from_compact_json(t_parsed, all_columns)
                    elif isinstance(t_parsed, list):
                        topup_rows = from_compact_json({"data": t_parsed}, all_columns) if t_parsed and isinstance(t_parsed[0], list) else t_parsed
                    else:
                        topup_rows = t_parsed.get("data", [])
                        
                    if topup_rows:
                        final_data.extend(topup_rows)
                        logger.info(f"[Top-up Loop {topup_round + 1} Success] Received {len(topup_rows)} valid rows. New Total: {len(final_data)}/{num_empty}.")
                        consecutive_zeros = 0
                    else:
                        consecutive_zeros += 1
                        logger.warning(f"[Top-up Loop {topup_round + 1} Failure] Received 0 rows. (Consecutive failures: {consecutive_zeros})")
                        if consecutive_zeros >= 5:
                            logger.error(f"Top-up Loop halting. LLM consistently failing to return data after {consecutive_zeros} sequential errors.")
                            break
                            
                except Exception as e:
                    logger.warning(f"Top-up Loop {topup_round + 1} suffered an exception error: {e}")
                    consecutive_zeros += 1
                    if consecutive_zeros >= 5:
                        break
                        
                topup_round += 1
                
            if len(final_data) < num_empty:
                logger.error(f"Generation fatally ended short: {len(final_data)}/{num_empty} rows generated. Limit reached or errors persisted.")
            else:
                logger.info(f"Algebraic Guarantee successfully satisfied: {len(final_data)}/{num_empty} rows strictly assembled.")

        # Trim to exact count in case of overshoot
        if not raw_mode and num_empty > 0 and len(final_data) > num_empty:
            final_data = final_data[:num_empty]
        accumulated_stats["delivered_rows"] = len(final_data)

        return {
            "data": final_data,
            "schema_updates": schema_updates,
            "stats": accumulated_stats
        }

    def _prepare_messages(self, meta, manual_ctx_json, all_columns):
        """Prepare initialization messages."""
        messages = []
        try:
            manual_data = json.loads(manual_ctx_json)
        except:
            manual_data = []
        compact_manual = to_compact_json(manual_data, all_columns)
        content = prompt_manager.get_prompt("generation_init", meta=meta, compact_manual_json=json.dumps(compact_manual))
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": "I understand the dataset background context. Please feed global pattern subsets next."})
        return messages

    def _process_chunk(self, subset, global_ctx, idx, total, messages, all_cols, algo, inst, temp, empty, limit, stats, depth=0):
        """Processes a target generation chunk with recursive splitting on overflow. Shielded to prevent crashes."""
        try:
            depth_str = f"[Depth {depth}] " if depth > 0 else ""
            logger.info(f"{depth_str}Processing Chunk {idx+1}/{total} ({len(subset)} targets, {empty} new).")
            
            # 1. Selection & Budgeting
            budget_info = self._calculate_budget(subset, global_ctx, all_cols, messages, inst, limit)
            ctx_rows = self._apply_rotation(global_ctx, idx, budget_info['rows_allowed'])
            
            # 2. Execution
            attempts = 0
            while attempts < 3:
                try:
                    turn_messages = list(messages)
                    if ctx_rows:
                        compact_ctx = to_compact_json(ctx_rows, all_cols)
                        ctx_content = prompt_manager.get_prompt("generation_global_ctx", compact_global_json=json.dumps(compact_ctx))
                        turn_messages.append({"role": "user", "content": ctx_content})
                        turn_messages.append({"role": "assistant", "content": "Received global context references."})
                        logger.info(f"{depth_str}Chunk {idx+1}: Injected {len(ctx_rows)} context rows into prompt.")

                    final_prompt = prompt_manager.get_prompt(
                        "generation_final", 
                        instructions=inst, 
                        all_columns=all_cols,
                        is_empty=(not subset and not ctx_rows),
                        num_rows_to_generate=empty,
                        compact_payload_json=json.dumps(to_compact_json(subset, all_cols)) if subset else "[]"
                    )
                    turn_messages.append({"role": "user", "content": final_prompt})

                    # Call LLM
                    response = self._call_llm(turn_messages, algo, temp)
                    usage = response.get("usage", {})
                    logger.info(f"{depth_str}Chunk {idx+1} [Attempt {attempts+1}] LLM Response Received. (Tokens: P={usage.get('prompt_tokens')} C={usage.get('completion_tokens')} T={usage.get('total_tokens')})")

                    stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    stats["candidates_tokens"] += usage.get("completion_tokens", 0)
                    stats["total_tokens"] += usage.get("total_tokens", 0)

                    # Parse & Fix JSON
                    content = response["choices"][0]["message"]["content"]
                    parsed = self._repair_json(content, turn_messages, algo, all_cols)
                    
                    if isinstance(parsed, dict) and "data" in parsed:
                        return from_compact_json(parsed, all_cols)
                    elif isinstance(parsed, list):
                        return from_compact_json({"data": parsed}, all_cols) if len(parsed)>0 and isinstance(parsed[0], list) else parsed
                    return parsed.get("data", [])

                except urllib.error.HTTPError as e:
                    # 400 Context Overflow Handling
                    if e.code in [400, 422]:
                        if ctx_rows:
                            logger.warning(f"{depth_str}Overflow in chunk {idx+1}. Reducing context from {len(ctx_rows)} to {int(len(ctx_rows)*0.6)}.")
                            ctx_rows = ctx_rows[:int(len(ctx_rows)*0.6)]
                        elif len(subset) > 1:
                            logger.warning(f"{depth_str}Overflow with 0 context. Splitting chunk {idx+1} (targets={len(subset)}) recursively.")
                            mid = len(subset) // 2
                            left = self._process_chunk(subset[:mid], [], idx, total, messages, all_cols, algo, inst, temp, 0, limit, stats, depth + 1)
                            right = self._process_chunk(subset[mid:], [], idx, total, messages, all_cols, algo, inst, temp, empty, limit, stats, depth + 1)
                            return left + right
                        else:
                            logger.error(f"{depth_str}Fatal Token Overflow: Target row is too large even for 0-context window of {limit} tokens.")
                            raise e
                    else: 
                        attempts += 1
                        time.sleep(2)
                except Exception as e:
                    attempts += 1
                    logger.error(f"{depth_str}Error in chunk {idx+1} (Attempt {attempts}/3): {e}")
                    if attempts >= 3:
                        logger.error(f"{depth_str}Fatal failure in chunk {idx+1}. Skipping these rows to prevent application crash.")
                        return []
                    time.sleep(2)
            return []
        except Exception as global_e:
            logger.critical(f"UNRECOVERABLE CHUNK ERROR: {global_e}. Skipping targets to persist generation.")
            return []

    def _calculate_budget(self, subset, global_ctx, all_cols, messages, inst, limit):
        """Calculates token costs and row limits strictly based on EXACT string measurement (Non-Sampling)."""
        # Heuristic Ratio for Compact JSON (brackets, quotes, and commas are dense)
        # 2.2 is a safer multiplier for structurally complex structured data.
        RATIO = 2.2 
        SAFE_THRESHOLD = int(limit * 0.85)

        # Full Overhead Calculation: Measure every character actually going into the messages
        overhead_chars = sum(len(m['content']) for m in messages) + len(inst) + (len(str(all_cols)) * 2)
        overhead_tok = int(overhead_chars / RATIO) + 400
        
        # Exact Target Calculation: No more sampling. Measure the real JSON payload.
        target_payload = [[row.get(col) for col in all_cols] for row in subset]
        target_json = json.dumps(target_payload) if subset else "[]"
        target_tok = int(len(target_json) / RATIO) + 300 # Output buffer
        
        budget_tok = SAFE_THRESHOLD - overhead_tok - target_tok
        
        # Reference Pool Calculation: Measures the REAL character weight of the entire context pool
        row_tok = 60
        if global_ctx:
            # Measure the average weight of the ACTUAL context pool by looking at all rows
            full_pool_payload = [[r.get(c) for c in all_cols] for r in global_ctx]
            full_pool_json = json.dumps(full_pool_payload)
            avg_row_char_len = len(full_pool_json) / len(global_ctx)
            row_tok = int(avg_row_char_len / RATIO) + 5
            logger.debug(f"Precision Budgeting: Measured pool avg_row_char={avg_row_char_len:.1f} | row_tok={row_tok}")
        
        allowed = max(0, int(budget_tok / row_tok))
        logger.info(f"Precision Budgeting: Threshold={SAFE_THRESHOLD} | Fixed={overhead_tok} | Target={target_tok} | AllowedRows={allowed}")
            
        return {'rows_allowed': allowed}

    def _apply_rotation(self, global_ctx, idx, allowed):
        """Samples context using a sliding window rotation."""
        if not global_ctx or allowed <= 0: return []
        offset = (idx * allowed) % len(global_ctx)
        subset = global_ctx[offset : offset + allowed]
        if len(subset) < allowed and len(global_ctx) > allowed:
            subset += global_ctx[0 : allowed - len(subset)]
        logger.debug(f"Rotation Offset: {offset} | Returning {len(subset)} rows from pool of {len(global_ctx)}.")
        return subset

    def _call_llm(self, messages, algo, temp):
        """Provider request handler."""
        return self._call_local_llm(messages, core_model=algo, temperature=temp, json_mode=True)

    def _repair_json(self, content, messages, algo, all_cols):
        """Attempts to fix malformed JSON using feedback turn."""
        try:
            return robust_json_loads(content)
        except RobustJSONError as re:
            logger.info("JSON repair turn initiated.")
            repair_prompt = prompt_manager.get_prompt("json_repair", line=re.error_data['line'], column=re.error_data['column'], error=re.error_data['error'], context=re.error_data['context'])
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": content})
            repair_messages.append({"role": "user", "content": repair_prompt})
            
            response = self._call_llm(repair_messages, algo, 0.1)
            return robust_json_loads(response["choices"][0]["message"]["content"])


class OpenAIProvider(LocalOpenAIProvider):
    """Provider connecting to the official OpenAI Chat Completions API (or any
    OpenAI-compatible endpoint reachable with a Bearer API key, e.g. Azure OpenAI
    or a proxy, via OPENAI_BASE_URL) using a user-supplied API key. Reuses all of
    LocalOpenAIProvider's chunking/budgeting/JSON-repair logic since the wire
    protocol is identical -- only auth and model discovery differ."""

    # OpenAI's /v1/models endpoint doesn't return context length, so known chat
    # models are enriched with it here (longest/most-specific prefix wins below).
    # Fallback for anything else discovered live is a conservative 128k.
    _KNOWN_CONTEXTS = {
        "gpt-4o-mini": 128000, "gpt-4o": 128000,
        "gpt-4.1-nano": 1047576, "gpt-4.1-mini": 1047576, "gpt-4.1": 1047576,
        "gpt-4-turbo": 128000, "gpt-4": 8192, "gpt-3.5-turbo": 16385,
        "o1-mini": 128000, "o1": 200000,
        "o3-mini": 200000, "o3": 200000, "o4-mini": 200000,
    }
    # Non-chat model families returned by /v1/models that can't serve this
    # pipeline's chat/completions calls -- filtered out of discovery.
    _EXCLUDE_SUBSTRINGS = (
        "embed", "whisper", "tts", "dall-e", "davinci", "babbage", "moderation",
        "audio", "realtime", "image", "search", "transcribe", "computer-use",
    )

    def __init__(self, model_name="gpt-4o-mini", api_key=None, endpoint_url=None):
        super().__init__(
            endpoint_url=endpoint_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
            model_name=model_name,
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    # Reasoning-model families (o1/o3/o4/...) only accept the default temperature (1) via
    # Chat Completions -- any other value is a 400 invalid_request_error ("Unsupported
    # value: 'temperature' does not support 0.7 with this model"). This pipeline always
    # passes an explicit temperature (0.1-0.7 depending on the call), so it has to be
    # normalized to 1 for these models rather than forwarded as-is -- see _call_local_llm.
    _REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")

    def _get_headers(self) -> dict:
        if not self.api_key:
            logger.warning("OpenAIProvider: no OPENAI_API_KEY set and no api_key supplied -- request will fail.")
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _call_local_llm(self, messages, core_model=None, temperature=0.7, json_mode=False, response_format=None):
        active_model = (core_model or self.default_model).lower()
        if active_model.startswith(self._REASONING_MODEL_PREFIXES):
            temperature = 1
        return super()._call_local_llm(messages, core_model=core_model, temperature=temperature, json_mode=json_mode, response_format=response_format)

    def get_available_models(self):
        """Query OpenAI's /v1/models for every model this key can access, filtered
        to chat-capable ones. Falls back to a hardcoded known-model list if the
        request fails (bad/missing key, offline, etc.)."""
        models_url = self.endpoint_url.replace("/chat/completions", "/models")
        try:
            req = urllib.request.Request(models_url, headers=self._get_headers())
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
            models = []
            for m in data.get("data", []):
                model_id = m.get("id") if isinstance(m, dict) else None
                if not model_id:
                    continue
                name_lower = model_id.lower()
                if any(x in name_lower for x in self._EXCLUDE_SUBSTRINGS):
                    continue
                ctx = next((v for k, v in self._KNOWN_CONTEXTS.items() if model_id.startswith(k)), 128000)
                models.append({"id": model_id, "max_context": ctx})
            if models:
                return sorted(models, key=lambda m: m["id"])
        except Exception as e:
            logger.warning(f"Could not reach {models_url} to discover OpenAI models ({e}).")
        return [{"id": mid, "max_context": ctx} for mid, ctx in self._KNOWN_CONTEXTS.items()]

    def can_handle(self, model_id: Optional[str]) -> bool:
        """Checks if the model ID belongs to OpenAI (gpt-*/o1/o3/o4/chatgpt families)."""
        if not model_id:
            return False
        name_lower = model_id.lower()
        return name_lower.startswith(("gpt-", "chatgpt", "o1", "o3", "o4"))


class GeminiProvider(BaseLLMProvider):
    """Provider connecting to Google Gemini AI API."""
    
    def __init__(self, model_name="gemini-2.0-flash"):
        self.default_model = model_name
        self._genai = None
        self._model_instances = {}

    def _get_genai(self):
        if self._genai is None:
            import google.generativeai as genai
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Warning: GEMINI_API_KEY not found in environment.")
            genai.configure(api_key=api_key)
            self._genai = genai
        return self._genai

    def _get_model(self, model_name=None):
        name = model_name or self.default_model
        if name not in self._model_instances:
            genai = self._get_genai()
            self._model_instances[name] = genai.GenerativeModel(name)
        return self._model_instances[name]

    def get_available_models(self):
        """Return supported Gemini models."""
        return [
            {"id": "gemini-2.0-flash", "max_context": 1048576},
            {"id": "gemini-1.5-flash", "max_context": 1048576},
            {"id": "gemini-1.5-pro", "max_context": 2097152},
        ]

    def can_handle(self, model_id: Optional[str]) -> bool:
        """Checks if the model ID belongs to Gemini."""
        return bool(model_id and "gemini" in model_id.lower())

    def get_dataset_description(self, df_sample, provided_metadata=None, algorithm=None):
        from services.prompt_manager import prompt_manager
        from services.logger import log_interaction
        df_sample_str = df_sample.to_string(index=False) if not df_sample.empty else "NO DATA ROWS"
        
        prompt = prompt_manager.get_prompt(
            "dataset_description",
            provided_metadata=provided_metadata,
            df_sample_str=df_sample_str,
            is_empty=df_sample.empty
        )

        try:
            model = self._get_model(algorithm)
            response = model.generate_content(prompt)
            log_interaction(f"Metadata Analysis [{algorithm or self.default_model}]", prompt, response.text)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return "Could not generate description from Gemini model."

    def detect_target_columns(self, df_sample, single_target=False, algorithm=None):
        from services.prompt_manager import prompt_manager
        from services.logger import log_interaction
        from utils.json_utils import robust_json_loads
        # 1. Generate Metadata-Only Schema String
        internal_cols = ['_row_id', 'is_context', 'is_target', 'ai_audit']
        col_meta = []
        for col in df_sample.columns:
            if col in internal_cols or col.startswith('is_ai_'):
                continue
            dtype = str(df_sample[col].dtype)
            col_meta.append(f"- {col} ({dtype})")
        
        column_metadata = "\n".join(col_meta)
        
        prompt = prompt_manager.get_prompt(
            "target_detection",
            column_metadata=column_metadata,
            single_target=single_target
        )

        try:
            model = self._get_model(algorithm)
            response = model.generate_content(
                prompt,
                generation_config=self._get_genai().types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            log_interaction(f"Target Detection [{algorithm or self.default_model}]", prompt, response.text)
            
            targets = robust_json_loads(response.text)
            if isinstance(targets, list):
                return targets
            if isinstance(targets, dict):
                if "targets" in targets and isinstance(targets["targets"], list):
                    return targets["targets"]
                if "target" in targets:
                    return [targets["target"]] if isinstance(targets["target"], str) else targets["target"]
                return [str(v) for v in targets.values() if isinstance(v, (str, int))]
            return []
        except Exception as e:
            print(f"Error calling Gemini for target detection: {e}")
            return []

    def generate_bulk_data(self, rows_payload, context, all_columns, all_fields, algorithm, instructions, temperature, num_empty=0, chunk_size=30, rules_metadata=None, enforce_rules=True):
        import json
        import time
        from services.prompt_manager import prompt_manager
        from utils.json_utils import to_compact_json, from_compact_json, robust_json_loads, RobustJSONError
        from services.rule_validator import RuleValidator

        # Bridge: display header names → internal field names
        h_to_f_bridge = dict(zip(all_columns, all_fields))

        # Compact payload and context for Gemini
        compact_payload = to_compact_json(rows_payload, all_columns)

        compact_ctx = {}
        if isinstance(context, dict):
            compact_ctx = {
                "metadata": context.get("metadata"),
                "manual_ctx": to_compact_json(json.loads(context.get("manual_ctx", "[]")), all_columns) if context.get("manual_ctx") != "[]" else [],
                "global_ctx": to_compact_json(context.get("global_ctx", []), all_columns)
            }

        prompt = prompt_manager.get_prompt(
            "generation_final",
            instructions=instructions,
            all_columns=all_columns,
            is_empty=(not rows_payload and not compact_ctx.get("manual_ctx") and not compact_ctx.get("global_ctx")),
            num_rows_to_generate=num_empty,
            compact_payload_json=json.dumps(compact_payload)
        )
        gen_stats = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0, "rule_violations": 0, "repair_attempts": 0}
        try:
            max_retries = 3
            model = self._get_model(algorithm)
            genai = self._get_genai()
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep((attempt + 1) * 5)
                    else:
                        time.sleep(1.0)

                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=temperature
                        )
                    )
                    usage = response.usage_metadata
                    gen_stats["prompt_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
                    gen_stats["candidates_tokens"] += getattr(usage, "candidates_token_count", 0) or 0
                    gen_stats["total_tokens"] += getattr(usage, "total_token_count", 0) or 0

                    # 1. Parse JSON (with repair if needed)
                    try:
                        parsed_data = robust_json_loads(response.text)
                    except RobustJSONError as re:
                        gen_stats["repair_attempts"] += 1
                        repair_prompt = prompt_manager.get_prompt(
                            "json_repair",
                            line=re.error_data['line'],
                            column=re.error_data['column'],
                            error=re.error_data['error'],
                            context=re.error_data['context']
                        )
                        response_repair = model.generate_content(
                            prompt + f"\n\nPREVIOUS MALFORMED OUTPUT:\n{response.text}\n\nFIX REQUEST:\n{repair_prompt}",
                            generation_config=genai.types.GenerationConfig(
                                response_mime_type="application/json",
                                temperature=0.1
                            )
                        )
                        parsed_data = robust_json_loads(response_repair.text)

                    # 2. Unpack response into final_rows
                    schema_updates = {}
                    if isinstance(parsed_data, dict):
                        schema_updates = parsed_data.get("schema_updates", {})
                        returned_cols = parsed_data.get("cols", [])
                        if returned_cols and len(returned_cols) == len(all_columns):
                            for i, rc in enumerate(returned_cols):
                                if str(rc).strip().lower() != str(all_columns[i]).strip().lower():
                                    schema_updates[all_fields[i]] = rc
                        final_rows = from_compact_json(parsed_data, all_fields, h_to_f=h_to_f_bridge)
                    elif isinstance(parsed_data, list):
                        final_rows = from_compact_json({"data": parsed_data}, all_fields, h_to_f=h_to_f_bridge) if parsed_data and isinstance(parsed_data[0], list) else parsed_data
                    else:
                        raise ValueError("Gemini returned unexpected structure")

                    # 3. Rule Enforcement (Self-Correction)
                    if enforce_rules and rules_metadata and final_rows:
                        if isinstance(rules_metadata, dict) and "integrity" in rules_metadata:
                            active_rules = list(rules_metadata.get("integrity", []))
                            if rules_metadata.get("enforce_boundaries"):
                                active_rules.extend(rules_metadata.get("boundaries", []))
                            if rules_metadata.get("enforce_user"):
                                active_rules.extend(rules_metadata.get("user", []))
                        else:
                            active_rules = rules_metadata

                        pydantic_model = RuleValidator.create_pydantic_model(active_rules)
                        if pydantic_model:
                            for retry in range(2):
                                violations = RuleValidator.validate_batch(final_rows, pydantic_model)
                                if not violations:
                                    break

                                gen_stats["rule_violations"] += len(violations)
                                feedback = RuleValidator.format_violations_for_llm(violations)
                                response_corr = model.generate_content(
                                    prompt + f"\n\nPREVIOUS OUTPUT:\n{response.text}\n\n{feedback}\n\n" + prompt_manager.get_prompt("generation_fix"),
                                    generation_config=genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
                                )
                                try:
                                    parsed_corr = robust_json_loads(response_corr.text)
                                    if isinstance(parsed_corr, dict) and "data" in parsed_corr:
                                        final_rows = from_compact_json(parsed_corr, all_fields, h_to_f=h_to_f_bridge)
                                    elif isinstance(parsed_corr, list):
                                        final_rows = from_compact_json({"data": parsed_corr}, all_fields, h_to_f=h_to_f_bridge) if parsed_corr and isinstance(parsed_corr[0], list) else parsed_corr
                                except:
                                    break

                    # 4. Top-up Guarantee
                    if num_empty > 0 and len(final_rows) < num_empty:
                        topup_round = 0
                        while len(final_rows) < num_empty and topup_round < 5:
                            shortfall = num_empty - len(final_rows)
                            topup_prompt = prompt_manager.get_prompt(
                                "generation_final",
                                instructions=instructions,
                                all_columns=all_columns,
                                is_empty=False,
                                num_rows_to_generate=min(chunk_size, shortfall),
                                compact_payload_json="[]"
                            )
                            try:
                                t_resp = model.generate_content(topup_prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json", temperature=temperature))
                                try:
                                    t_parsed = robust_json_loads(t_resp.text)
                                except RobustJSONError as r_err:
                                    logger.info("Gemini Top-up: JSON Malformatted. Triggering repair...")
                                    rep_prompt = prompt_manager.get_prompt(
                                        "json_repair",
                                        line=r_err.error_data['line'],
                                        column=r_err.error_data['column'],
                                        error=r_err.error_data['error'],
                                        context=r_err.error_data['context']
                                    )
                                    t_resp = model.generate_content(rep_prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1))
                                    t_parsed = robust_json_loads(t_resp.text)

                                if isinstance(t_parsed, dict) and "data" in t_parsed:
                                    t_rows = from_compact_json(t_parsed, all_fields, h_to_f=h_to_f_bridge)
                                elif isinstance(t_parsed, list):
                                    t_rows = from_compact_json({"data": t_parsed}, all_fields, h_to_f=h_to_f_bridge) if t_parsed and isinstance(t_parsed[0], list) else t_parsed
                                else:
                                    t_rows = []
                                if t_rows:
                                    final_rows.extend(t_rows)
                            except:
                                break
                            topup_round += 1

                    # 5. Trim to exact count in case of overshoot
                    if num_empty > 0 and len(final_rows) > num_empty:
                        final_rows = final_rows[:num_empty]

                    return {
                        "data": final_rows,
                        "schema_updates": schema_updates,
                        "stats": gen_stats
                    }
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        continue
                    else:
                        raise e
        except Exception as e:
            print(f"Error in Gemini Client: {e}")
            return None

    def call_llm(self, messages: list, model_id: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
        """Implementation for Gemini API."""
        prompt = messages[-1]["content"] if messages else ""
        try:
            model = self._get_model(model_id)
            response = model.generate_content(prompt, generation_config=self._get_genai().types.GenerationConfig(temperature=temperature))
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"


# ==========================================
class AnthropicProvider(LocalOpenAIProvider):
    """Provider for Anthropic's Claude models through the official Anthropic SDK.

    Reuses LocalOpenAIProvider's chunking, budgeting and repair logic by translating each
    request to the Messages API and the response back to the Chat Completions shape that
    logic reads (choices[0].message.content and usage.prompt/completion/total_tokens).
    Settings, applied identically to every Claude model so that only the model differs:
      - no temperature: the 5.5 models reject non-default sampling parameters, so it is
        left at each model's default for all four;
      - thinking off: omitted on Haiku 4.5 / Sonnet 4.6 (no thinking by default),
        {"type": "disabled"} on Haiku 5.5 and {"type": "between_tools"} on Sonnet 5.5
        (which accepts no "disabled"); with no tools, the latter also means no thinking.
    The base URL is fixed to the public API (ANTHROPIC_BASE_URL may point elsewhere in
    some environments) unless TABULLM_ANTHROPIC_BASE_URL overrides it.
    """

    _KNOWN_CONTEXTS = {
        "claude-haiku-4-5": 200000, "claude-sonnet-4-6": 1000000,
        "claude-haiku-5-5": 1000000, "claude-sonnet-5-5": 1000000,
    }
    _MAX_TOKENS = 16000

    def __init__(self, model_name="claude-haiku-5-5", api_key=None):
        super().__init__(endpoint_url="https://api.anthropic.com/v1/messages", model_name=model_name)
        # TABULLM_ANTHROPIC_API_KEY first: Claude Code cloud environments reserve
        # ANTHROPIC_API_KEY for their own authentication and do not pass it through.
        self.api_key = api_key or os.getenv("TABULLM_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=os.getenv("TABULLM_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
                timeout=float(self.timeout),
                max_retries=self._max_retries(),
            )
        return self._client

    @staticmethod
    def _thinking_for(model: str) -> Optional[dict]:
        if model.startswith("claude-sonnet-5-5"):
            return {"type": "between_tools"}
        if model.startswith(("claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5")):
            return {"type": "disabled"}
        return None

    @staticmethod
    def _to_anthropic_messages(messages: list) -> "tuple[str, list]":
        """System messages go to the top-level system prompt; the rest keep their order."""
        system_parts, out = [], []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(str(m.get("content", "")))
            else:
                out.append({"role": m["role"], "content": str(m.get("content", ""))})
        return "\n\n".join(system_parts), out

    def _call_local_llm(self, messages, core_model=None, temperature=0.7, json_mode=False, response_format=None):
        import anthropic
        model = core_model or self.default_model
        system, msgs = self._to_anthropic_messages(messages)
        kwargs = {"model": model, "max_tokens": self._MAX_TOKENS, "messages": msgs}
        if system:
            kwargs["system"] = system
        thinking = self._thinking_for(model)
        if thinking:
            kwargs["thinking"] = thinking
        if response_format is not None:
            schema = response_format.get("json_schema", {}).get("schema", response_format.get("schema"))
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        logger.info(f"--- [SERVICE] CALLING ANTHROPIC: {model} (Timeout: {self.timeout}s) ---")
        try:
            resp = self._get_client().messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            # Same contract as the HTTP providers: callers handle 400/422 themselves.
            raise urllib.error.HTTPError(self.endpoint_url, e.status_code, str(e), None, None) from e
        except anthropic.APIConnectionError as e:
            raise ConnectionError(str(e)) from e

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if resp.stop_reason == "refusal":
            logger.warning(f"Anthropic refusal on {model}: {getattr(resp, 'stop_details', None)}")
        u = resp.usage
        prompt_tok = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(u, "cache_read_input_tokens", 0) or 0)
        completion_tok = u.output_tokens or 0
        return {
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": resp.stop_reason}],
            "usage": {"prompt_tokens": prompt_tok, "completion_tokens": completion_tok,
                      "total_tokens": prompt_tok + completion_tok},
        }

    def get_available_models(self):
        return [{"id": mid, "max_context": ctx} for mid, ctx in self._KNOWN_CONTEXTS.items()]

    def can_handle(self, model_id: Optional[str]) -> bool:
        return bool(model_id) and model_id.lower().startswith("claude-")


# MULTI-PROVIDER REGISTRY CONFIGURATION
# ==========================================

class LLMRegistry:
    def __init__(self):
        self.providers = [
            GeminiProvider(),
            OpenAIProvider(),
            AnthropicProvider(),
            LocalOpenAIProvider() # Fallback
        ]

    def get_provider(self, model_id: Optional[str]) -> BaseLLMProvider:
        for p in self.providers:
            if p.can_handle(model_id):
                return p
        return self.providers[-1]

    def get_all_models(self):
        all_models = []
        for p in self.providers:
            try:
                models = p.get_available_models()
                if models:
                    all_models.extend(models)
            except Exception as e:
                print(f"Error fetching models from provider {p.__class__.__name__}: {e}")
        return all_models

# Singleton instances
registry = LLMRegistry()

# Module-level convenience functions mirroring the old API (backward compatibility)
def get_dataset_description(df_sample, provided_metadata=None, algorithm=None):
    p = registry.get_provider(algorithm)
    return p.get_dataset_description(df_sample, provided_metadata, algorithm=algorithm)

def generate_bulk_data(rows_payload, context, all_columns, all_fields, algorithm, instructions, temperature, num_empty=0, chunk_size=30, rules_metadata=None, enforce_rules=True, raw_mode=False, response_format=None):
    p = registry.get_provider(algorithm)
    if raw_mode:
        # Only the OpenAI-compatible providers implement the no-controller ablation.
        return p.generate_bulk_data(rows_payload, context, all_columns, all_fields, algorithm, instructions, temperature, num_empty, chunk_size, rules_metadata, enforce_rules, raw_mode=True, response_format=response_format)
    return p.generate_bulk_data(rows_payload, context, all_columns, all_fields, algorithm, instructions, temperature, num_empty, chunk_size, rules_metadata, enforce_rules)

def detect_target_columns(df_sample, single_target=False, algorithm=None):
    p = registry.get_provider(algorithm)
    return p.detect_target_columns(df_sample, single_target=single_target, algorithm=algorithm)

def call_llm(messages: list, model_id: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
    p = registry.get_provider(model_id)
    return p.call_llm(messages, model_id=model_id, temperature=temperature)

def get_available_models():
    return registry.get_all_models()