import os
import time
import json
import urllib.request
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from services.logger import log_interaction
from utils.logging_config import logger
from utils.json_utils import robust_json_loads, RobustJSONError, to_compact_json, from_compact_json
from services.prompt_manager import prompt_manager

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    def get_dataset_description(self, df_sample, provided_metadata=None, algorithm=None):
        pass

    @abstractmethod
    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0, chunk_size=30):
        pass

    @abstractmethod
    def detect_target_columns(self, df_sample, single_target=False, algorithm=None):
        pass

    @abstractmethod
    def get_available_models(self):
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

    def _call_local_llm(self, messages, core_model=None, temperature=0.7, json_mode=False):
        active_model = core_model if core_model else self.default_model
        payload = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature
        }
        
        logger.info(f"--- [SERVICE] CALLING LOCAL LLM: {active_model} (Timeout: {self.timeout}s) ---")
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.endpoint_url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw_response = response.read()
                result = json.loads(raw_response.decode('utf-8'))
                return result
        except (OSError, ConnectionResetError) as socket_err:
            if "[WinError 10038]" in str(socket_err):
                logger.error(f"Critical Socket Failure (WinError 10038). The local server connection was dropped. Error: {socket_err}")
            else:
                logger.error(f"Local LLM Socket Connection Reset: {socket_err}")
            raise socket_err
        except Exception as e:
            logger.error(f"Unexpected error during LLM request: {e}")
            raise e

    def get_dataset_description(self, df_sample, provided_metadata=None, algorithm=None):
        prompt = prompt_manager.get_prompt(
            "dataset_description",
            provided_metadata=provided_metadata,
            df_sample_str=df_sample.to_string(index=False)
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
        col_meta = []
        for col in df_sample.columns:
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

    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0, chunk_size=30):
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
        
        if not subsets and num_empty > 0:
            subsets = [[]]

        # 4. Token Budgeting Setup
        models_info = self.get_available_models()
        model_ctx_limit = 4096
        for m in models_info:
            if m["id"] == algorithm:
                model_ctx_limit = m["max_context"]
                break
        
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
        accumulated_stats = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
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

                tasks = []
                if subset:
                    compact_subset = to_compact_json(subset, all_columns)
                    tasks.append(f"- Complete these specific PARTIAL TARGET ROWS in compact format (Chunk {subset_idx+1}):\n{json.dumps(compact_subset)}")
                if subset_idx == len(subsets) - 1 and num_empty > 0:
                    tasks.append(f"- Generate {num_empty} entirely NEW rows following the fed distribution patterns.")
                
                final_prompt = prompt_manager.get_prompt(
                    "generation_final",
                    instructions=instructions,
                    all_columns=all_columns,
                    task_str="\n".join(tasks)
                )
                messages.append({"role": "user", "content": final_prompt})

                try:
                    response = self._call_local_llm(messages, core_model=algorithm, temperature=temperature, json_mode=True)
                    content = response["choices"][0]["message"]["content"]
                    usage = response.get("usage", {})

                    # Accumulate stats
                    accumulated_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    accumulated_stats["candidates_tokens"] += usage.get("completion_tokens", 0)
                    accumulated_stats["total_tokens"] += usage.get("total_tokens", 0)

                    # Use unified robust_json_loads with repair step
                    try:
                        parsed_content = robust_json_loads(content)
                    except RobustJSONError as re:
                        logger.info(f"Initial JSON parsing failed. Attempting focused repair... Error: {re.error_data['error']}")
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

                    if isinstance(parsed_content, dict) and "cols" in parsed_content and "data" in parsed_content:
                        subset_data = from_compact_json(parsed_content, all_columns)
                    elif isinstance(parsed_content, list):
                        subset_data = from_compact_json({"data": parsed_content}, all_columns) if len(parsed_content)>0 and isinstance(parsed_content[0], list) else parsed_content
                    else:
                        subset_data = parsed_content.get("data", [])

                    final_data.extend(subset_data)
                    break # Success! Continue onto next chunk subset

                except urllib.error.HTTPError as e:
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
                    logger.error(f"Fatally failed chunk index {subset_idx+1}: {e}")
                    raise e

        return {
            "data": final_data,
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

                    tasks = []
                    if subset:
                        compact_subset = to_compact_json(subset, all_cols)
                        tasks.append(f"- Complete these specific PARTIAL TARGET ROWS in compact format (Chunk {idx+1}):\n{json.dumps(compact_subset)}")
                    if empty > 0:
                        tasks.append(f"- Generate {empty} entirely NEW rows following the fed distribution patterns.")
                    
                    final_prompt = prompt_manager.get_prompt("generation_final", instructions=inst, all_columns=all_cols, task_str="\n".join(tasks))
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


class GeminiProvider(BaseLLMProvider):
    """Provider connecting to Google Gemini AI API."""
    
    def __init__(self, model_name="gemini-2.5-flash"):
        import google.generativeai as genai
        load_dotenv()
        # Fallback to key in environment if custom key is not supplied.
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyA9k_qr9yyFLhZAWGboITSHLCYjIZlFaRA")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai

    def get_dataset_description(self, df_sample, provided_metadata=None):
        prompt = prompt_manager.get_prompt(
            "dataset_description",
            provided_metadata=provided_metadata,
            df_sample_str=df_sample.to_string(index=False)
        )

        try:
            response = self.model.generate_content(prompt)
            log_interaction("Metadata Analysis", prompt, response.text)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return "Could not generate description from Gemini model."

    def detect_target_columns(self, df_sample, single_target=False):
        # 1. Generate Metadata-Only Schema String
        col_meta = []
        for col in df_sample.columns:
            dtype = str(df_sample[col].dtype)
            col_meta.append(f"- {col} ({dtype})")
        
        column_metadata = "\n".join(col_meta)
        
        prompt = prompt_manager.get_prompt(
            "target_detection",
            column_metadata=column_metadata,
            single_target=single_target
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            log_interaction("Target Detection", prompt, response.text)
            
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
            logger.error(f"Error calling Gemini for target detection: {e}")
            return []

    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0, chunk_size=30):
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
            task_str=f"Complete partial rows: {json.dumps(compact_payload)}\nGenerate {num_empty} new rows."
        )
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep((attempt + 1) * 5)
                    else:
                        time.sleep(1.5)

                    response = self.model.generate_content(
                        prompt,
                        generation_config=self.genai.types.GenerationConfig(
                            response_mime_type="application/json",
                            temperature=temperature
                        )
                    )
                    usage = response.usage_metadata
                    try:
                        parsed_data = robust_json_loads(response.text)
                    except RobustJSONError as re:
                        logger.info(f"Gemini JSON parsing failed. Attempting focused repair... Error: {re.error_data['error']}")
                        repair_prompt = prompt_manager.get_prompt(
                            "json_repair",
                            line=re.error_data['line'],
                            column=re.error_data['column'],
                            error=re.error_data['error'],
                            context=re.error_data['context']
                        )
                        response_repair = self.model.generate_content(
                            prompt + f"\n\nPREVIOUS MALFORMED OUTPUT:\n{response.text}\n\nFIX REQUEST:\n{repair_prompt}",
                            generation_config=self.genai.types.GenerationConfig(
                                response_mime_type="application/json",
                                temperature=0.1
                            )
                        )
                        parsed_data = robust_json_loads(response_repair.text)
                    
                    if isinstance(parsed_data, dict) and "cols" in parsed_data and "data" in parsed_data:
                        final_rows = from_compact_json(parsed_data, all_columns)
                    elif isinstance(parsed_data, dict) and "data" in parsed_data:
                         # Handle list of lists without cols key fallback
                         first_row = parsed_data["data"][0] if parsed_data["data"] else None
                         if isinstance(first_row, list):
                             final_rows = from_compact_json(parsed_data, all_columns)
                         else:
                             final_rows = parsed_data["data"]
                    elif isinstance(parsed_data, list):
                        first_row = parsed_data[0] if parsed_data else None
                        if isinstance(first_row, list):
                            final_rows = from_compact_json({"data": parsed_data}, all_columns)
                        else:
                            final_rows = parsed_data
                    else:
                        raise ValueError("Gemini returned unexpected structure")

                    return {
                        "data": final_rows,
                        "stats": {
                            "prompt_tokens": usage.prompt_token_count,
                            "candidates_tokens": usage.candidates_token_count,
                            "total_tokens": usage.total_token_count
                        }
                    }
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        logger.warning(f"Rate limit exceeded (429). Retrying in {(attempt + 1) * 5}s... ({attempt + 1}/{max_retries})")
                        continue
                    else:
                        raise e
        except Exception as e:
            logger.error(f"Error in Gemini Client: {e}")
            return None


# ==========================================
# DEFAULT PROVIDER CONFIGURATION
# ==========================================

# By default, use the Local LM Studio model
provider = LocalOpenAIProvider()
# If you ever want to switch back to Gemini, simply uncomment the line below:
# provider = GeminiProvider()

# Module-level convenience functions mirroring the old API (backward compatibility)
def get_dataset_description(df_sample, provided_metadata=None, algorithm=None):
    return provider.get_dataset_description(df_sample, provided_metadata, algorithm=algorithm)

def generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0, chunk_size=30):
    return provider.generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty, chunk_size)

def detect_target_columns(df_sample, single_target=False, algorithm=None):
    return provider.detect_target_columns(df_sample, single_target=single_target, algorithm=algorithm)

