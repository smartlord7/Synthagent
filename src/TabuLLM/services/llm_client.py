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
    def get_dataset_description(self, df_sample, provided_metadata=None):
        pass

    @abstractmethod
    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0, chunk_size=30):
        pass

    @abstractmethod
    def detect_target_columns(self, df_sample, single_target=False):
        pass



class LocalOpenAIProvider(BaseLLMProvider):
    """Provider connecting to an OpenAI-compatible Local Server (e.g. LM Studio)."""
    
    def __init__(self, endpoint_url="http://127.0.0.1:1234/v1/chat/completions", model_name="google/gemma-3-4b"):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        # Load timeout from env, default to 300 seconds (5 minutes)
        try:
            self.timeout = int(os.getenv("LOCAL_LLM_TIMEOUT", 300))
        except (ValueError, TypeError):
            self.timeout = 300

    def _call_local_llm(self, messages, temperature=0.7, json_mode=False):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }
        
        logger.info(f"Sending Local LLM Request with {len(messages)} messages for model {self.model_name}...")
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.endpoint_url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            raw_response = response.read()
            result = json.loads(raw_response.decode('utf-8'))
            return result

    def get_dataset_description(self, df_sample, provided_metadata=None):
        prompt = prompt_manager.get_prompt(
            "dataset_description",
            provided_metadata=provided_metadata,
            df_sample_str=df_sample.to_string(index=False)
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._call_local_llm(messages, temperature=0.3)
            content = response["choices"][0]["message"]["content"]
            log_interaction("Metadata Analysis", prompt, content)
            return content
        except Exception as e:
            logger.error(f"Error calling local LLM for description: {e}")
            return "Could not generate description from local model."

    def detect_target_columns(self, df_sample, single_target=False):
        prompt = prompt_manager.get_prompt(
            "target_detection",
            df_sample_str=df_sample.to_string(index=False),
            single_target=single_target
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._call_local_llm(messages, temperature=0.1, json_mode=True)
            content = response["choices"][0]["message"]["content"]
            log_interaction("Target Detection", prompt, content)
            
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

        # 3. Stream Global Context In compact frame
        if global_ctx and len(global_ctx) > 0:
            compact_global = to_compact_json(global_ctx, all_columns)
            global_content = prompt_manager.get_prompt(
                "generation_global_ctx",
                compact_global_json=json.dumps(compact_global)
            )
            base_messages.append({"role": "user", "content": global_content})
            base_messages.append({"role": "assistant", "content": "Received global context references."})

        # --- CHUNKED TARGET PROCESSING ---
        chunk_sz = chunk_size if chunk_size else 30
        subsets = [rows_payload[i:i+chunk_sz] for i in range(0, len(rows_payload), chunk_sz)] if rows_payload else []
        
        # If only adding empty rows
        if not subsets and num_empty > 0:
            subsets = [[]]

        final_data = []
        accumulated_stats = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

        logger.info(f"Chunk splitting rows payload into {len(subsets)} iterative subsets.")

        for subset_idx, subset in enumerate(subsets):
            messages = list(base_messages)
            tasks = []
            if subset:
                compact_subset = to_compact_json(subset, all_columns)
                tasks.append(f"- Complete these specific PARTIAL TARGET ROWS in compact format (Chunk {subset_idx+1}):\n{json.dumps(compact_subset)}")
            if subset_idx == len(subsets) - 1 and num_empty > 0:
                tasks.append(f"- Generate {num_empty} entirely NEW rows following the fed distribution patterns.")
            
            task_str = "\n".join(tasks)

            final_prompt = prompt_manager.get_prompt(
                "generation_final",
                instructions=instructions,
                all_columns=all_columns,
                task_str=task_str
            )
            messages.append({"role": "user", "content": final_prompt})

            try:
                max_retries = 3
                subset_data = []
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            time.sleep((attempt + 1) * 2)

                        response = self._call_local_llm(messages, temperature=temperature, json_mode=True)
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
                            
                            response_repair = self._call_local_llm(messages_repair, temperature=0.1, json_mode=True)
                            content_repair = response_repair["choices"][0]["message"]["content"]
                            parsed_content = robust_json_loads(content_repair)
                        except Exception as e:
                            logger.error(f"Failed to parse content from Local LLM: {content[:200]}... Error: {e}")
                            raise e

                        if isinstance(parsed_content, dict) and "cols" in parsed_content and "data" in parsed_content:
                            subset_data = from_compact_json(parsed_content, all_columns)
                        elif isinstance(parsed_content, dict) and "data" in parsed_content and isinstance(parsed_content["data"], list):
                            # Fallback if LLM returns list of lists without 'cols' key
                            first_row = parsed_content["data"][0] if parsed_content["data"] else None
                            if isinstance(first_row, list):
                                subset_data = from_compact_json(parsed_content, all_columns)
                            else:
                                subset_data = parsed_content["data"]
                        elif isinstance(parsed_content, list):
                            # If LLM returns raw list of lists or list of dicts
                            first_row = parsed_content[0] if parsed_content else None
                            if isinstance(first_row, list):
                                subset_data = from_compact_json({"data": parsed_content}, all_columns)
                            else:
                                subset_data = parsed_content
                        else:
                            raise ValueError("Unexpected JSON structure")

                        if isinstance(subset_data, list):
                            final_data.extend(subset_data)
                            break # Success! Continue onto next chunk subset
                        else:
                            raise ValueError("Subset data is not a list")

                    except Exception as e:
                        logger.warning(f"Error on chunk {subset_idx+1} local LLM call. Retrying... ({attempt + 1}/{max_retries}) - {e}")
                        if attempt == max_retries - 1:
                            raise e
            except Exception as e:
                logger.error(f"Fatally failed chunk index {subset_idx+1}: {e}")
                if not final_data:  # if completely nothing generated yet
                    return None

        return {
            "data": final_data,
            "stats": accumulated_stats
        }



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
        prompt = prompt_manager.get_prompt(
            "target_detection",
            df_sample_str=df_sample.to_string(index=False),
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
def get_dataset_description(df_sample, provided_metadata=None):
    return provider.get_dataset_description(df_sample, provided_metadata)

def generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0, chunk_size=30):
    return provider.generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty, chunk_size)

def detect_target_columns(df_sample, single_target=False):
    return provider.detect_target_columns(df_sample, single_target=single_target)

