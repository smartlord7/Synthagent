import os
import time
import json
import urllib.request
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from services.logger import log_interaction
from core import logger

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""
    
    @abstractmethod
    def get_dataset_description(self, df_sample, provided_metadata=None):
        pass

    @abstractmethod
    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0):
        pass


class LocalOpenAIProvider(BaseLLMProvider):
    """Provider connecting to an OpenAI-compatible Local Server (e.g. LM Studio)."""
    
    def __init__(self, endpoint_url="http://127.0.0.1:1234/v1/chat/completions", model_name="google/gemma-3-4b"):
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    def _call_local_llm(self, prompt, temperature=0.7, json_mode=False):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        # NOTE: Many local models (like Gemma) do not support the strict `json_object` 
        # response_format flag in LM Studio and will return a 400 Bad Request.
        # We rely on the prompt instructions and fallback parsing instead.
        # if json_mode:
        #     payload["response_format"] = {"type": "json_object"}
        
        logger.info(f"Sending Local LLM Request for model {self.model_name}...")
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(self.endpoint_url, data=data, headers={'Content-Type': 'application/json'})
        
        # Added timeout=90 to account for slower local model processing, preventing silent app freezes.
        with urllib.request.urlopen(req, timeout=90) as response:
            logger.info("Local LLM response headers received, reading content...")
            raw_response = response.read()
            logger.info("Local LLM content read complete.")
            result = json.loads(raw_response.decode('utf-8'))
            return result

    def get_dataset_description(self, df_sample, provided_metadata=None):
        prompt = f"""
        Analyze this dataset sample and provide a concise summary of what this data represents.
        Metadata Provided: {provided_metadata if provided_metadata else "None"}
        Data Sample:
        {df_sample.to_string(index=False)}

        Return a 2-sentence description of the dataset context and the relationship between columns.
        """
        try:
            response = self._call_local_llm(prompt, temperature=0.3)
            content = response["choices"][0]["message"]["content"]
            log_interaction("Metadata Analysis", prompt, content)
            return content
        except Exception as e:
            logger.error(f"Error calling local LLM for description: {e}")
            return "Could not generate description from local model."

    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0):
        # Dynamically build tasks to avoid confusing the model with empty items (which triggers looping)
        tasks = []
        task_counter = 1
        
        if rows_payload and len(rows_payload) > 0:
            tasks.append(f"{task_counter}. Complete the data for these specific PARTIAL TARGET ROWS:\n{json.dumps(rows_payload)}")
            task_counter += 1
            
        if num_empty > 0:
            tasks.append(f"{task_counter}. Generate {num_empty} entirely NEW rows from scratch that follow the patterns, style, and distributions found in the SYSTEM CONTEXT.")
            
        task_str = "\n\n".join(tasks)

        prompt = f"""
        SYSTEM CONTEXT: {context}
        USER INSTRUCTIONS: {instructions}
        DATA STRUCTURE: The columns are {all_columns}. 
        CRITICAL: For numeric columns, return RAW NUMBERS only (no currency symbols, no commas).
        MUST Output a JSON object containing a strictly formatted array inside a key "data"
        like this: {{"data": [{{...}}, {{...}}]}}

        TASK:
        {task_str}

        Return the final JSON object containing all the completed and newly generated rows.
        """
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep((attempt + 1) * 2)

                    response = self._call_local_llm(prompt, temperature=temperature, json_mode=True)
                    content = response["choices"][0]["message"]["content"]
                    usage = response.get("usage", {})
                    
                    try:
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, dict) and "data" in parsed_content:
                            final_data = parsed_content["data"]
                        elif isinstance(parsed_content, list):
                            final_data = parsed_content
                        else:
                            raise ValueError("Unexpected JSON structure")
                    except json.JSONDecodeError:
                        cleaned = content.replace("```json", "").replace("```", "").strip()
                        parsed_content = json.loads(cleaned)
                        if isinstance(parsed_content, dict) and "data" in parsed_content:
                            final_data = parsed_content["data"]
                        else:
                            final_data = parsed_content

                    if not isinstance(final_data, list):
                        raise ValueError("Final parsed data must be a list of dictionaries.")

                    return {
                        "data": final_data,
                        "stats": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "candidates_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    }
                except Exception as e:
                    logger.warning(f"Error calling local LLM. Retrying... ({attempt + 1}/{max_retries}) - {e}")
                    if attempt == max_retries - 1:
                        raise e
        except Exception as e:
            logger.error(f"Error in Local Client: {e}")
            return None


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
        prompt = f"""
        Analyze this dataset sample and provide a concise summary of what this data represents.
        Metadata Provided: {provided_metadata if provided_metadata else "None"}
        Data Sample:
        {df_sample.to_string(index=False)}

        Return a 2-sentence description of the dataset context and the relationship between columns.
        """
        try:
            response = self.model.generate_content(prompt)
            log_interaction("Metadata Analysis", prompt, response.text)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return "Could not generate description from Gemini model."

    def generate_bulk_data(self, rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0):
        prompt = f"""
        SYSTEM CONTEXT: {context}
        USER INSTRUCTIONS: {instructions}
        DATA STRUCTURE: The columns are {all_columns}. 
        CRITICAL: For numeric columns, return RAW NUMBERS only (no currency symbols, no commas).

        TASK:
        1. Complete the data for these specific PARTIAL TARGET ROWS:
        {json.dumps(rows_payload)}

        2. Additionally, generate {num_empty} entirely NEW rows from scratch that follow the patterns, 
           style, and distributions found in the SYSTEM CONTEXT.

        Return a single valid JSON array containing both the completed partial rows and the new rows.
        """
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
                    return {
                        "data": json.loads(response.text),
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

def generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0):
    return provider.generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty)
