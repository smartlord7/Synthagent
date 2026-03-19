import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from services.logger import log_interaction

load_dotenv()
# Keep your key in the .env file instead of hardcoding it!
genai.configure(api_key="AIzaSyA9k_qr9yyFLhZAWGboITSHLCYjIZlFaRA")
model = genai.GenerativeModel('gemini-2.5-flash')


def get_dataset_description(df_sample, provided_metadata=None):
    prompt = f"""
    Analyze this dataset sample and provide a concise summary of what this data represents.
    Metadata Provided: {provided_metadata if provided_metadata else "None"}
    Data Sample:
    {df_sample.to_string(index=False)}

    Return a 2-sentence description of the dataset context and the relationship between columns.
    """
    response = model.generate_content(prompt)
    log_interaction("Metadata Analysis", prompt, response.text)
    return response.text


def generate_bulk_data(rows_payload, context, all_columns, algorithm, instructions, temperature, num_empty=0):
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
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
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
        logger.error(f"Error in Gemini Client: {e}")
        return None