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


def generate_bulk_data(rows_payload, dataset_context, all_columns, algorithm="Bulk"):
    """
    Takes an array of row dictionaries, sends them to Gemini in one shot,
    and forces the LLM to return a completed JSON array.
    """
    prompt = f"""
    Dataset Context: {dataset_context}
    Columns: {', '.join(all_columns)}

    You are an intelligent data entry assistant. 
    I am providing a JSON array of rows. Some values are null.
    Fill in the missing null values logically based on the context and the other values in the row.
    If a row is entirely null, invent a realistic, consistent entity that fits the dataset.

    Input JSON:
    {json.dumps(rows_payload, indent=2)}

    Return ONLY a valid JSON array of objects with the exact same structure, where all missing values have been filled.
    """

    try:
        # Force the model to output structured JSON
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        result = response.text.strip()
        log_interaction(algorithm, prompt, result)

        # Parse the JSON string back into a Python list of dictionaries
        return json.loads(result)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        log_interaction(algorithm, prompt, str(e), status="error")
        return None