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


def generate_bulk_data(rows_payload, context, all_columns, algorithm, user_instructions=None, temperature=0.3):
    """
    Generates bulk data, incorporating user-defined instructions and temperature.
    """
    instruction_block = f"\nCRITICAL USER INSTRUCTIONS: {user_instructions}\n" if user_instructions else ""

    prompt = f"""
    Context: {context}
    Columns: {all_columns}
    {instruction_block}

    Task: Fill the 'null' values in the following JSON array logically.
    Input JSON: {json.dumps(rows_payload)}

    Return ONLY a valid JSON array.
    """
    try:
        # We pass the temperature into the generation config
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=temperature
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error: {e}")
        return None